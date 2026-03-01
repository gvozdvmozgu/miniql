use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::hash::{Hash, Hasher};

use rustc_hash::FxHasher;
use sqlparser::ast::{
    BinaryOperator as SqlBinaryOperator, Distinct, Expr as SqlExpr, Function, FunctionArg,
    FunctionArgExpr, FunctionArguments, GroupByExpr, Ident, JoinConstraint, JoinOperator,
    LimitClause, ObjectName, ObjectNamePart, OrderBy as SqlOrderBy, OrderByExpr as SqlOrderByExpr,
    OrderByKind, Query as SqlQuery, Select, SelectFlavor, SelectItem,
    SelectItemQualifiedWildcardKind, SetExpr, Statement, TableFactor, UnaryOperator,
    Value as SqlValue, WildcardAdditionalOptions,
};
use sqlparser::dialect::{GenericDialect, SQLiteDialect};
use sqlparser::parser::Parser;

use crate::compare::compare_value_refs;
use crate::db::Db;
use crate::join::{
    Join as ExecJoin, JoinKey, JoinScratch, JoinSide, JoinStrategy, JoinType, JoinedRow,
    PreparedJoin,
};
use crate::pager::PageId;
use crate::query::{
    AggExpr, Expr, OrderBy, OrderDir, PreparedAggregate, PreparedScan, Row, ScanScratch, ValueLit,
};
use crate::schema::{parse_index_columns, parse_table_schema};
use crate::table::{self, BytesSpan, Corruption, QueryError, RawBytes, ValueRef, ValueSlot};

const DEFAULT_SQL_DISTINCT_HASH_MEM_CAP_BYTES: usize = 64 * 1024 * 1024;
// Optional override for DISTINCT hash dedup memory cap (bytes).
const SQL_DISTINCT_HASH_MEM_CAP_ENV: &str = "SQL_DISTINCT_HASH_MEM_CAP_BYTES";

pub(crate) struct PreparedSqlQuery<'db> {
    exec: SqlExec<'db>,
}

impl<'db> PreparedSqlQuery<'db> {
    pub(crate) fn for_each<F>(&mut self, scratch: &mut ScanScratch, cb: &mut F) -> table::Result<()>
    where
        F: for<'row> FnMut(Row<'row>) -> table::Result<()>,
    {
        self.exec.run(scratch, cb)
    }
}

pub(crate) fn prepare_sql_query<'db>(
    db: &'db Db,
    sql: &str,
) -> table::Result<PreparedSqlQuery<'db>> {
    let statement = parse_statement(sql)?;
    let Statement::Query(query) = statement else {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    };
    let exec = build_exec(db, *query)?;
    Ok(PreparedSqlQuery { exec })
}

pub(crate) fn execute_query_sql<F>(
    db: &Db,
    sql: &str,
    scratch: &mut ScanScratch,
    cb: &mut F,
) -> table::Result<()>
where
    F: for<'row> FnMut(Row<'row>) -> table::Result<()>,
{
    let mut prepared = prepare_sql_query(db, sql)?;
    prepared.for_each(scratch, cb)
}

enum SqlExec<'db> {
    Scan {
        prepared: PreparedScan<'db>,
        offset: usize,
        limit: Option<usize>,
        projection: Vec<u16>,
        ordered: bool,
        distinct: bool,
    },
    Aggregate {
        prepared: Box<PreparedAggregate<'db>>,
        offset: usize,
        limit: Option<usize>,
        visible_cols: usize,
        distinct: bool,
        order_by: Option<Vec<OrderBy>>,
    },
    Join {
        prepared: Box<PreparedJoin<'db>>,
        projection: Vec<u16>,
        where_expr: Option<Expr>,
        order_by: Option<Vec<OrderBy>>,
        offset: usize,
        limit: Option<usize>,
        distinct: bool,
    },
}

impl<'db> SqlExec<'db> {
    fn run<F>(&mut self, scratch: &mut ScanScratch, cb: &mut F) -> table::Result<()>
    where
        F: for<'row> FnMut(Row<'row>) -> table::Result<()>,
    {
        match self {
            Self::Scan { prepared, offset, limit, projection, ordered, distinct } => {
                if !*distinct {
                    let mut skipped = 0usize;
                    let mut emitted = 0usize;
                    if !*ordered {
                        return prepared
                            .for_each_eager(scratch, |_, row| {
                                if skipped < *offset {
                                    skipped += 1;
                                    return Ok(());
                                }
                                if limit_reached(*limit, emitted) {
                                    return Err(table::Error::Corrupted(Corruption::LimitReached));
                                }
                                emitted += 1;
                                cb(row)
                            })
                            .or_else(ignore_limit_reached);
                    }

                    let mut projected = Vec::with_capacity(projection.len());
                    let mut owned_bytes = Vec::<Vec<u8>>::new();
                    return prepared
                        .for_each(scratch, |_, row| {
                            projected.clear();
                            owned_bytes.clear();
                            for &col in projection.iter() {
                                project_value(
                                    row.get(col as usize)?,
                                    &mut projected,
                                    &mut owned_bytes,
                                );
                            }

                            let row = Row::from_raw(projected.as_slice(), None);
                            if skipped < *offset {
                                skipped += 1;
                                return Ok(());
                            }
                            if limit_reached(*limit, emitted) {
                                return Err(table::Error::Corrupted(Corruption::LimitReached));
                            }
                            emitted += 1;
                            cb(row)
                        })
                        .or_else(ignore_limit_reached);
                }

                let mut rows = Vec::<Vec<OwnedSqlValue>>::new();
                if !*ordered {
                    prepared.for_each_eager(scratch, |_, row| {
                        rows.push(owned_values_from_row(row, row.len()));
                        Ok(())
                    })?;
                } else {
                    let mut projected = Vec::with_capacity(projection.len());
                    let mut owned_bytes = Vec::<Vec<u8>>::new();
                    prepared.for_each(scratch, |_, row| {
                        projected.clear();
                        owned_bytes.clear();
                        for &col in projection.iter() {
                            project_value(row.get(col as usize)?, &mut projected, &mut owned_bytes);
                        }
                        let row = Row::from_raw(projected.as_slice(), None);
                        rows.push(owned_values_from_row(row, row.len()));
                        Ok(())
                    })?;
                }

                let keep = distinct_indices_for_rows(rows.as_slice())?;
                emit_owned_rows(select_rows_by_indices(rows, keep), *offset, *limit, cb)
            }
            Self::Aggregate { prepared, offset, limit, visible_cols, distinct, order_by } => {
                let mut rows: Vec<Vec<OwnedSqlValue>> = Vec::new();

                prepared.for_each(scratch, |row| {
                    let full_row = owned_values_from_row(row, row.len());
                    rows.push(full_row);
                    Ok(())
                })?;

                if *distinct {
                    let keep = distinct_indices_for_prefix(rows.as_slice(), *visible_cols)?;
                    rows = select_rows_by_indices(rows, keep);
                }

                if let Some(order_by) = order_by.as_ref() {
                    rows.sort_by(|left, right| compare_owned_row_values(left, right, order_by));
                }

                emit_owned_rows_visible(rows, *visible_cols, *offset, *limit, cb)
            }
            Self::Join { prepared, projection, where_expr, order_by, offset, limit, distinct } => {
                run_join_query(
                    prepared,
                    JoinRunOptions {
                        projection,
                        where_expr: where_expr.as_ref(),
                        order_by: order_by.as_deref(),
                        offset: *offset,
                        limit: *limit,
                        distinct: *distinct,
                    },
                    cb,
                )
            }
        }
    }
}

fn limit_reached(limit: Option<usize>, emitted: usize) -> bool {
    limit.is_some_and(|limit| emitted >= limit)
}

fn ignore_limit_reached(err: table::Error) -> table::Result<()> {
    if matches!(&err, table::Error::Corrupted(Corruption::LimitReached)) {
        Ok(())
    } else {
        Err(err)
    }
}

fn sql_distinct_hash_mem_cap_bytes() -> usize {
    std::env::var(SQL_DISTINCT_HASH_MEM_CAP_ENV)
        .ok()
        .and_then(|value| parse_sql_distinct_hash_mem_cap_override(value.as_str()))
        .unwrap_or(DEFAULT_SQL_DISTINCT_HASH_MEM_CAP_BYTES)
}

fn parse_sql_distinct_hash_mem_cap_override(value: &str) -> Option<usize> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    value.parse::<usize>().ok().filter(|value| *value > 0)
}

fn project_value(
    value: Option<ValueRef<'_>>,
    projected: &mut Vec<ValueSlot>,
    owned_bytes: &mut Vec<Vec<u8>>,
) {
    let slot = match value {
        Some(ValueRef::Null) | None => ValueSlot::Null,
        Some(ValueRef::Integer(value)) => ValueSlot::Integer(value),
        Some(ValueRef::Real(value)) => ValueSlot::Real(value),
        Some(ValueRef::Text(bytes)) => {
            owned_bytes.push(bytes.to_vec());
            let stored =
                owned_bytes.last().expect("owned bytes contains the row value we just pushed");
            ValueSlot::Text(BytesSpan::Mmap(RawBytes::from_slice(stored.as_slice())))
        }
        Some(ValueRef::Blob(bytes)) => {
            owned_bytes.push(bytes.to_vec());
            let stored =
                owned_bytes.last().expect("owned bytes contains the row value we just pushed");
            ValueSlot::Blob(BytesSpan::Mmap(RawBytes::from_slice(stored.as_slice())))
        }
    };
    projected.push(slot);
}

#[derive(Clone)]
enum OwnedSqlValue {
    Null,
    Integer(i64),
    Real(f64),
    Text(Vec<u8>),
    Blob(Vec<u8>),
}

impl OwnedSqlValue {
    fn from_row_value(value: Option<ValueRef<'_>>) -> Self {
        match value {
            Some(ValueRef::Null) | None => Self::Null,
            Some(ValueRef::Integer(value)) => Self::Integer(value),
            Some(ValueRef::Real(value)) => Self::Real(value),
            Some(ValueRef::Text(bytes)) => Self::Text(bytes.to_vec()),
            Some(ValueRef::Blob(bytes)) => Self::Blob(bytes.to_vec()),
        }
    }

    fn as_value_ref(&self) -> ValueRef<'_> {
        match self {
            Self::Null => ValueRef::Null,
            Self::Integer(value) => ValueRef::Integer(*value),
            Self::Real(value) => ValueRef::Real(*value),
            Self::Text(bytes) => ValueRef::Text(bytes.as_slice()),
            Self::Blob(bytes) => ValueRef::Blob(bytes.as_slice()),
        }
    }

    fn to_slot(&self) -> ValueSlot {
        match self {
            Self::Null => ValueSlot::Null,
            Self::Integer(value) => ValueSlot::Integer(*value),
            Self::Real(value) => ValueSlot::Real(*value),
            Self::Text(bytes) => {
                ValueSlot::Text(BytesSpan::Mmap(RawBytes::from_slice(bytes.as_slice())))
            }
            Self::Blob(bytes) => {
                ValueSlot::Blob(BytesSpan::Mmap(RawBytes::from_slice(bytes.as_slice())))
            }
        }
    }
}

struct DistinctRows {
    enabled: bool,
    mem_cap_bytes: usize,
    mem_used_bytes: usize,
    seen: HashMap<u64, Vec<Vec<OwnedSqlValue>>>,
}

impl DistinctRows {
    fn with_cap(enabled: bool, mem_cap_bytes: usize) -> Self {
        Self { enabled, mem_cap_bytes, mem_used_bytes: 0, seen: HashMap::new() }
    }

    fn accept_values<I>(&mut self, values: I) -> table::Result<bool>
    where
        I: IntoIterator<Item = OwnedSqlValue>,
    {
        if !self.enabled {
            return Ok(true);
        }

        let owned = values.into_iter().collect::<Vec<_>>();
        let hash = hash_sql_row(owned.as_slice());

        let mut has_bucket = false;
        if let Some(bucket) = self.seen.get(&hash) {
            has_bucket = true;
            if bucket.iter().any(|existing| sql_row_equal(existing.as_slice(), owned.as_slice())) {
                return Ok(false);
            }
        }

        let row_bytes = estimate_owned_row_bytes(owned.as_slice());
        let entry_overhead = if has_bucket {
            0
        } else {
            std::mem::size_of::<u64>() + std::mem::size_of::<Vec<Vec<OwnedSqlValue>>>()
        };
        self.charge_distinct_bytes(entry_overhead.saturating_add(row_bytes))?;
        match self.seen.entry(hash) {
            Entry::Occupied(mut entry) => entry.get_mut().push(owned),
            Entry::Vacant(entry) => {
                entry.insert(vec![owned]);
            }
        }
        Ok(true)
    }

    fn charge_distinct_bytes(&mut self, bytes: usize) -> table::Result<()> {
        let next = self.mem_used_bytes.saturating_add(bytes);
        if next > self.mem_cap_bytes {
            return Err(table::Error::Query(QueryError::SqlDistinctMemoryLimitExceeded));
        }
        self.mem_used_bytes = next;
        Ok(())
    }
}

fn sql_row_equal(left: &[OwnedSqlValue], right: &[OwnedSqlValue]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .all(|(l, r)| compare_value_refs(l.as_value_ref(), r.as_value_ref()) == Ordering::Equal)
}

fn estimate_owned_row_bytes(row: &[OwnedSqlValue]) -> usize {
    let base = std::mem::size_of::<Vec<OwnedSqlValue>>()
        .saturating_add(row.len().saturating_mul(std::mem::size_of::<OwnedSqlValue>()));
    base.saturating_add(
        row.iter()
            .map(|value| match value {
                OwnedSqlValue::Text(bytes) | OwnedSqlValue::Blob(bytes) => {
                    std::mem::size_of::<Vec<u8>>().saturating_add(bytes.len())
                }
                _ => 0,
            })
            .sum::<usize>(),
    )
}

fn hash_sql_row(row: &[OwnedSqlValue]) -> u64 {
    let mut hasher = FxHasher::default();
    row.len().hash(&mut hasher);
    for value in row {
        hash_sql_value(value, &mut hasher);
    }
    hasher.finish()
}

fn hash_sql_value(value: &OwnedSqlValue, hasher: &mut FxHasher) {
    match value {
        OwnedSqlValue::Null => {
            0u8.hash(hasher);
        }
        OwnedSqlValue::Integer(value) => {
            1u8.hash(hasher);
            normalize_numeric_hash(*value as f64).hash(hasher);
        }
        OwnedSqlValue::Real(value) => {
            1u8.hash(hasher);
            normalize_numeric_hash(*value).hash(hasher);
        }
        OwnedSqlValue::Text(bytes) => {
            2u8.hash(hasher);
            bytes.hash(hasher);
        }
        OwnedSqlValue::Blob(bytes) => {
            3u8.hash(hasher);
            bytes.hash(hasher);
        }
    }
}

fn normalize_numeric_hash(value: f64) -> u64 {
    if value == 0.0 {
        0.0f64.to_bits()
    } else if value.is_nan() {
        0x7ff8_0000_0000_0000
    } else {
        value.to_bits()
    }
}

fn owned_values_from_row(row: Row<'_>, len: usize) -> Vec<OwnedSqlValue> {
    let mut values = Vec::with_capacity(len);
    for idx in 0..len {
        values.push(OwnedSqlValue::from_row_value(row.get(idx)));
    }
    values
}

fn compare_owned_row_values(
    left: &[OwnedSqlValue],
    right: &[OwnedSqlValue],
    order_by: &[OrderBy],
) -> Ordering {
    for item in order_by {
        let idx = item.col as usize;
        let cmp = match (left.get(idx), right.get(idx)) {
            (Some(l), Some(r)) => compare_value_refs(l.as_value_ref(), r.as_value_ref()),
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Less,
            (Some(_), None) => Ordering::Greater,
        };
        if cmp != Ordering::Equal {
            return if matches!(item.dir, OrderDir::Desc) { cmp.reverse() } else { cmp };
        }
    }
    Ordering::Equal
}

fn compare_owned_row_lex(left: &[OwnedSqlValue], right: &[OwnedSqlValue]) -> Ordering {
    for (l, r) in left.iter().zip(right.iter()) {
        let cmp = compare_value_refs(l.as_value_ref(), r.as_value_ref());
        if cmp != Ordering::Equal {
            return cmp;
        }
    }
    left.len().cmp(&right.len())
}

fn is_distinct_hash_cap_error(err: &table::Error) -> bool {
    matches!(err, table::Error::Query(QueryError::SqlDistinctMemoryLimitExceeded))
}

fn distinct_indices_for_rows(rows: &[Vec<OwnedSqlValue>]) -> table::Result<Vec<usize>> {
    distinct_indices_by_key(rows, |row| Cow::Borrowed(row))
}

fn distinct_indices_for_prefix(
    rows: &[Vec<OwnedSqlValue>],
    key_len: usize,
) -> table::Result<Vec<usize>> {
    distinct_indices_by_key(rows, |row| {
        Cow::Owned(row.iter().take(key_len).cloned().collect::<Vec<_>>())
    })
}

fn distinct_indices_for_projection(
    rows: &[Vec<OwnedSqlValue>],
    projection: &[u16],
) -> table::Result<Vec<usize>> {
    distinct_indices_by_key(rows, |row| {
        Cow::Owned(
            projection
                .iter()
                .map(|idx| row.get(*idx as usize).cloned().unwrap_or(OwnedSqlValue::Null))
                .collect::<Vec<_>>(),
        )
    })
}

fn distinct_indices_by_key<F>(
    rows: &[Vec<OwnedSqlValue>],
    key_for_row: F,
) -> table::Result<Vec<usize>>
where
    F: for<'row> FnMut(&'row [OwnedSqlValue]) -> Cow<'row, [OwnedSqlValue]>,
{
    distinct_indices_by_key_cap(rows, sql_distinct_hash_mem_cap_bytes(), key_for_row)
}

fn distinct_indices_by_key_cap<F>(
    rows: &[Vec<OwnedSqlValue>],
    mem_cap_bytes: usize,
    mut key_for_row: F,
) -> table::Result<Vec<usize>>
where
    F: for<'row> FnMut(&'row [OwnedSqlValue]) -> Cow<'row, [OwnedSqlValue]>,
{
    let mut distinct_rows = DistinctRows::with_cap(true, mem_cap_bytes);
    let mut keep = Vec::with_capacity(rows.len());
    for (idx, row) in rows.iter().enumerate() {
        let key = key_for_row(row.as_slice());
        match distinct_rows.accept_values(key.iter().cloned()) {
            Ok(true) => keep.push(idx),
            Ok(false) => {}
            Err(err) if is_distinct_hash_cap_error(&err) => {
                return Ok(distinct_indices_via_sort_by_key(rows, &mut key_for_row));
            }
            Err(err) => return Err(err),
        }
    }
    Ok(keep)
}

fn distinct_indices_via_sort_by_key<F>(
    rows: &[Vec<OwnedSqlValue>],
    key_for_row: &mut F,
) -> Vec<usize>
where
    F: for<'row> FnMut(&'row [OwnedSqlValue]) -> Cow<'row, [OwnedSqlValue]>,
{
    let keys = rows.iter().map(|row| key_for_row(row.as_slice()).into_owned()).collect::<Vec<_>>();
    distinct_indices_via_sort(keys.as_slice())
}

#[cfg(test)]
fn distinct_indices_from_keys_with_hash_fallback_cap(
    keys: &[Vec<OwnedSqlValue>],
    mem_cap_bytes: usize,
) -> table::Result<Vec<usize>> {
    distinct_indices_by_key_cap(keys, mem_cap_bytes, |row| Cow::Borrowed(row))
}

fn distinct_indices_via_sort(keys: &[Vec<OwnedSqlValue>]) -> Vec<usize> {
    let mut keyed =
        keys.iter().cloned().enumerate().map(|(idx, key)| (key, idx)).collect::<Vec<_>>();
    keyed.sort_by(|(left_key, left_idx), (right_key, right_idx)| {
        compare_owned_row_lex(left_key, right_key).then(left_idx.cmp(right_idx))
    });

    let mut keep = Vec::new();
    let mut iter = keyed.into_iter();
    let Some((mut current_key, mut min_idx)) = iter.next() else {
        return keep;
    };
    for (key, idx) in iter {
        if compare_owned_row_lex(current_key.as_slice(), key.as_slice()) == Ordering::Equal {
            if idx < min_idx {
                min_idx = idx;
            }
        } else {
            keep.push(min_idx);
            current_key = key;
            min_idx = idx;
        }
    }
    keep.push(min_idx);
    keep.sort_unstable();
    keep
}

fn select_rows_by_indices(
    mut rows: Vec<Vec<OwnedSqlValue>>,
    indices: Vec<usize>,
) -> Vec<Vec<OwnedSqlValue>> {
    let mut out = Vec::with_capacity(indices.len());
    for idx in indices {
        out.push(std::mem::take(&mut rows[idx]));
    }
    out
}

fn emit_owned_rows<F>(
    rows: Vec<Vec<OwnedSqlValue>>,
    offset: usize,
    limit: Option<usize>,
    cb: &mut F,
) -> table::Result<()>
where
    F: for<'row> FnMut(Row<'row>) -> table::Result<()>,
{
    emit_owned_rows_visible(rows, usize::MAX, offset, limit, cb)
}

fn emit_owned_rows_visible<F>(
    rows: Vec<Vec<OwnedSqlValue>>,
    visible_cols: usize,
    offset: usize,
    limit: Option<usize>,
    cb: &mut F,
) -> table::Result<()>
where
    F: for<'row> FnMut(Row<'row>) -> table::Result<()>,
{
    let mut output_slots = Vec::new();
    let mut skipped = 0usize;
    let mut emitted = 0usize;
    for values in rows {
        if skipped < offset {
            skipped += 1;
            continue;
        }
        if limit_reached(limit, emitted) {
            break;
        }
        emitted += 1;

        output_slots.clear();
        let take = visible_cols.min(values.len());
        for value in values.iter().take(take) {
            output_slots.push(value.to_slot());
        }
        cb(Row::from_raw(output_slots.as_slice(), None))?;
    }
    Ok(())
}

struct JoinRunOptions<'a> {
    projection: &'a [u16],
    where_expr: Option<&'a Expr>,
    order_by: Option<&'a [OrderBy]>,
    offset: usize,
    limit: Option<usize>,
    distinct: bool,
}

fn run_join_query<F>(
    prepared: &mut PreparedJoin<'_>,
    opts: JoinRunOptions<'_>,
    cb: &mut F,
) -> table::Result<()>
where
    F: for<'row> FnMut(Row<'row>) -> table::Result<()>,
{
    let JoinRunOptions { projection, where_expr, order_by, offset, limit, distinct } = opts;
    let mut scratch = JoinScratch::with_capacity(8, 8, 0);

    if distinct {
        let mut rows = Vec::<Vec<OwnedSqlValue>>::new();
        prepared.for_each(&mut scratch, |joined| {
            let full_row = owned_values_from_joined_row(joined);
            if let Some(where_expr) = where_expr
                && eval_expr_on_owned(where_expr, full_row.as_slice())? != SqlTruth::True
            {
                return Ok(());
            }
            rows.push(full_row);
            Ok(())
        })?;

        let keep = distinct_indices_for_projection(rows.as_slice(), projection)?;
        rows = select_rows_by_indices(rows, keep);

        if let Some(order_by) = order_by {
            rows.sort_by(|left, right| compare_owned_row_values(left, right, order_by));
        }

        let mut output_slots = Vec::with_capacity(projection.len());
        let mut skipped = 0usize;
        let mut emitted = 0usize;
        for row in rows {
            if skipped < offset {
                skipped += 1;
                continue;
            }
            if limit_reached(limit, emitted) {
                break;
            }
            emitted += 1;

            output_slots.clear();
            for idx in projection {
                push_projected_slot_from_owned_row(row.as_slice(), *idx, &mut output_slots)?;
            }
            cb(Row::from_raw(output_slots.as_slice(), None))?;
        }

        return Ok(());
    }

    if let Some(order_by) = order_by {
        let mut rows = Vec::<Vec<OwnedSqlValue>>::new();
        prepared.for_each(&mut scratch, |joined| {
            let full_row = owned_values_from_joined_row(joined);
            if let Some(where_expr) = where_expr
                && eval_expr_on_owned(where_expr, full_row.as_slice())? != SqlTruth::True
            {
                return Ok(());
            }
            rows.push(full_row);
            Ok(())
        })?;

        rows.sort_by(|left, right| compare_owned_row_values(left, right, order_by));

        let mut output_slots = Vec::with_capacity(projection.len());
        let mut skipped = 0usize;
        let mut emitted = 0usize;
        for row in rows {
            if skipped < offset {
                skipped += 1;
                continue;
            }
            if limit_reached(limit, emitted) {
                break;
            }
            emitted += 1;
            output_slots.clear();
            for idx in projection {
                push_projected_slot_from_owned_row(row.as_slice(), *idx, &mut output_slots)?;
            }
            cb(Row::from_raw(output_slots.as_slice(), None))?;
        }

        return Ok(());
    }

    let mut output_slots = Vec::with_capacity(projection.len());
    let mut skipped = 0usize;
    let mut emitted = 0usize;
    prepared
        .for_each(&mut scratch, |joined| {
            let full_row = owned_values_from_joined_row(joined);
            if let Some(where_expr) = where_expr
                && eval_expr_on_owned(where_expr, full_row.as_slice())? != SqlTruth::True
            {
                return Ok(());
            }
            if skipped < offset {
                skipped += 1;
                return Ok(());
            }
            if limit_reached(limit, emitted) {
                return Err(table::Error::Corrupted(Corruption::LimitReached));
            }
            emitted += 1;

            output_slots.clear();
            for idx in projection {
                push_projected_slot_from_owned_row(full_row.as_slice(), *idx, &mut output_slots)?;
            }
            cb(Row::from_raw(output_slots.as_slice(), None))
        })
        .or_else(ignore_limit_reached)
}

fn owned_values_from_joined_row(joined: JoinedRow<'_>) -> Vec<OwnedSqlValue> {
    let mut values = Vec::with_capacity(joined.left.len() + joined.right.len());
    for idx in 0..joined.left.len() {
        values.push(OwnedSqlValue::from_row_value(joined.left.get(idx)));
    }
    for idx in 0..joined.right.len() {
        values.push(OwnedSqlValue::from_row_value(joined.right.get(idx)));
    }
    values
}

fn push_projected_slot_from_owned_row(
    values: &[OwnedSqlValue],
    idx: u16,
    out: &mut Vec<ValueSlot>,
) -> table::Result<()> {
    let value = values
        .get(idx as usize)
        .ok_or(table::Error::InvalidColumnIndex { col: idx, column_count: values.len() })?;
    let slot = match value {
        OwnedSqlValue::Null => ValueSlot::Null,
        OwnedSqlValue::Integer(v) => ValueSlot::Integer(*v),
        OwnedSqlValue::Real(v) => ValueSlot::Real(*v),
        OwnedSqlValue::Text(bytes) => {
            ValueSlot::Text(BytesSpan::Mmap(RawBytes::from_slice(bytes.as_slice())))
        }
        OwnedSqlValue::Blob(bytes) => {
            ValueSlot::Blob(BytesSpan::Mmap(RawBytes::from_slice(bytes.as_slice())))
        }
    };
    out.push(slot);
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SqlTruth {
    True,
    False,
    Null,
}

fn eval_expr_on_owned(expr: &Expr, values: &[OwnedSqlValue]) -> table::Result<SqlTruth> {
    match expr {
        Expr::Col(_) | Expr::Lit(_) => Ok(SqlTruth::Null),
        Expr::Eq(left, right) => eval_cmp(CmpOp::Eq, left, right, values),
        Expr::Ne(left, right) => eval_cmp(CmpOp::Ne, left, right, values),
        Expr::Lt(left, right) => eval_cmp(CmpOp::Lt, left, right, values),
        Expr::Le(left, right) => eval_cmp(CmpOp::Le, left, right, values),
        Expr::Gt(left, right) => eval_cmp(CmpOp::Gt, left, right, values),
        Expr::Ge(left, right) => eval_cmp(CmpOp::Ge, left, right, values),
        Expr::Like(left, right) => eval_like(left, right, values),
        Expr::And(left, right) => {
            let left = eval_expr_on_owned(left, values)?;
            if left == SqlTruth::False {
                return Ok(SqlTruth::False);
            }
            let right = eval_expr_on_owned(right, values)?;
            Ok(truth_and(left, right))
        }
        Expr::Or(left, right) => {
            let left = eval_expr_on_owned(left, values)?;
            if left == SqlTruth::True {
                return Ok(SqlTruth::True);
            }
            let right = eval_expr_on_owned(right, values)?;
            Ok(truth_or(left, right))
        }
        Expr::Not(inner) => Ok(truth_not(eval_expr_on_owned(inner, values)?)),
        Expr::IsNull(inner) => eval_is_null(inner, values),
        Expr::IsNotNull(inner) => eval_is_not_null(inner, values),
    }
}

fn eval_cmp(
    op: CmpOp,
    left: &Expr,
    right: &Expr,
    values: &[OwnedSqlValue],
) -> table::Result<SqlTruth> {
    let left = eval_operand(left, values)?;
    let right = eval_operand(right, values)?;
    Ok(match (left, right) {
        (None, _) | (_, None) => SqlTruth::Null,
        (Some(left), Some(right)) => compare_sql_values(op, left, right),
    })
}

fn eval_like(left: &Expr, right: &Expr, values: &[OwnedSqlValue]) -> table::Result<SqlTruth> {
    let left = eval_operand(left, values)?;
    let right = eval_operand(right, values)?;
    Ok(match (left, right) {
        (None, _) | (_, None) => SqlTruth::Null,
        (Some(left), Some(right)) => compare_sql_like(left, right),
    })
}

fn eval_operand<'a>(
    expr: &'a Expr,
    values: &'a [OwnedSqlValue],
) -> table::Result<Option<ValueRef<'a>>> {
    match expr {
        Expr::Col(idx) => values
            .get(*idx as usize)
            .map(OwnedSqlValue::as_value_ref)
            .ok_or(table::Error::InvalidColumnIndex { col: *idx, column_count: values.len() })
            .map(Some),
        Expr::Lit(lit) => Ok(Some(value_lit_to_ref(lit))),
        _ => Ok(None),
    }
}

fn eval_is_null(expr: &Expr, values: &[OwnedSqlValue]) -> table::Result<SqlTruth> {
    match expr {
        Expr::Col(idx) => {
            let value = values.get(*idx as usize).ok_or(table::Error::InvalidColumnIndex {
                col: *idx,
                column_count: values.len(),
            })?;
            Ok(if matches!(value, OwnedSqlValue::Null) { SqlTruth::True } else { SqlTruth::False })
        }
        Expr::Lit(ValueLit::Null) => Ok(SqlTruth::True),
        Expr::Lit(_) => Ok(SqlTruth::False),
        _ => Ok(if eval_expr_on_owned(expr, values)? == SqlTruth::Null {
            SqlTruth::True
        } else {
            SqlTruth::False
        }),
    }
}

fn eval_is_not_null(expr: &Expr, values: &[OwnedSqlValue]) -> table::Result<SqlTruth> {
    Ok(match eval_is_null(expr, values)? {
        SqlTruth::True => SqlTruth::False,
        SqlTruth::False => SqlTruth::True,
        SqlTruth::Null => SqlTruth::Null,
    })
}

fn truth_and(left: SqlTruth, right: SqlTruth) -> SqlTruth {
    match (left, right) {
        (SqlTruth::False, _) | (_, SqlTruth::False) => SqlTruth::False,
        (SqlTruth::True, SqlTruth::True) => SqlTruth::True,
        _ => SqlTruth::Null,
    }
}

fn truth_or(left: SqlTruth, right: SqlTruth) -> SqlTruth {
    match (left, right) {
        (SqlTruth::True, _) | (_, SqlTruth::True) => SqlTruth::True,
        (SqlTruth::False, SqlTruth::False) => SqlTruth::False,
        _ => SqlTruth::Null,
    }
}

fn truth_not(value: SqlTruth) -> SqlTruth {
    match value {
        SqlTruth::True => SqlTruth::False,
        SqlTruth::False => SqlTruth::True,
        SqlTruth::Null => SqlTruth::Null,
    }
}

#[derive(Clone, Copy)]
enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

fn compare_sql_values(op: CmpOp, left: ValueRef<'_>, right: ValueRef<'_>) -> SqlTruth {
    match (left, right) {
        (ValueRef::Null, _) | (_, ValueRef::Null) => SqlTruth::Null,
        (ValueRef::Integer(l), ValueRef::Integer(r)) => cmp_order(op, l.cmp(&r)),
        (ValueRef::Integer(l), ValueRef::Real(r)) => cmp_f64(op, l as f64, r),
        (ValueRef::Real(l), ValueRef::Integer(r)) => cmp_f64(op, l, r as f64),
        (ValueRef::Real(l), ValueRef::Real(r)) => cmp_f64(op, l, r),
        (ValueRef::Text(l), ValueRef::Text(r)) => cmp_order(op, l.cmp(r)),
        (ValueRef::Blob(l), ValueRef::Blob(r)) => cmp_order(op, l.cmp(r)),
        _ => SqlTruth::False,
    }
}

fn compare_sql_like(value: ValueRef<'_>, pattern: ValueRef<'_>) -> SqlTruth {
    match (value, pattern) {
        (ValueRef::Null, _) | (_, ValueRef::Null) => SqlTruth::Null,
        (ValueRef::Text(value), ValueRef::Text(pattern)) => {
            if like_match(value, pattern) {
                SqlTruth::True
            } else {
                SqlTruth::False
            }
        }
        _ => SqlTruth::False,
    }
}

fn cmp_f64(op: CmpOp, left: f64, right: f64) -> SqlTruth {
    match left.partial_cmp(&right) {
        Some(order) => cmp_order(op, order),
        None => SqlTruth::False,
    }
}

fn cmp_order(op: CmpOp, order: Ordering) -> SqlTruth {
    let matches = match op {
        CmpOp::Eq => order == Ordering::Equal,
        CmpOp::Ne => order != Ordering::Equal,
        CmpOp::Lt => order == Ordering::Less,
        CmpOp::Le => order != Ordering::Greater,
        CmpOp::Gt => order == Ordering::Greater,
        CmpOp::Ge => order != Ordering::Less,
    };
    if matches { SqlTruth::True } else { SqlTruth::False }
}

fn like_match(value: &[u8], pattern: &[u8]) -> bool {
    let mut value_pos = 0usize;
    let mut pattern_pos = 0usize;
    let mut star_pattern_pos: Option<usize> = None;
    let mut star_value_pos = 0usize;

    while value_pos < value.len() {
        if pattern_pos < pattern.len() {
            let token = pattern[pattern_pos];
            if token == b'%' {
                star_pattern_pos = Some(pattern_pos);
                pattern_pos += 1;
                star_value_pos = value_pos;
                continue;
            }
            if token == b'_' || ascii_like_eq(token, value[value_pos]) {
                pattern_pos += 1;
                value_pos += 1;
                continue;
            }
        }

        let Some(star_pos) = star_pattern_pos else {
            return false;
        };
        pattern_pos = star_pos + 1;
        star_value_pos += 1;
        value_pos = star_value_pos;
    }

    while pattern_pos < pattern.len() && pattern[pattern_pos] == b'%' {
        pattern_pos += 1;
    }
    pattern_pos == pattern.len()
}

fn ascii_like_eq(left: u8, right: u8) -> bool {
    left.eq_ignore_ascii_case(&right)
}

fn value_lit_to_ref(value: &ValueLit) -> ValueRef<'_> {
    match value {
        ValueLit::Null => ValueRef::Null,
        ValueLit::Integer(value) => ValueRef::Integer(*value),
        ValueLit::Real(value) => ValueRef::Real(*value),
        ValueLit::Text(bytes) => ValueRef::Text(bytes.as_slice()),
    }
}

fn build_exec<'db>(db: &'db Db, query: SqlQuery) -> table::Result<SqlExec<'db>> {
    if query.with.is_some()
        || query.fetch.is_some()
        || !query.locks.is_empty()
        || query.for_clause.is_some()
        || query.settings.is_some()
        || query.format_clause.is_some()
        || !query.pipe_operators.is_empty()
    {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    }

    let SetExpr::Select(select) = *query.body else {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    };
    validate_select_shape(&select)?;

    let distinct = parse_distinct(select.distinct.as_ref())?;
    if is_join_select(&select) {
        return build_join_exec(
            db,
            &select,
            query.order_by.as_ref(),
            query.limit_clause.as_ref(),
            distinct,
        );
    }
    let (table, resolver) = resolve_table(db, &select)?;
    let projection = parse_projection(select.projection.as_slice(), &resolver)?;
    if projection.is_empty() {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    }
    let output_names =
        OutputNameResolver::from_select_items(select.projection.as_slice(), &resolver)?;

    let where_expr =
        select.selection.as_ref().map(|expr| parse_filter_expr(expr, &resolver)).transpose()?;
    let group_by = parse_group_by(&select.group_by, &resolver)?;
    let (limit, offset) = parse_limit_offset(query.limit_clause.as_ref())?;

    let has_agg = projection.iter().any(|item| matches!(item, ProjectionExpr::Aggregate(_)));
    let aggregate_mode = has_agg || !group_by.is_empty();

    if aggregate_mode {
        let mut select_items = projection
            .into_iter()
            .map(|item| match item {
                ProjectionExpr::Column(col) => AggExpr::value(Expr::Col(col)),
                ProjectionExpr::Aggregate(agg) => agg,
            })
            .collect::<Vec<_>>();
        let visible_cols = select_items.len();
        let having = select
            .having
            .as_ref()
            .map(|expr| parse_having_expr(expr, &resolver, &output_names, &mut select_items))
            .transpose()?;
        let order_by = parse_aggregate_order_by(
            query.order_by.as_ref(),
            &resolver,
            &output_names,
            &mut select_items,
            visible_cols,
        )?;

        let mut scan = table.scan();
        if let Some(expr) = where_expr {
            scan = scan.filter(expr);
        }
        let mut aggregate = scan.aggregate_dyn(select_items).with_group_by_override(group_by);
        if let Some(expr) = having {
            aggregate = aggregate.having(expr);
        }
        let prepared = aggregate.compile()?;
        return Ok(SqlExec::Aggregate {
            prepared: Box::new(prepared),
            offset,
            limit,
            visible_cols,
            distinct,
            order_by,
        });
    } else if select.having.is_some() {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    }

    let projection_cols = projection
        .into_iter()
        .map(|item| match item {
            ProjectionExpr::Column(col) => Ok(col),
            ProjectionExpr::Aggregate(_) => Err(table::Error::Query(QueryError::SqlUnsupported)),
        })
        .collect::<table::Result<Vec<_>>>()?;
    let order_by =
        parse_order_by(query.order_by.as_ref(), &resolver, &projection_cols, Some(&output_names))?;
    let ordered = order_by.is_some();

    let mut scan = table.scan().with_projection_override(Some(projection_cols.clone()));
    if let Some(expr) = where_expr {
        scan = scan.filter(expr);
    }
    scan = scan.with_order_by_override(order_by);
    if !distinct && let Some(limit) = limit {
        let scan_limit = offset
            .checked_add(limit)
            .ok_or(table::Error::Query(QueryError::SqlInvalidLimitOffset))?;
        scan = scan.limit(scan_limit);
    }

    let prepared = scan.compile()?;
    Ok(SqlExec::Scan { prepared, offset, limit, projection: projection_cols, ordered, distinct })
}

fn is_join_select(select: &Select) -> bool {
    select.from.len() == 1 && !select.from[0].joins.is_empty()
}

fn group_by_is_empty(group_by: &GroupByExpr) -> bool {
    matches!(group_by, GroupByExpr::Expressions(exprs, modifiers) if exprs.is_empty() && modifiers.is_empty())
}

#[derive(Clone)]
struct JoinTableResolver {
    table_name: String,
    alias: Option<String>,
    columns: Vec<String>,
}

#[derive(Clone, Copy)]
struct ResolvedJoinCol {
    side: JoinSide,
    col: u16,
    global: u16,
}

struct JoinColumnResolver {
    left: JoinTableResolver,
    right: JoinTableResolver,
}

impl JoinColumnResolver {
    fn new(left: JoinTableResolver, right: JoinTableResolver) -> Self {
        Self { left, right }
    }

    fn left_len(&self) -> usize {
        self.left.columns.len()
    }

    fn resolve_column_expr(&self, expr: &SqlExpr) -> table::Result<u16> {
        let resolved = self.resolve_side_column_expr(expr)?;
        Ok(resolved.global)
    }

    fn resolve_side_column_expr(&self, expr: &SqlExpr) -> table::Result<ResolvedJoinCol> {
        match expr {
            SqlExpr::Identifier(ident) => self.resolve_unqualified(&ident.value),
            SqlExpr::CompoundIdentifier(idents) => self.resolve_compound(idents.as_slice()),
            _ => Err(table::Error::Query(QueryError::SqlUnsupported)),
        }
    }

    fn resolve_unqualified(&self, name: &str) -> table::Result<ResolvedJoinCol> {
        let left_match = self
            .left
            .columns
            .iter()
            .enumerate()
            .filter_map(|(idx, col)| col.eq_ignore_ascii_case(name).then_some(idx))
            .collect::<Vec<_>>();
        let right_match = self
            .right
            .columns
            .iter()
            .enumerate()
            .filter_map(|(idx, col)| col.eq_ignore_ascii_case(name).then_some(idx))
            .collect::<Vec<_>>();

        let left_one = left_match.first().copied();
        let right_one = right_match.first().copied();
        let left_ambiguous = left_match.len() > 1;
        let right_ambiguous = right_match.len() > 1;

        if left_ambiguous || right_ambiguous || (left_one.is_some() && right_one.is_some()) {
            return Err(table::Error::Query(QueryError::SqlAmbiguousColumn));
        }
        if let Some(idx) = left_one {
            return self.side_col_to_resolved(JoinSide::Left, idx);
        }
        if let Some(idx) = right_one {
            return self.side_col_to_resolved(JoinSide::Right, idx);
        }
        Err(table::Error::Query(QueryError::SqlUnknownColumn))
    }

    fn resolve_compound(&self, parts: &[Ident]) -> table::Result<ResolvedJoinCol> {
        if parts.len() < 2 {
            return Err(table::Error::Query(QueryError::SqlUnsupported));
        }
        let qualifier = &parts[parts.len() - 2].value;
        let column = &parts[parts.len() - 1].value;
        let Some(side) = self.qualifier_side(qualifier)? else {
            return Err(table::Error::Query(QueryError::SqlUnknownColumn));
        };
        self.resolve_on_side(side, column)
    }

    fn resolve_on_side(&self, side: JoinSide, name: &str) -> table::Result<ResolvedJoinCol> {
        let columns = match side {
            JoinSide::Left => &self.left.columns,
            JoinSide::Right => &self.right.columns,
        };
        let matches = columns
            .iter()
            .enumerate()
            .filter_map(|(idx, col)| col.eq_ignore_ascii_case(name).then_some(idx))
            .collect::<Vec<_>>();
        let Some(idx) = matches.first().copied() else {
            return Err(table::Error::Query(QueryError::SqlUnknownColumn));
        };
        if matches.len() > 1 {
            return Err(table::Error::Query(QueryError::SqlAmbiguousColumn));
        }
        self.side_col_to_resolved(side, idx)
    }

    fn qualifier_side(&self, qualifier: &str) -> table::Result<Option<JoinSide>> {
        let left_matches = self.qualifier_matches(&self.left, qualifier);
        let right_matches = self.qualifier_matches(&self.right, qualifier);
        match (left_matches, right_matches) {
            (false, false) => Ok(None),
            (true, false) => Ok(Some(JoinSide::Left)),
            (false, true) => Ok(Some(JoinSide::Right)),
            (true, true) => Err(table::Error::Query(QueryError::SqlAmbiguousColumn)),
        }
    }

    fn qualifier_matches(&self, table: &JoinTableResolver, qualifier: &str) -> bool {
        table.table_name.eq_ignore_ascii_case(qualifier)
            || table.alias.as_ref().is_some_and(|alias| alias.eq_ignore_ascii_case(qualifier))
    }

    fn side_col_to_resolved(&self, side: JoinSide, idx: usize) -> table::Result<ResolvedJoinCol> {
        let col =
            u16::try_from(idx).map_err(|_| table::Error::Query(QueryError::SqlUnsupported))?;
        let global = match side {
            JoinSide::Left => col,
            JoinSide::Right => {
                let left_len = self.left_len();
                let global = left_len
                    .checked_add(idx)
                    .ok_or(table::Error::Query(QueryError::SqlUnsupported))?;
                u16::try_from(global)
                    .map_err(|_| table::Error::Query(QueryError::SqlUnsupported))?
            }
        };
        Ok(ResolvedJoinCol { side, col, global })
    }

    fn expand_wildcard(&self, side: Option<JoinSide>) -> table::Result<Vec<u16>> {
        let mut cols = Vec::new();
        match side {
            None => {
                cols.extend((0..self.left.columns.len()).map(|idx| idx as u16));
                let left_len = self.left_len();
                cols.extend(
                    (0..self.right.columns.len())
                        .map(|idx| left_len + idx)
                        .map(u16::try_from)
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|_| table::Error::Query(QueryError::SqlUnsupported))?,
                );
            }
            Some(JoinSide::Left) => {
                cols.extend((0..self.left.columns.len()).map(|idx| idx as u16));
            }
            Some(JoinSide::Right) => {
                let left_len = self.left_len();
                cols.extend(
                    (0..self.right.columns.len())
                        .map(|idx| left_len + idx)
                        .map(u16::try_from)
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|_| table::Error::Query(QueryError::SqlUnsupported))?,
                );
            }
        }
        Ok(cols)
    }

    fn column_name_by_global(&self, global: u16) -> Option<&str> {
        let global = global as usize;
        if global < self.left.columns.len() {
            return self.left.columns.get(global).map(String::as_str);
        }
        self.right.columns.get(global - self.left.columns.len()).map(String::as_str)
    }
}

fn build_join_exec<'db>(
    db: &'db Db,
    select: &Select,
    order_by: Option<&SqlOrderBy>,
    limit_clause: Option<&LimitClause>,
    distinct: bool,
) -> table::Result<SqlExec<'db>> {
    if !group_by_is_empty(&select.group_by) || select.having.is_some() {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    }
    if select.from.len() != 1 {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    }
    let table_with_joins = &select.from[0];
    if table_with_joins.joins.len() != 1 {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    }

    let join = &table_with_joins.joins[0];
    if join.global {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    }

    let (left_table, left_resolver) = resolve_join_table_factor(db, &table_with_joins.relation)?;
    let (right_table, right_resolver) = resolve_join_table_factor(db, &join.relation)?;
    let resolver = JoinColumnResolver::new(left_resolver, right_resolver);

    let (join_type, on_expr) = parse_join_operator(&join.join_operator)?;
    let (left_key, right_key) = parse_join_on(on_expr, &resolver)?;

    let (projection, output_names) =
        parse_join_projection(select.projection.as_slice(), &resolver)?;
    if projection.is_empty() {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    }

    let where_expr = select
        .selection
        .as_ref()
        .map(|expr| parse_join_filter_expr(expr, &resolver))
        .transpose()?;
    let order_by = parse_join_order_by(order_by, &resolver, projection.as_slice(), &output_names)?;
    let (limit, offset) = parse_limit_offset(limit_clause)?;

    let mut join_builder = match join_type {
        JoinType::Inner => {
            ExecJoin::inner(left_table.scan(), right_table.scan()).on(left_key, right_key)
        }
        JoinType::Left => {
            ExecJoin::left(left_table.scan(), right_table.scan()).on(left_key, right_key)
        }
    };
    if let Some(strategy) = choose_join_strategy_from_right_index(db, &resolver.right, right_key)? {
        join_builder = join_builder.strategy(strategy);
    }
    let prepared = join_builder.compile()?;

    Ok(SqlExec::Join {
        prepared: Box::new(prepared),
        projection,
        where_expr,
        order_by,
        offset,
        limit,
        distinct,
    })
}

fn choose_join_strategy_from_right_index(
    db: &Db,
    right_resolver: &JoinTableResolver,
    right_key: JoinKey,
) -> table::Result<Option<JoinStrategy>> {
    let JoinKey::Col(right_col) = right_key else {
        return Ok(None);
    };
    let Some(right_col_name) = right_resolver.columns.get(right_col as usize) else {
        return Ok(None);
    };

    db.find_schema(|row| {
        if !row.kind.eq_ignore_ascii_case("index")
            || !row.tbl_name.eq_ignore_ascii_case(right_resolver.table_name.as_str())
        {
            return Ok(None);
        }
        let Some(sql) = row.sql.as_str() else {
            return Ok(None);
        };
        let Some(index_cols) = parse_index_columns(sql) else {
            return Ok(None);
        };
        if index_cols
            .first()
            .is_some_and(|index_col| index_col.eq_ignore_ascii_case(right_col_name.as_str()))
        {
            return Ok(Some(JoinStrategy::IndexNestedLoop {
                index_root: row.root,
                index_key_col: 0,
            }));
        }

        Ok(None)
    })
}

fn parse_join_operator(join: &JoinOperator) -> table::Result<(JoinType, &SqlExpr)> {
    let (join_type, constraint) = match join {
        JoinOperator::Join(constraint) | JoinOperator::Inner(constraint) => {
            (JoinType::Inner, constraint)
        }
        JoinOperator::Left(constraint) | JoinOperator::LeftOuter(constraint) => {
            (JoinType::Left, constraint)
        }
        _ => return Err(table::Error::Query(QueryError::SqlUnsupported)),
    };
    let JoinConstraint::On(expr) = constraint else {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    };
    Ok((join_type, expr))
}

fn parse_join_on(
    expr: &SqlExpr,
    resolver: &JoinColumnResolver,
) -> table::Result<(JoinKey, JoinKey)> {
    if let SqlExpr::Nested(inner) = expr {
        return parse_join_on(inner, resolver);
    }
    let SqlExpr::BinaryOp { left, op: SqlBinaryOperator::Eq, right } = expr else {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    };
    let left_col = resolver.resolve_side_column_expr(left)?;
    let right_col = resolver.resolve_side_column_expr(right)?;
    if left_col.side == right_col.side {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    }

    let (left_key, right_key) = if matches!(left_col.side, JoinSide::Left) {
        (JoinKey::Col(left_col.col), JoinKey::Col(right_col.col))
    } else {
        (JoinKey::Col(right_col.col), JoinKey::Col(left_col.col))
    };
    Ok((left_key, right_key))
}

fn parse_join_projection(
    items: &[SelectItem],
    resolver: &JoinColumnResolver,
) -> table::Result<(Vec<u16>, OutputNameResolver)> {
    let mut projection = Vec::new();
    let mut output_names = OutputNameResolver::new();
    for item in items {
        match item {
            SelectItem::UnnamedExpr(expr) => {
                let col = parse_join_projection_expr(expr, resolver)?;
                let idx = projection.len();
                projection.push(col);
                if let Some(name) = join_projection_expr_name(expr, resolver)? {
                    output_names.register(name, idx);
                }
            }
            SelectItem::ExprWithAlias { expr, alias } => {
                let col = parse_join_projection_expr(expr, resolver)?;
                let idx = projection.len();
                projection.push(col);
                output_names.register(alias.value.clone(), idx);
            }
            SelectItem::Wildcard(options) => {
                validate_wildcard_options(options)?;
                for col in resolver.expand_wildcard(None)? {
                    let idx = projection.len();
                    projection.push(col);
                    if let Some(name) = resolver.column_name_by_global(col) {
                        output_names.register(name.to_string(), idx);
                    }
                }
            }
            SelectItem::QualifiedWildcard(kind, options) => {
                validate_wildcard_options(options)?;
                match kind {
                    SelectItemQualifiedWildcardKind::ObjectName(name) => {
                        let qualifier = object_name_last_ident(name)?;
                        let side = resolver.qualifier_side(qualifier.as_str())?;
                        let Some(side) = side else {
                            return Err(table::Error::Query(QueryError::SqlUnknownColumn));
                        };
                        for col in resolver.expand_wildcard(Some(side))? {
                            let idx = projection.len();
                            projection.push(col);
                            if let Some(name) = resolver.column_name_by_global(col) {
                                output_names.register(name.to_string(), idx);
                            }
                        }
                    }
                    SelectItemQualifiedWildcardKind::Expr(_) => {
                        return Err(table::Error::Query(QueryError::SqlUnsupported));
                    }
                }
            }
        }
    }
    Ok((projection, output_names))
}

fn parse_join_projection_expr(expr: &SqlExpr, resolver: &JoinColumnResolver) -> table::Result<u16> {
    match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            resolver.resolve_column_expr(expr)
        }
        _ => Err(table::Error::Query(QueryError::SqlUnsupported)),
    }
}

fn join_projection_expr_name(
    expr: &SqlExpr,
    resolver: &JoinColumnResolver,
) -> table::Result<Option<String>> {
    match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            let col = resolver.resolve_column_expr(expr)?;
            Ok(resolver.column_name_by_global(col).map(ToOwned::to_owned))
        }
        _ => Ok(None),
    }
}

fn parse_join_order_by(
    order_by: Option<&SqlOrderBy>,
    resolver: &JoinColumnResolver,
    projection: &[u16],
    output_names: &OutputNameResolver,
) -> table::Result<Option<Vec<OrderBy>>> {
    let Some(order_by) = order_by else {
        return Ok(None);
    };
    if order_by.interpolate.is_some() {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    }
    let OrderByKind::Expressions(exprs) = &order_by.kind else {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    };
    if exprs.is_empty() {
        return Ok(None);
    }

    let mut out = Vec::with_capacity(exprs.len());
    for expr in exprs {
        if expr.with_fill.is_some() || expr.options.nulls_first.is_some() {
            return Err(table::Error::Query(QueryError::SqlUnsupported));
        }
        let col = if let Some(idx) = parse_ordinal(expr.expr.clone())? {
            let Some(global) = projection.get(idx) else {
                return Err(table::Error::Query(QueryError::SqlInvalidLimitOffset));
            };
            *global
        } else if let Some(idx) = output_names.resolve_expr(&expr.expr)? {
            let Some(global) = projection.get(idx) else {
                return Err(table::Error::Query(QueryError::SqlUnknownColumn));
            };
            *global
        } else {
            resolver.resolve_column_expr(&expr.expr)?
        };
        let dir = if expr.options.asc == Some(false) { OrderDir::Desc } else { OrderDir::Asc };
        out.push(OrderBy { col, dir });
    }

    Ok(Some(out))
}

fn parse_join_filter_expr(expr: &SqlExpr, resolver: &JoinColumnResolver) -> table::Result<Expr> {
    match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            Ok(Expr::Col(resolver.resolve_column_expr(expr)?))
        }
        SqlExpr::Value(value) => Ok(Expr::Lit(parse_value(&value.value)?)),
        SqlExpr::Nested(inner) => parse_join_filter_expr(inner, resolver),
        SqlExpr::UnaryOp { op, expr } => match op {
            UnaryOperator::Not | UnaryOperator::BangNot => {
                Ok(parse_join_filter_expr(expr, resolver)?.not())
            }
            UnaryOperator::Plus => parse_join_filter_expr(expr, resolver),
            UnaryOperator::Minus => {
                let value = parse_join_filter_expr(expr, resolver)?;
                match value {
                    Expr::Lit(ValueLit::Integer(v)) => v
                        .checked_neg()
                        .map(crate::query::lit_i64)
                        .ok_or(table::Error::Query(QueryError::SqlUnsupported)),
                    Expr::Lit(ValueLit::Real(v)) => Ok(crate::query::lit_f64(-v)),
                    _ => Err(table::Error::Query(QueryError::SqlUnsupported)),
                }
            }
            _ => Err(table::Error::Query(QueryError::SqlUnsupported)),
        },
        SqlExpr::BinaryOp { left, op, right } => {
            let left = parse_join_filter_expr(left, resolver)?;
            let right = parse_join_filter_expr(right, resolver)?;
            match op {
                SqlBinaryOperator::Eq => Ok(left.eq(right)),
                SqlBinaryOperator::NotEq => Ok(left.ne(right)),
                SqlBinaryOperator::Lt => Ok(left.lt(right)),
                SqlBinaryOperator::LtEq => Ok(left.le(right)),
                SqlBinaryOperator::Gt => Ok(left.gt(right)),
                SqlBinaryOperator::GtEq => Ok(left.ge(right)),
                SqlBinaryOperator::And => Ok(left.and(right)),
                SqlBinaryOperator::Or => Ok(left.or(right)),
                _ => Err(table::Error::Query(QueryError::SqlUnsupported)),
            }
        }
        SqlExpr::InList { expr, list, negated } => {
            let expr = parse_join_filter_expr(expr, resolver)?;
            let list = list
                .iter()
                .map(|value| parse_join_filter_expr(value, resolver))
                .collect::<table::Result<Vec<_>>>()?;
            Ok(fold_in_expr(expr, list, *negated))
        }
        SqlExpr::Between { expr, negated, low, high } => {
            let expr = parse_join_filter_expr(expr, resolver)?;
            let low = parse_join_filter_expr(low, resolver)?;
            let high = parse_join_filter_expr(high, resolver)?;
            Ok(fold_between_expr(expr, low, high, *negated))
        }
        SqlExpr::Like { negated, any, expr, pattern, escape_char } => {
            if *any || escape_char.is_some() {
                return Err(table::Error::Query(QueryError::SqlUnsupported));
            }
            let expr = parse_join_filter_expr(expr, resolver)?;
            let pattern = parse_join_filter_expr(pattern, resolver)?;
            Ok(fold_like_expr(expr, pattern, *negated))
        }
        SqlExpr::IsNull(inner) => Ok(parse_join_filter_expr(inner, resolver)?.is_null()),
        SqlExpr::IsNotNull(inner) => Ok(parse_join_filter_expr(inner, resolver)?.is_not_null()),
        _ => Err(table::Error::Query(QueryError::SqlUnsupported)),
    }
}

fn resolve_join_table_factor<'db>(
    db: &'db Db,
    relation: &TableFactor,
) -> table::Result<(crate::db::Table<'db>, JoinTableResolver)> {
    let TableFactor::Table {
        name,
        alias,
        args,
        with_hints,
        version,
        with_ordinality,
        partitions,
        json_path,
        sample,
        index_hints,
    } = relation
    else {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    };
    if args.is_some()
        || !with_hints.is_empty()
        || version.is_some()
        || *with_ordinality
        || !partitions.is_empty()
        || json_path.is_some()
        || sample.is_some()
        || !index_hints.is_empty()
    {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    }
    if alias.as_ref().is_some_and(|alias| !alias.columns.is_empty()) {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    }

    let table_name = object_name_last_ident(name)?;
    let table = if table_name.eq_ignore_ascii_case("sqlite_schema") {
        db.table_from_root(PageId::ROOT)
    } else {
        db.table(table_name.as_str())?
    };
    let columns = load_table_columns(db, table_name.as_str())?;
    let alias = alias.as_ref().map(|alias| alias.name.value.to_ascii_lowercase());
    Ok((table, JoinTableResolver { table_name, alias, columns }))
}

fn parse_statement(sql: &str) -> table::Result<Statement> {
    let sqlite = SQLiteDialect {};
    let statements = Parser::parse_sql(&sqlite, sql).or_else(|_| {
        let generic = GenericDialect {};
        Parser::parse_sql(&generic, sql)
    });
    let mut statements = statements.map_err(|_| table::Error::Query(QueryError::SqlParse))?;
    if statements.len() != 1 {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    }
    Ok(statements.remove(0))
}

fn validate_select_shape(select: &Select) -> table::Result<()> {
    let has_modifiers =
        select.select_modifiers.as_ref().is_some_and(sqlparser::ast::SelectModifiers::is_any_set);
    if select.optimizer_hint.is_some()
        || has_modifiers
        || select.top.is_some()
        || select.exclude.is_some()
        || select.into.is_some()
        || !select.lateral_views.is_empty()
        || select.prewhere.is_some()
        || !select.connect_by.is_empty()
        || !select.cluster_by.is_empty()
        || !select.distribute_by.is_empty()
        || !select.sort_by.is_empty()
        || !select.named_window.is_empty()
        || select.qualify.is_some()
        || select.value_table_mode.is_some()
        || !matches!(select.flavor, SelectFlavor::Standard)
    {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    }
    Ok(())
}

fn parse_distinct(distinct: Option<&Distinct>) -> table::Result<bool> {
    match distinct {
        None | Some(Distinct::All) => Ok(false),
        Some(Distinct::Distinct) => Ok(true),
        Some(Distinct::On(_)) => Err(table::Error::Query(QueryError::SqlUnsupported)),
    }
}

enum NameBinding {
    Unique(usize),
    Ambiguous,
}

struct OutputNameResolver {
    names: HashMap<String, NameBinding>,
}

impl OutputNameResolver {
    fn new() -> Self {
        Self { names: HashMap::new() }
    }

    fn from_select_items(items: &[SelectItem], resolver: &ColumnResolver) -> table::Result<Self> {
        let mut out = Self::new();
        let mut index = 0usize;
        for item in items {
            match item {
                SelectItem::UnnamedExpr(expr) => {
                    if let Some(name) = projection_expr_name(expr, resolver)? {
                        out.register(name, index);
                    }
                    index += 1;
                }
                SelectItem::ExprWithAlias { alias, .. } => {
                    out.register(alias.value.clone(), index);
                    index += 1;
                }
                SelectItem::Wildcard(options) => {
                    validate_wildcard_options(options)?;
                    for name in &resolver.columns {
                        out.register(name.clone(), index);
                        index += 1;
                    }
                }
                SelectItem::QualifiedWildcard(kind, options) => {
                    validate_wildcard_options(options)?;
                    match kind {
                        SelectItemQualifiedWildcardKind::ObjectName(name) => {
                            let qualifier = object_name_last_ident(name)?;
                            if !resolver.qualifier_matches(qualifier.as_str()) {
                                return Err(table::Error::Query(QueryError::SqlUnknownColumn));
                            }
                            for name in &resolver.columns {
                                out.register(name.clone(), index);
                                index += 1;
                            }
                        }
                        SelectItemQualifiedWildcardKind::Expr(_) => {
                            return Err(table::Error::Query(QueryError::SqlUnsupported));
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    fn register(&mut self, name: String, index: usize) {
        let key = name.to_ascii_lowercase();
        match self.names.entry(key) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(NameBinding::Unique(index));
            }
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                entry.insert(NameBinding::Ambiguous);
            }
        }
    }

    fn resolve_name(&self, name: &str) -> table::Result<Option<usize>> {
        match self.names.get(&name.to_ascii_lowercase()) {
            None => Ok(None),
            Some(NameBinding::Unique(index)) => Ok(Some(*index)),
            Some(NameBinding::Ambiguous) => {
                Err(table::Error::Query(QueryError::SqlAmbiguousColumn))
            }
        }
    }

    fn resolve_expr(&self, expr: &SqlExpr) -> table::Result<Option<usize>> {
        match expr {
            SqlExpr::Identifier(ident) => self.resolve_name(ident.value.as_str()),
            _ => Ok(None),
        }
    }
}

fn projection_expr_name(
    expr: &SqlExpr,
    resolver: &ColumnResolver,
) -> table::Result<Option<String>> {
    match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            let col = resolver.resolve_column_expr(expr)? as usize;
            Ok(resolver.columns.get(col).map(|name| name.to_ascii_lowercase()))
        }
        _ => Ok(None),
    }
}

fn parse_having_expr(
    expr: &SqlExpr,
    resolver: &ColumnResolver,
    output_names: &OutputNameResolver,
    select_items: &mut Vec<AggExpr>,
) -> table::Result<Expr> {
    match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            if let Some(index) = output_names.resolve_expr(expr)? {
                return output_col_expr(index);
            }
            let col = resolver.resolve_column_expr(expr)?;
            let index = select_items.len();
            select_items.push(AggExpr::value(Expr::Col(col)));
            output_col_expr(index)
        }
        SqlExpr::Function(fun) => {
            let agg = parse_aggregate_function(fun, resolver)?;
            let index = select_items.len();
            select_items.push(agg);
            output_col_expr(index)
        }
        SqlExpr::Value(value) => Ok(Expr::Lit(parse_value(&value.value)?)),
        SqlExpr::Nested(inner) => parse_having_expr(inner, resolver, output_names, select_items),
        SqlExpr::UnaryOp { op, expr } => match op {
            UnaryOperator::Not | UnaryOperator::BangNot => {
                Ok(parse_having_expr(expr, resolver, output_names, select_items)?.not())
            }
            UnaryOperator::Plus => parse_having_expr(expr, resolver, output_names, select_items),
            UnaryOperator::Minus => {
                let value = parse_having_expr(expr, resolver, output_names, select_items)?;
                match value {
                    Expr::Lit(ValueLit::Integer(v)) => v
                        .checked_neg()
                        .map(crate::query::lit_i64)
                        .ok_or(table::Error::Query(QueryError::SqlUnsupported)),
                    Expr::Lit(ValueLit::Real(v)) => Ok(crate::query::lit_f64(-v)),
                    _ => Err(table::Error::Query(QueryError::SqlUnsupported)),
                }
            }
            _ => Err(table::Error::Query(QueryError::SqlUnsupported)),
        },
        SqlExpr::BinaryOp { left, op, right } => {
            let left = parse_having_expr(left, resolver, output_names, select_items)?;
            let right = parse_having_expr(right, resolver, output_names, select_items)?;
            match op {
                SqlBinaryOperator::Eq => Ok(left.eq(right)),
                SqlBinaryOperator::NotEq => Ok(left.ne(right)),
                SqlBinaryOperator::Lt => Ok(left.lt(right)),
                SqlBinaryOperator::LtEq => Ok(left.le(right)),
                SqlBinaryOperator::Gt => Ok(left.gt(right)),
                SqlBinaryOperator::GtEq => Ok(left.ge(right)),
                SqlBinaryOperator::And => Ok(left.and(right)),
                SqlBinaryOperator::Or => Ok(left.or(right)),
                _ => Err(table::Error::Query(QueryError::SqlUnsupported)),
            }
        }
        SqlExpr::IsNull(inner) => {
            Ok(parse_having_expr(inner, resolver, output_names, select_items)?.is_null())
        }
        SqlExpr::IsNotNull(inner) => {
            Ok(parse_having_expr(inner, resolver, output_names, select_items)?.is_not_null())
        }
        _ => Err(table::Error::Query(QueryError::SqlUnsupported)),
    }
}

fn output_col_expr(index: usize) -> table::Result<Expr> {
    let col = u16::try_from(index).map_err(|_| table::Error::Query(QueryError::SqlUnsupported))?;
    Ok(Expr::Col(col))
}

struct ColumnResolver {
    table_name: String,
    alias: Option<String>,
    columns: Vec<String>,
}

impl ColumnResolver {
    fn resolve_column_expr(&self, expr: &SqlExpr) -> table::Result<u16> {
        match expr {
            SqlExpr::Identifier(ident) => self.resolve_unqualified(&ident.value),
            SqlExpr::CompoundIdentifier(idents) => self.resolve_compound(idents.as_slice()),
            _ => Err(table::Error::Query(QueryError::SqlUnsupported)),
        }
    }

    fn resolve_unqualified(&self, name: &str) -> table::Result<u16> {
        let mut matches = self
            .columns
            .iter()
            .enumerate()
            .filter_map(|(idx, col)| col.eq_ignore_ascii_case(name).then_some(idx));
        let first = matches.next().ok_or(table::Error::Query(QueryError::SqlUnknownColumn))?;
        if matches.next().is_some() {
            return Err(table::Error::Query(QueryError::SqlAmbiguousColumn));
        }
        u16::try_from(first).map_err(|_| table::Error::Query(QueryError::SqlUnsupported))
    }

    fn resolve_compound(&self, parts: &[Ident]) -> table::Result<u16> {
        if parts.len() < 2 {
            return Err(table::Error::Query(QueryError::SqlUnsupported));
        }
        let qualifier = &parts[parts.len() - 2].value;
        let column = &parts[parts.len() - 1].value;
        if !self.qualifier_matches(qualifier) {
            return Err(table::Error::Query(QueryError::SqlUnknownColumn));
        }
        self.resolve_unqualified(column)
    }

    fn qualifier_matches(&self, name: &str) -> bool {
        if self.table_name.eq_ignore_ascii_case(name) {
            return true;
        }
        self.alias.as_ref().is_some_and(|alias| alias.eq_ignore_ascii_case(name))
    }
}

fn resolve_table<'db>(
    db: &'db Db,
    select: &Select,
) -> table::Result<(crate::db::Table<'db>, ColumnResolver)> {
    if select.from.len() != 1 {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    }
    let table_with_joins = &select.from[0];
    if !table_with_joins.joins.is_empty() {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    }

    let TableFactor::Table {
        name,
        alias,
        args,
        with_hints,
        version,
        with_ordinality,
        partitions,
        json_path,
        sample,
        index_hints,
    } = &table_with_joins.relation
    else {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    };
    if args.is_some()
        || !with_hints.is_empty()
        || version.is_some()
        || *with_ordinality
        || !partitions.is_empty()
        || json_path.is_some()
        || sample.is_some()
        || !index_hints.is_empty()
    {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    }
    if alias.as_ref().is_some_and(|alias| !alias.columns.is_empty()) {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    }

    let table_name = object_name_last_ident(name)?;
    let table = if table_name.eq_ignore_ascii_case("sqlite_schema") {
        db.table_from_root(PageId::ROOT)
    } else {
        db.table(table_name.as_str())?
    };
    let columns = load_table_columns(db, table_name.as_str())?;
    let alias = alias.as_ref().map(|alias| alias.name.value.to_ascii_lowercase());
    let resolver = ColumnResolver { table_name, alias, columns };

    Ok((table, resolver))
}

fn load_table_columns(db: &Db, table_name: &str) -> table::Result<Vec<String>> {
    if table_name.eq_ignore_ascii_case("sqlite_schema") {
        return Ok(["type", "name", "tbl_name", "rootpage", "sql"]
            .iter()
            .map(|s| s.to_string())
            .collect());
    }

    let sql = db.find_schema(|row| {
        if !row.kind.eq_ignore_ascii_case("table") || !row.name.eq_ignore_ascii_case(table_name) {
            return Ok(None);
        }
        Ok(Some(row.sql.as_str().map(|s| s.to_owned())))
    })?;
    let Some(sql) = sql else {
        return Err(table::Error::TableNotFound);
    };
    let Some(sql) = sql else {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    };
    let schema = parse_table_schema(sql.as_str());
    if schema.columns.is_empty() {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    }
    Ok(schema.columns)
}

enum ProjectionExpr {
    Column(u16),
    Aggregate(AggExpr),
}

fn parse_projection(
    items: &[SelectItem],
    resolver: &ColumnResolver,
) -> table::Result<Vec<ProjectionExpr>> {
    let mut projection = Vec::new();
    for item in items {
        match item {
            SelectItem::UnnamedExpr(expr) => {
                projection.push(parse_projection_expr(expr, resolver)?);
            }
            SelectItem::ExprWithAlias { expr, .. } => {
                projection.push(parse_projection_expr(expr, resolver)?);
            }
            SelectItem::Wildcard(options) => {
                validate_wildcard_options(options)?;
                projection.extend(
                    (0..resolver.columns.len()).map(|idx| ProjectionExpr::Column(idx as u16)),
                );
            }
            SelectItem::QualifiedWildcard(kind, options) => {
                validate_wildcard_options(options)?;
                match kind {
                    SelectItemQualifiedWildcardKind::ObjectName(name) => {
                        let qualifier = object_name_last_ident(name)?;
                        if !resolver.qualifier_matches(qualifier.as_str()) {
                            return Err(table::Error::Query(QueryError::SqlUnknownColumn));
                        }
                        projection.extend(
                            (0..resolver.columns.len())
                                .map(|idx| ProjectionExpr::Column(idx as u16)),
                        );
                    }
                    SelectItemQualifiedWildcardKind::Expr(_) => {
                        return Err(table::Error::Query(QueryError::SqlUnsupported));
                    }
                }
            }
        }
    }
    Ok(projection)
}

fn validate_wildcard_options(options: &WildcardAdditionalOptions) -> table::Result<()> {
    if options.opt_ilike.is_some()
        || options.opt_exclude.is_some()
        || options.opt_except.is_some()
        || options.opt_replace.is_some()
        || options.opt_rename.is_some()
    {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    }
    Ok(())
}

fn parse_projection_expr(
    expr: &SqlExpr,
    resolver: &ColumnResolver,
) -> table::Result<ProjectionExpr> {
    match expr {
        SqlExpr::Function(fun) => {
            Ok(ProjectionExpr::Aggregate(parse_aggregate_function(fun, resolver)?))
        }
        _ => Ok(ProjectionExpr::Column(resolver.resolve_column_expr(expr)?)),
    }
}

fn parse_aggregate_function(fun: &Function, resolver: &ColumnResolver) -> table::Result<AggExpr> {
    if !matches!(fun.parameters, FunctionArguments::None)
        || fun.filter.is_some()
        || fun.null_treatment.is_some()
        || fun.over.is_some()
        || !fun.within_group.is_empty()
    {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    }

    let FunctionArguments::List(args) = &fun.args else {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    };
    if args.duplicate_treatment.is_some() || !args.clauses.is_empty() {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    }

    let name = object_name_last_ident(&fun.name)?.to_ascii_uppercase();
    match name.as_str() {
        "COUNT" => parse_count_aggregate(args.args.as_slice(), resolver),
        "SUM" => parse_single_arg_aggregate(args.args.as_slice(), resolver, AggExpr::sum),
        "AVG" => parse_single_arg_aggregate(args.args.as_slice(), resolver, AggExpr::avg),
        "MIN" => parse_single_arg_aggregate(args.args.as_slice(), resolver, AggExpr::min),
        "MAX" => parse_single_arg_aggregate(args.args.as_slice(), resolver, AggExpr::max),
        _ => Err(table::Error::Query(QueryError::SqlUnsupported)),
    }
}

fn parse_count_aggregate(
    args: &[FunctionArg],
    resolver: &ColumnResolver,
) -> table::Result<AggExpr> {
    let [arg] = args else {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    };
    let FunctionArg::Unnamed(arg) = arg else {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    };
    match arg {
        FunctionArgExpr::Wildcard => Ok(AggExpr::count_star()),
        FunctionArgExpr::QualifiedWildcard(name) => {
            let qualifier = object_name_last_ident(name)?;
            if resolver.qualifier_matches(qualifier.as_str()) {
                Ok(AggExpr::count_star())
            } else {
                Err(table::Error::Query(QueryError::SqlUnknownColumn))
            }
        }
        FunctionArgExpr::Expr(expr) => Ok(AggExpr::count(parse_value_expr(expr, resolver)?)),
    }
}

fn parse_single_arg_aggregate(
    args: &[FunctionArg],
    resolver: &ColumnResolver,
    make: fn(Expr) -> AggExpr,
) -> table::Result<AggExpr> {
    let [arg] = args else {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    };
    let FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) = arg else {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    };
    Ok(make(parse_value_expr(expr, resolver)?))
}

fn parse_value_expr(expr: &SqlExpr, resolver: &ColumnResolver) -> table::Result<Expr> {
    let expr = parse_filter_expr(expr, resolver)?;
    match expr {
        Expr::Col(_) | Expr::Lit(_) => Ok(expr),
        _ => Err(table::Error::Query(QueryError::SqlUnsupported)),
    }
}

fn parse_group_by(group_by: &GroupByExpr, resolver: &ColumnResolver) -> table::Result<Vec<Expr>> {
    match group_by {
        GroupByExpr::All(_) => Err(table::Error::Query(QueryError::SqlUnsupported)),
        GroupByExpr::Expressions(exprs, modifiers) => {
            if !modifiers.is_empty() {
                return Err(table::Error::Query(QueryError::SqlUnsupported));
            }
            exprs.iter().map(|expr| parse_value_expr(expr, resolver)).collect()
        }
    }
}

fn parse_aggregate_order_by(
    order_by: Option<&SqlOrderBy>,
    resolver: &ColumnResolver,
    output_names: &OutputNameResolver,
    select_items: &mut Vec<AggExpr>,
    visible_cols: usize,
) -> table::Result<Option<Vec<OrderBy>>> {
    parse_order_by_items(order_by, |expr| {
        parse_aggregate_order_key(expr, resolver, output_names, select_items, visible_cols)
    })
}

fn parse_aggregate_order_key(
    expr: &SqlExpr,
    resolver: &ColumnResolver,
    output_names: &OutputNameResolver,
    select_items: &mut Vec<AggExpr>,
    visible_cols: usize,
) -> table::Result<u16> {
    if let Some(idx) = parse_ordinal(expr.clone())? {
        if idx >= visible_cols {
            return Err(table::Error::Query(QueryError::SqlInvalidLimitOffset));
        }
        return u16::try_from(idx).map_err(|_| table::Error::Query(QueryError::SqlUnsupported));
    }

    if let Some(idx) = output_names.resolve_expr(expr)? {
        return u16::try_from(idx).map_err(|_| table::Error::Query(QueryError::SqlUnsupported));
    }

    let item = match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            AggExpr::value(Expr::Col(resolver.resolve_column_expr(expr)?))
        }
        SqlExpr::Function(fun) => parse_aggregate_function(fun, resolver)?,
        _ => AggExpr::value(parse_value_expr(expr, resolver)?),
    };

    let idx = select_items.len();
    select_items.push(item);
    u16::try_from(idx).map_err(|_| table::Error::Query(QueryError::SqlUnsupported))
}

fn parse_order_by(
    order_by: Option<&SqlOrderBy>,
    resolver: &ColumnResolver,
    projection_cols: &[u16],
    output_names: Option<&OutputNameResolver>,
) -> table::Result<Option<Vec<OrderBy>>> {
    parse_order_by_items(order_by, |expr| {
        if let Some(idx) = parse_ordinal(expr.clone())? {
            let Some(mapped) = projection_cols.get(idx) else {
                return Err(table::Error::Query(QueryError::SqlInvalidLimitOffset));
            };
            return Ok(*mapped);
        }

        if let Some(names) = output_names
            && let Some(idx) = names.resolve_expr(expr)?
        {
            let Some(mapped) = projection_cols.get(idx) else {
                return Err(table::Error::Query(QueryError::SqlUnknownColumn));
            };
            return Ok(*mapped);
        }

        resolver.resolve_column_expr(expr)
    })
}

fn parse_order_by_items<F>(
    order_by: Option<&SqlOrderBy>,
    mut resolve_col: F,
) -> table::Result<Option<Vec<OrderBy>>>
where
    F: FnMut(&SqlExpr) -> table::Result<u16>,
{
    let Some(order_by) = order_by else {
        return Ok(None);
    };
    if order_by.interpolate.is_some() {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    }
    let OrderByKind::Expressions(exprs) = &order_by.kind else {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    };
    if exprs.is_empty() {
        return Ok(None);
    }

    let mut out = Vec::with_capacity(exprs.len());
    for expr in exprs {
        let dir = parse_order_by_item_dir(expr)?;
        let col = resolve_col(&expr.expr)?;
        out.push(OrderBy { col, dir });
    }
    Ok(Some(out))
}

#[inline]
fn parse_order_by_item_dir(expr: &SqlOrderByExpr) -> table::Result<OrderDir> {
    if expr.with_fill.is_some() || expr.options.nulls_first.is_some() {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    }
    Ok(if expr.options.asc == Some(false) { OrderDir::Desc } else { OrderDir::Asc })
}

fn parse_limit_offset(limit_clause: Option<&LimitClause>) -> table::Result<(Option<usize>, usize)> {
    let Some(limit_clause) = limit_clause else {
        return Ok((None, 0));
    };

    match limit_clause {
        LimitClause::LimitOffset { limit, offset, limit_by } => {
            if !limit_by.is_empty() {
                return Err(table::Error::Query(QueryError::SqlUnsupported));
            }
            let limit = limit.as_ref().map(parse_usize_expr).transpose()?;
            let offset =
                offset.as_ref().map(|off| parse_usize_expr(&off.value)).transpose()?.unwrap_or(0);
            Ok((limit, offset))
        }
        LimitClause::OffsetCommaLimit { offset, limit } => {
            let offset = parse_usize_expr(offset)?;
            let limit = parse_usize_expr(limit)?;
            Ok((Some(limit), offset))
        }
    }
}

fn parse_ordinal(expr: SqlExpr) -> table::Result<Option<usize>> {
    let Some(value) = parse_i64_expr(&expr)? else {
        return Ok(None);
    };
    if value <= 0 {
        return Err(table::Error::Query(QueryError::SqlInvalidLimitOffset));
    }
    usize::try_from(value - 1)
        .map(Some)
        .map_err(|_| table::Error::Query(QueryError::SqlInvalidLimitOffset))
}

fn parse_usize_expr(expr: &SqlExpr) -> table::Result<usize> {
    let Some(value) = parse_i64_expr(expr)? else {
        return Err(table::Error::Query(QueryError::SqlInvalidLimitOffset));
    };
    usize::try_from(value).map_err(|_| table::Error::Query(QueryError::SqlInvalidLimitOffset))
}

fn parse_i64_expr(expr: &SqlExpr) -> table::Result<Option<i64>> {
    match expr {
        SqlExpr::Value(value) => parse_i64_value(&value.value),
        SqlExpr::UnaryOp { op: UnaryOperator::Plus, expr } => parse_i64_expr(expr),
        SqlExpr::UnaryOp { op: UnaryOperator::Minus, expr } => {
            let Some(value) = parse_i64_expr(expr)? else {
                return Ok(None);
            };
            value
                .checked_neg()
                .map(Some)
                .ok_or(table::Error::Query(QueryError::SqlInvalidLimitOffset))
        }
        _ => Ok(None),
    }
}

fn parse_i64_value(value: &SqlValue) -> table::Result<Option<i64>> {
    match value {
        SqlValue::Number(raw, _) => {
            if raw.contains('.') || raw.contains('e') || raw.contains('E') {
                return Ok(None);
            }
            raw.parse::<i64>()
                .map(Some)
                .map_err(|_| table::Error::Query(QueryError::SqlInvalidLimitOffset))
        }
        _ => Ok(None),
    }
}

fn bool_expr(value: bool) -> Expr {
    if value {
        Expr::Lit(ValueLit::Integer(1)).eq(Expr::Lit(ValueLit::Integer(1)))
    } else {
        Expr::Lit(ValueLit::Integer(1)).eq(Expr::Lit(ValueLit::Integer(0)))
    }
}

fn apply_negated(expr: Expr, negated: bool) -> Expr {
    if negated { expr.not() } else { expr }
}

fn fold_in_expr(expr: Expr, list: Vec<Expr>, negated: bool) -> Expr {
    let mut iter = list.into_iter();
    let Some(first) = iter.next() else {
        return apply_negated(bool_expr(false), negated);
    };

    let mut out = expr.clone().eq(first);
    for value in iter {
        out = out.or(expr.clone().eq(value));
    }
    apply_negated(out, negated)
}

fn fold_between_expr(expr: Expr, low: Expr, high: Expr, negated: bool) -> Expr {
    apply_negated(expr.clone().ge(low).and(expr.le(high)), negated)
}

fn fold_like_expr(expr: Expr, pattern: Expr, negated: bool) -> Expr {
    apply_negated(expr.like(pattern), negated)
}

fn parse_filter_expr(expr: &SqlExpr, resolver: &ColumnResolver) -> table::Result<Expr> {
    match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            Ok(Expr::Col(resolver.resolve_column_expr(expr)?))
        }
        SqlExpr::Value(value) => Ok(Expr::Lit(parse_value(&value.value)?)),
        SqlExpr::Nested(inner) => parse_filter_expr(inner, resolver),
        SqlExpr::UnaryOp { op, expr } => match op {
            UnaryOperator::Not | UnaryOperator::BangNot => {
                Ok(parse_filter_expr(expr, resolver)?.not())
            }
            UnaryOperator::Plus => parse_filter_expr(expr, resolver),
            UnaryOperator::Minus => {
                let value = parse_filter_expr(expr, resolver)?;
                match value {
                    Expr::Lit(ValueLit::Integer(v)) => v
                        .checked_neg()
                        .map(crate::query::lit_i64)
                        .ok_or(table::Error::Query(QueryError::SqlUnsupported)),
                    Expr::Lit(ValueLit::Real(v)) => Ok(crate::query::lit_f64(-v)),
                    _ => Err(table::Error::Query(QueryError::SqlUnsupported)),
                }
            }
            _ => Err(table::Error::Query(QueryError::SqlUnsupported)),
        },
        SqlExpr::BinaryOp { left, op, right } => {
            let left = parse_filter_expr(left, resolver)?;
            let right = parse_filter_expr(right, resolver)?;
            match op {
                SqlBinaryOperator::Eq => Ok(left.eq(right)),
                SqlBinaryOperator::NotEq => Ok(left.ne(right)),
                SqlBinaryOperator::Lt => Ok(left.lt(right)),
                SqlBinaryOperator::LtEq => Ok(left.le(right)),
                SqlBinaryOperator::Gt => Ok(left.gt(right)),
                SqlBinaryOperator::GtEq => Ok(left.ge(right)),
                SqlBinaryOperator::And => Ok(left.and(right)),
                SqlBinaryOperator::Or => Ok(left.or(right)),
                _ => Err(table::Error::Query(QueryError::SqlUnsupported)),
            }
        }
        SqlExpr::InList { expr, list, negated } => {
            let expr = parse_filter_expr(expr, resolver)?;
            let list = list
                .iter()
                .map(|value| parse_filter_expr(value, resolver))
                .collect::<table::Result<Vec<_>>>()?;
            Ok(fold_in_expr(expr, list, *negated))
        }
        SqlExpr::Between { expr, negated, low, high } => {
            let expr = parse_filter_expr(expr, resolver)?;
            let low = parse_filter_expr(low, resolver)?;
            let high = parse_filter_expr(high, resolver)?;
            Ok(fold_between_expr(expr, low, high, *negated))
        }
        SqlExpr::Like { negated, any, expr, pattern, escape_char } => {
            if *any || escape_char.is_some() {
                return Err(table::Error::Query(QueryError::SqlUnsupported));
            }
            let expr = parse_filter_expr(expr, resolver)?;
            let pattern = parse_filter_expr(pattern, resolver)?;
            Ok(fold_like_expr(expr, pattern, *negated))
        }
        SqlExpr::IsNull(inner) => Ok(parse_filter_expr(inner, resolver)?.is_null()),
        SqlExpr::IsNotNull(inner) => Ok(parse_filter_expr(inner, resolver)?.is_not_null()),
        _ => Err(table::Error::Query(QueryError::SqlUnsupported)),
    }
}

fn parse_value(value: &SqlValue) -> table::Result<ValueLit> {
    match value {
        SqlValue::Number(raw, _) => {
            if !raw.contains('.')
                && !raw.contains('e')
                && !raw.contains('E')
                && let Ok(value) = raw.parse::<i64>()
            {
                return Ok(ValueLit::Integer(value));
            }
            raw.parse::<f64>()
                .map(ValueLit::Real)
                .map_err(|_| table::Error::Query(QueryError::SqlUnsupported))
        }
        SqlValue::Null => Ok(ValueLit::Null),
        SqlValue::Boolean(value) => Ok(ValueLit::Integer(if *value { 1 } else { 0 })),
        _ => {
            if let Some(text) = value.clone().into_string() {
                Ok(ValueLit::Text(text.into_bytes()))
            } else {
                Err(table::Error::Query(QueryError::SqlUnsupported))
            }
        }
    }
}

fn object_name_last_ident(name: &ObjectName) -> table::Result<String> {
    let Some(part) = name.0.last() else {
        return Err(table::Error::Query(QueryError::SqlUnsupported));
    };
    match part {
        ObjectNamePart::Identifier(ident) => Ok(ident.value.to_ascii_lowercase()),
        ObjectNamePart::Function(_) => Err(table::Error::Query(QueryError::SqlUnsupported)),
    }
}

#[cfg(test)]
mod tests {
    use rusqlite::Connection;
    use sqlparser::ast::{
        Expr as SqlExpr, Ident, OrderByExpr as SqlOrderByExpr, OrderByOptions, WithFill,
    };
    use tempfile::NamedTempFile;

    use super::{
        JoinTableResolver, OwnedSqlValue, choose_join_strategy_from_right_index,
        distinct_indices_for_prefix, distinct_indices_for_projection,
        distinct_indices_from_keys_with_hash_fallback_cap, parse_order_by_item_dir,
        parse_sql_distinct_hash_mem_cap_override,
    };
    use crate::db::Db;
    use crate::join::{JoinKey, JoinStrategy};
    use crate::query::OrderDir;
    use crate::table::{self, QueryError};

    fn make_db<F: FnOnce(&Connection)>(f: F) -> NamedTempFile {
        let file = NamedTempFile::new().expect("create temp db file");
        let conn = Connection::open(file.path()).expect("open temp sqlite db");
        conn.execute_batch("PRAGMA journal_mode=DELETE; PRAGMA synchronous=OFF;")
            .expect("set sqlite pragmas");
        f(&conn);
        drop(conn);
        file
    }

    #[test]
    fn parses_distinct_hash_cap_override_when_valid() {
        assert_eq!(parse_sql_distinct_hash_mem_cap_override("1"), Some(1));
        assert_eq!(parse_sql_distinct_hash_mem_cap_override("1048576"), Some(1_048_576));
        assert_eq!(parse_sql_distinct_hash_mem_cap_override(" 2097152 "), Some(2_097_152));
    }

    #[test]
    fn rejects_distinct_hash_cap_override_when_invalid() {
        assert_eq!(parse_sql_distinct_hash_mem_cap_override(""), None);
        assert_eq!(parse_sql_distinct_hash_mem_cap_override("   "), None);
        assert_eq!(parse_sql_distinct_hash_mem_cap_override("0"), None);
        assert_eq!(parse_sql_distinct_hash_mem_cap_override("-1"), None);
        assert_eq!(parse_sql_distinct_hash_mem_cap_override("abc"), None);
    }

    #[test]
    fn distinct_falls_back_to_sort_when_hash_cap_is_too_small() {
        let keys = vec![
            vec![OwnedSqlValue::Text(b"a".to_vec())],
            vec![OwnedSqlValue::Text(b"a".to_vec())],
            vec![OwnedSqlValue::Text(b"b".to_vec())],
            vec![OwnedSqlValue::Text(b"a".to_vec())],
            vec![OwnedSqlValue::Text(b"b".to_vec())],
            vec![OwnedSqlValue::Text(b"c".to_vec())],
        ];

        let keep = distinct_indices_from_keys_with_hash_fallback_cap(keys.as_slice(), 1)
            .expect("fallback distinct");
        assert_eq!(keep, vec![0, 2, 5]);
    }

    #[test]
    fn distinct_prefix_uses_only_requested_leading_columns() {
        let rows = vec![
            vec![
                OwnedSqlValue::Text(b"a".to_vec()),
                OwnedSqlValue::Integer(1),
                OwnedSqlValue::Integer(10),
            ],
            vec![
                OwnedSqlValue::Text(b"a".to_vec()),
                OwnedSqlValue::Integer(1),
                OwnedSqlValue::Integer(99),
            ],
            vec![
                OwnedSqlValue::Text(b"a".to_vec()),
                OwnedSqlValue::Integer(2),
                OwnedSqlValue::Integer(11),
            ],
            vec![
                OwnedSqlValue::Text(b"b".to_vec()),
                OwnedSqlValue::Integer(3),
                OwnedSqlValue::Integer(12),
            ],
            vec![
                OwnedSqlValue::Text(b"b".to_vec()),
                OwnedSqlValue::Integer(3),
                OwnedSqlValue::Integer(13),
            ],
        ];

        let keep = distinct_indices_for_prefix(rows.as_slice(), 2).expect("distinct prefix");
        assert_eq!(keep, vec![0, 2, 3]);
    }

    #[test]
    fn distinct_projection_uses_selected_columns_and_null_for_missing_values() {
        let rows = vec![
            vec![OwnedSqlValue::Text(b"a".to_vec())],
            vec![OwnedSqlValue::Text(b"a".to_vec()), OwnedSqlValue::Integer(1)],
            vec![
                OwnedSqlValue::Text(b"a".to_vec()),
                OwnedSqlValue::Integer(2),
                OwnedSqlValue::Integer(5),
            ],
            vec![OwnedSqlValue::Text(b"b".to_vec())],
        ];

        let keep =
            distinct_indices_for_projection(rows.as_slice(), &[0, 2]).expect("distinct projection");
        assert_eq!(keep, vec![0, 2, 3]);
    }

    #[test]
    fn order_by_item_dir_rejects_nulls_first() {
        let expr = SqlOrderByExpr {
            expr: SqlExpr::Identifier(Ident::new("col")),
            options: OrderByOptions { asc: Some(true), nulls_first: Some(true) },
            with_fill: None,
        };

        let err = parse_order_by_item_dir(&expr).expect_err("NULLS FIRST is unsupported");
        assert!(matches!(err, table::Error::Query(QueryError::SqlUnsupported)));
    }

    #[test]
    fn order_by_item_dir_rejects_with_fill_and_parses_direction() {
        let with_fill_expr = SqlOrderByExpr {
            expr: SqlExpr::Identifier(Ident::new("col")),
            options: OrderByOptions { asc: Some(false), nulls_first: None },
            with_fill: Some(WithFill { from: None, to: None, step: None }),
        };
        let err = parse_order_by_item_dir(&with_fill_expr).expect_err("WITH FILL is unsupported");
        assert!(matches!(err, table::Error::Query(QueryError::SqlUnsupported)));

        let desc_expr = SqlOrderByExpr {
            expr: SqlExpr::Identifier(Ident::new("col")),
            options: OrderByOptions { asc: Some(false), nulls_first: None },
            with_fill: None,
        };
        assert_eq!(parse_order_by_item_dir(&desc_expr).expect("DESC"), OrderDir::Desc);

        let default_expr = SqlOrderByExpr {
            expr: SqlExpr::Identifier(Ident::new("col")),
            options: OrderByOptions::default(),
            with_fill: None,
        };
        assert_eq!(parse_order_by_item_dir(&default_expr).expect("default ASC"), OrderDir::Asc);
    }

    #[test]
    fn chooses_index_nested_loop_for_right_join_key_when_index_exists() {
        let file = make_db(|conn| {
            conn.execute_batch(
                "CREATE TABLE users(id INTEGER, name TEXT);
                 CREATE TABLE orders(user_id INTEGER, amount INTEGER);
                 CREATE INDEX orders_user_id_idx ON orders(user_id);",
            )
            .expect("create schema");
        });

        let db = Db::open(file.path()).expect("open db");
        let right = JoinTableResolver {
            table_name: "orders".to_string(),
            alias: None,
            columns: vec!["user_id".to_string(), "amount".to_string()],
        };

        let strategy =
            choose_join_strategy_from_right_index(&db, &right, JoinKey::Col(0)).expect("plan");
        assert!(matches!(strategy, Some(JoinStrategy::IndexNestedLoop { index_key_col: 0, .. })));
    }

    #[test]
    fn skips_index_nested_loop_when_join_key_is_not_leading_index_column() {
        let file = make_db(|conn| {
            conn.execute_batch(
                "CREATE TABLE users(id INTEGER, name TEXT);
                 CREATE TABLE orders(user_id INTEGER, amount INTEGER);
                 CREATE INDEX orders_amount_user_idx ON orders(amount, user_id);",
            )
            .expect("create schema");
        });

        let db = Db::open(file.path()).expect("open db");
        let right = JoinTableResolver {
            table_name: "orders".to_string(),
            alias: None,
            columns: vec!["user_id".to_string(), "amount".to_string()],
        };

        let strategy =
            choose_join_strategy_from_right_index(&db, &right, JoinKey::Col(0)).expect("plan");
        assert!(strategy.is_none());
    }
}
