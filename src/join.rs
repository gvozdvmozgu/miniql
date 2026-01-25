use std::collections::hash_map::Entry;

use rustc_hash::FxHashMap;

use crate::index::{self, IndexCursor, IndexScratch};
use crate::pager::{PageId, Pager};
use crate::query::{CompiledScan, Row, Scan, ScanScratch};
use crate::table::{self, Value, ValueRef, ValueRefRaw};

pub type Result<T> = table::Result<T>;

#[derive(Debug)]
pub enum JoinError {
    UnsupportedJoinKeyType,
    IndexKeyNotComparable,
    MissingIndexRowId,
    HashMemoryLimitExceeded,
    MissingJoinCondition,
    UnsupportedJoinType,
    UnsupportedJoinStrategy,
    LeftJoinMissingRightColumns,
}

impl std::fmt::Display for JoinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedJoinKeyType => f.write_str("Unsupported join key type"),
            Self::IndexKeyNotComparable => f.write_str("Index key is not comparable to join key"),
            Self::MissingIndexRowId => f.write_str("Index record does not end with a rowid"),
            Self::HashMemoryLimitExceeded => f.write_str("Hash join memory limit exceeded"),
            Self::MissingJoinCondition => f.write_str("Join condition is missing"),
            Self::UnsupportedJoinType => f.write_str("Join type is not supported"),
            Self::UnsupportedJoinStrategy => f.write_str("Join strategy is not supported"),
            Self::LeftJoinMissingRightColumns => {
                f.write_str("Left join could not determine right-side column count")
            }
        }
    }
}

impl std::error::Error for JoinError {}

#[derive(Debug, Clone, Copy)]
pub enum JoinType {
    Inner,
    Left,
}

#[derive(Debug, Clone, Copy)]
pub enum JoinKey {
    Col(u16),
    RowId,
}

#[derive(Debug, Clone, Copy)]
pub enum JoinStrategy {
    Auto,
    IndexNestedLoop { index_root: PageId, index_key_col: u16 },
    HashJoin,
    NestedLoopScan,
}

pub struct Join<'db> {
    join_type: JoinType,
    left: Scan<'db>,
    right: Scan<'db>,
    left_key: Option<JoinKey>,
    right_key: Option<JoinKey>,
    strategy: JoinStrategy,
    hash_mem_limit: Option<usize>,
    project_left: Option<Vec<u16>>,
    project_right: Option<Vec<u16>>,
}

// For left joins without a match, right_rowid is set to 0 and right values are
// NULL.
const NULL_ROWID: i64 = 0;

impl<'db> Join<'db> {
    pub fn new(join_type: JoinType, left: Scan<'db>, right: Scan<'db>) -> Self {
        Self {
            join_type,
            left,
            right,
            left_key: None,
            right_key: None,
            strategy: JoinStrategy::Auto,
            hash_mem_limit: None,
            project_left: None,
            project_right: None,
        }
    }

    pub fn on(mut self, left: JoinKey, right: JoinKey) -> Self {
        self.left_key = Some(left);
        self.right_key = Some(right);
        self
    }

    pub fn strategy(mut self, s: JoinStrategy) -> Self {
        self.strategy = s;
        self
    }

    pub fn hash_mem_limit(mut self, bytes: usize) -> Self {
        self.hash_mem_limit = Some(bytes);
        self
    }

    pub fn project_left<const N: usize>(mut self, cols: [u16; N]) -> Self {
        self.project_left = Some(cols.to_vec());
        self
    }

    pub fn project_right<const N: usize>(mut self, cols: [u16; N]) -> Self {
        self.project_right = Some(cols.to_vec());
        self
    }

    pub fn for_each<F>(self, scratch: &mut JoinScratch, mut cb: F) -> Result<()>
    where
        F: for<'row> FnMut(JoinedRow<'row>) -> Result<()>,
    {
        let left_key = self.left_key.ok_or(JoinError::MissingJoinCondition)?;
        let right_key = self.right_key.ok_or(JoinError::MissingJoinCondition)?;

        let ((mut left_scan, left_meta), (mut right_scan, right_meta)) = build_sides(
            self.left,
            self.right,
            left_key,
            right_key,
            self.project_left,
            self.project_right,
        )?;

        let (
            left_scan_scratch,
            right_scan_scratch,
            right_decoded,
            right_overflow,
            index_scratch,
            right_nulls,
        ) = scratch.split_mut();

        let right_null_len = if matches!(self.join_type, JoinType::Left) {
            Some(resolve_right_null_len(
                &mut right_scan,
                &right_meta,
                right_decoded,
                right_overflow,
            )?)
        } else {
            None
        };

        match self.strategy {
            JoinStrategy::IndexNestedLoop { index_root, index_key_col } => {
                if !matches!(right_key, JoinKey::Col(_)) {
                    return Err(JoinError::UnsupportedJoinKeyType.into());
                }
                index_nested_loop(
                    &mut left_scan,
                    &left_meta,
                    &mut right_scan,
                    &right_meta,
                    index_root,
                    index_key_col,
                    left_scan_scratch,
                    right_decoded,
                    right_overflow,
                    index_scratch,
                    right_nulls,
                    right_null_len,
                    &mut cb,
                )
            }
            JoinStrategy::HashJoin => hash_join(
                &mut left_scan,
                &left_meta,
                &mut right_scan,
                &right_meta,
                self.hash_mem_limit,
                left_scan_scratch,
                right_scan_scratch,
                right_decoded,
                right_overflow,
                right_nulls,
                right_null_len,
                &mut cb,
            ),
            JoinStrategy::NestedLoopScan => nested_loop_scan(
                &mut left_scan,
                &left_meta,
                &mut right_scan,
                &right_meta,
                left_scan_scratch,
                right_scan_scratch,
                right_nulls,
                right_null_len,
                &mut cb,
            ),
            JoinStrategy::Auto => {
                if matches!(right_key, JoinKey::RowId) {
                    rowid_nested_loop(
                        &mut left_scan,
                        &left_meta,
                        &mut right_scan,
                        &right_meta,
                        left_scan_scratch,
                        right_decoded,
                        right_overflow,
                        right_nulls,
                        right_null_len,
                        &mut cb,
                    )
                } else if let JoinKey::Col(col) = right_key {
                    if let Some((index_root, index_key_col)) =
                        discover_index_for_join(right_scan.pager(), right_scan.root(), col)?
                    {
                        index_nested_loop(
                            &mut left_scan,
                            &left_meta,
                            &mut right_scan,
                            &right_meta,
                            index_root,
                            index_key_col,
                            left_scan_scratch,
                            right_decoded,
                            right_overflow,
                            index_scratch,
                            right_nulls,
                            right_null_len,
                            &mut cb,
                        )
                    } else {
                        hash_join(
                            &mut left_scan,
                            &left_meta,
                            &mut right_scan,
                            &right_meta,
                            self.hash_mem_limit,
                            left_scan_scratch,
                            right_scan_scratch,
                            right_decoded,
                            right_overflow,
                            right_nulls,
                            right_null_len,
                            &mut cb,
                        )
                    }
                } else {
                    hash_join(
                        &mut left_scan,
                        &left_meta,
                        &mut right_scan,
                        &right_meta,
                        self.hash_mem_limit,
                        left_scan_scratch,
                        right_scan_scratch,
                        right_decoded,
                        right_overflow,
                        right_nulls,
                        right_null_len,
                        &mut cb,
                    )
                }
            }
        }
    }
}

pub struct JoinedRow<'row> {
    pub left_rowid: i64,
    pub right_rowid: i64,
    pub left: Row<'row>,
    pub right: Row<'row>,
}

#[derive(Debug)]
pub struct JoinScratch {
    left_scan: ScanScratch,
    right_scan: ScanScratch,
    right_decoded: Vec<ValueRefRaw>,
    right_overflow: Vec<u8>,
    index: IndexScratch,
    right_nulls: Vec<ValueRefRaw>,
}

impl JoinScratch {
    pub fn new() -> Self {
        Self {
            left_scan: ScanScratch::new(),
            right_scan: ScanScratch::new(),
            right_decoded: Vec::new(),
            right_overflow: Vec::new(),
            index: IndexScratch::new(),
            right_nulls: Vec::new(),
        }
    }

    pub fn with_capacity(left_values: usize, right_values: usize, overflow: usize) -> Self {
        Self {
            left_scan: ScanScratch::with_capacity(left_values, overflow),
            right_scan: ScanScratch::with_capacity(right_values, overflow),
            right_decoded: Vec::with_capacity(right_values),
            right_overflow: Vec::with_capacity(overflow),
            index: IndexScratch::with_capacity(right_values, overflow),
            right_nulls: Vec::with_capacity(right_values),
        }
    }

    fn split_mut(
        &mut self,
    ) -> (
        &mut ScanScratch,
        &mut ScanScratch,
        &mut Vec<ValueRefRaw>,
        &mut Vec<u8>,
        &mut IndexScratch,
        &mut Vec<ValueRefRaw>,
    ) {
        (
            &mut self.left_scan,
            &mut self.right_scan,
            &mut self.right_decoded,
            &mut self.right_overflow,
            &mut self.index,
            &mut self.right_nulls,
        )
    }
}

impl Default for JoinScratch {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
struct SideMeta {
    join_key: JoinKey,
    join_key_index: Option<usize>,
    proj_map: Option<Vec<usize>>,
    output_map: Option<Vec<usize>>,
}

impl SideMeta {
    fn output_row<'row>(&'row self, values: &'row [ValueRefRaw]) -> Row<'row> {
        if let Some(map) = self.output_map.as_deref() {
            return Row::from_raw(values, Some(map));
        }
        if let Some(map) = self.proj_map.as_deref() {
            return Row::from_raw(values, Some(map));
        }
        Row::from_raw(values, None)
    }

    fn join_key<'row>(&self, rowid: i64, row: &Row<'row>) -> Result<Option<ValueRef<'row>>> {
        match self.join_key {
            JoinKey::RowId => Ok(Some(ValueRef::Integer(rowid))),
            JoinKey::Col(_) => {
                let idx = self.join_key_index.ok_or(JoinError::UnsupportedJoinKeyType)?;
                match row.get(idx) {
                    Some(ValueRef::Null) | None => Ok(None),
                    Some(value) => Ok(Some(value)),
                }
            }
        }
    }
}

fn build_sides<'db>(
    left: Scan<'db>,
    right: Scan<'db>,
    left_key: JoinKey,
    right_key: JoinKey,
    project_left: Option<Vec<u16>>,
    project_right: Option<Vec<u16>>,
) -> Result<((CompiledScan<'db>, SideMeta), (CompiledScan<'db>, SideMeta))> {
    let left_state = build_side(left, left_key, project_left)?;
    let right_state = build_side(right, right_key, project_right)?;
    Ok((left_state, right_state))
}

fn build_side<'db>(
    scan: Scan<'db>,
    join_key: JoinKey,
    output_override: Option<Vec<u16>>,
) -> Result<(CompiledScan<'db>, SideMeta)> {
    let output_cols = output_override.or_else(|| scan.projection().map(|cols| cols.to_vec()));
    let mut scan_proj = output_cols.clone();
    if let (Some(cols), JoinKey::Col(col)) = (scan_proj.as_mut(), join_key)
        && !cols.contains(&col)
    {
        cols.push(col);
    }

    let join_key_index = match join_key {
        JoinKey::RowId => None,
        JoinKey::Col(col) => match scan_proj.as_ref() {
            Some(cols) => Some(
                cols.iter()
                    .position(|value| *value == col)
                    .ok_or(JoinError::UnsupportedJoinKeyType)?,
            ),
            None => Some(col as usize),
        },
    };

    let scan = scan.with_projection_override(scan_proj);
    let compiled = scan.compile()?;
    let output_len = output_cols.as_ref().map(|cols| cols.len());
    let proj_map = compiled.proj_map().map(|map| map.to_vec());
    let output_map = match (output_len, proj_map.as_deref()) {
        (Some(len), Some(map)) if len < map.len() => Some(map[..len].to_vec()),
        (Some(len), None) => Some((0..len).collect()),
        _ => None,
    };

    let meta = SideMeta { join_key, join_key_index, proj_map, output_map };
    Ok((compiled, meta))
}

#[allow(clippy::too_many_arguments)]
fn index_nested_loop<F>(
    left_scan: &mut CompiledScan<'_>,
    left_meta: &SideMeta,
    right_scan: &mut CompiledScan<'_>,
    right_meta: &SideMeta,
    index_root: PageId,
    index_key_col: u16,
    left_scratch: &mut ScanScratch,
    right_decoded: &mut Vec<ValueRefRaw>,
    right_overflow: &mut Vec<u8>,
    index_scratch: &mut IndexScratch,
    right_nulls: &mut Vec<ValueRefRaw>,
    right_null_len: Option<usize>,
    cb: &mut F,
) -> Result<()>
where
    F: for<'row> FnMut(JoinedRow<'row>) -> Result<()>,
{
    let right_root = right_scan.root();
    let pager = right_scan.pager();
    let mut cursor = IndexCursor::new(pager, index_root, index_key_col, index_scratch);

    left_scan.for_each(left_scratch, |left_rowid, left_row| {
        let left_out = left_meta.output_row(left_row.values_raw());
        let Some(left_key_value) = left_meta.join_key(left_rowid, &left_row)? else {
            return emit_left_only(left_rowid, left_out, right_nulls, right_null_len, cb);
        };
        let mut matched = false;

        if !cursor.seek_ge(left_key_value)? {
            if let Some(len) = right_null_len {
                let right_out = null_right_row(right_nulls, len);
                cb(JoinedRow {
                    left_rowid,
                    right_rowid: NULL_ROWID,
                    left: left_out,
                    right: right_out,
                })?;
            }
            return Ok(());
        }

        while cursor.key_eq(left_key_value)? {
            let right_rowid = cursor.current_rowid()?;
            if let Some(payload) =
                table::lookup_rowid_payload(pager, right_root, right_rowid, right_overflow)?
                && let Some(right_row) = right_scan.eval_payload(payload, right_decoded)?
            {
                let right_out = right_meta.output_row(right_row.values_raw());
                cb(JoinedRow { left_rowid, right_rowid, left: left_out, right: right_out })?;
                matched = true;
            }

            if !cursor.next()? {
                break;
            }
        }

        if !matched && let Some(len) = right_null_len {
            let right_out = null_right_row(right_nulls, len);
            cb(JoinedRow {
                left_rowid,
                right_rowid: NULL_ROWID,
                left: left_out,
                right: right_out,
            })?;
        }

        Ok(())
    })?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn rowid_nested_loop<F>(
    left_scan: &mut CompiledScan<'_>,
    left_meta: &SideMeta,
    right_scan: &mut CompiledScan<'_>,
    right_meta: &SideMeta,
    left_scratch: &mut ScanScratch,
    right_decoded: &mut Vec<ValueRefRaw>,
    right_overflow: &mut Vec<u8>,
    right_nulls: &mut Vec<ValueRefRaw>,
    right_null_len: Option<usize>,
    cb: &mut F,
) -> Result<()>
where
    F: for<'row> FnMut(JoinedRow<'row>) -> Result<()>,
{
    let right_root = right_scan.root();
    let pager = right_scan.pager();

    left_scan.for_each(left_scratch, |left_rowid, left_row| {
        let left_out = left_meta.output_row(left_row.values_raw());
        let Some(left_key_value) = left_meta.join_key(left_rowid, &left_row)? else {
            return emit_left_only(left_rowid, left_out, right_nulls, right_null_len, cb);
        };

        let ValueRef::Integer(target_rowid) = left_key_value else {
            return emit_left_only(left_rowid, left_out, right_nulls, right_null_len, cb);
        };
        if let Some(payload) =
            table::lookup_rowid_payload(pager, right_root, target_rowid, right_overflow)?
            && let Some(right_row) = right_scan.eval_payload(payload, right_decoded)?
        {
            let right_out = right_meta.output_row(right_row.values_raw());
            cb(JoinedRow {
                left_rowid,
                right_rowid: target_rowid,
                left: left_out,
                right: right_out,
            })?;
            return Ok(());
        }

        if let Some(len) = right_null_len {
            let right_out = null_right_row(right_nulls, len);
            cb(JoinedRow {
                left_rowid,
                right_rowid: NULL_ROWID,
                left: left_out,
                right: right_out,
            })?;
        }
        Ok(())
    })?;

    Ok(())
}

#[derive(Debug, Clone)]
enum RowIdList {
    One(i64),
    Many(Vec<i64>),
}

impl RowIdList {
    #[inline(always)]
    fn new(v: i64) -> Self {
        RowIdList::One(v)
    }

    #[inline(always)]
    fn push(&mut self, v: i64) -> (usize, usize) {
        match self {
            RowIdList::One(first) => {
                let vec = vec![*first, v];
                let new = vec.capacity();
                *self = RowIdList::Many(vec);
                (0, new)
            }
            RowIdList::Many(vec) => {
                let old = vec.capacity();
                vec.push(v);
                (old, vec.capacity())
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn hash_join<F>(
    left_scan: &mut CompiledScan<'_>,
    left_meta: &SideMeta,
    right_scan: &mut CompiledScan<'_>,
    right_meta: &SideMeta,
    mem_limit: Option<usize>,
    left_scratch: &mut ScanScratch,
    right_scratch: &mut ScanScratch,
    right_decoded: &mut Vec<ValueRefRaw>,
    right_overflow: &mut Vec<u8>,
    right_nulls: &mut Vec<ValueRefRaw>,
    right_null_len: Option<usize>,
    cb: &mut F,
) -> Result<()>
where
    F: for<'row> FnMut(JoinedRow<'row>) -> Result<()>,
{
    let mut mem = MemTracker::new(mem_limit);
    let mut numeric: FxHashMap<u64, RowIdList> = FxHashMap::default();
    let mut text: FxHashMap<Box<[u8]>, RowIdList> = FxHashMap::default();
    let mut blob: FxHashMap<Box<[u8]>, RowIdList> = FxHashMap::default();

    right_scan.for_each(right_scratch, |rowid, row| {
        let Some(key_ref) = join_key_ref(right_meta, rowid, &row)? else {
            return Ok(());
        };
        match key_ref {
            HashKeyRef::Number(bits) => match numeric.entry(bits) {
                Entry::Occupied(mut e) => {
                    let (old, new) = e.get_mut().push(rowid);
                    mem.charge_capacity_growth(old, new)?;
                }
                Entry::Vacant(e) => {
                    mem.charge(std::mem::size_of::<u64>())?;
                    e.insert(RowIdList::new(rowid));
                }
            },
            HashKeyRef::Text(bytes) => match text.entry(bytes.to_vec().into_boxed_slice()) {
                Entry::Occupied(mut e) => {
                    let (old, new) = e.get_mut().push(rowid);
                    mem.charge_capacity_growth(old, new)?;
                }
                Entry::Vacant(e) => {
                    mem.charge(bytes.len())?;
                    e.insert(RowIdList::new(rowid));
                }
            },
            HashKeyRef::Blob(bytes) => match blob.entry(bytes.to_vec().into_boxed_slice()) {
                Entry::Occupied(mut e) => {
                    let (old, new) = e.get_mut().push(rowid);
                    mem.charge_capacity_growth(old, new)?;
                }
                Entry::Vacant(e) => {
                    mem.charge(bytes.len())?;
                    e.insert(RowIdList::new(rowid));
                }
            },
        }
        Ok(())
    })?;

    let right_root = right_scan.root();
    let pager = right_scan.pager();

    left_scan.for_each(left_scratch, |left_rowid, left_row| {
        let left_out = left_meta.output_row(left_row.values_raw());
        let Some(key_ref) = join_key_ref(left_meta, left_rowid, &left_row)? else {
            return emit_left_only(left_rowid, left_out, right_nulls, right_null_len, cb);
        };
        let rowids = match key_ref {
            HashKeyRef::Number(bits) => numeric.get(&bits),
            HashKeyRef::Text(bytes) => text.get(bytes),
            HashKeyRef::Blob(bytes) => blob.get(bytes),
        };

        if let Some(rowids) = rowids {
            let mut matched = false;
            match rowids {
                RowIdList::One(v) => {
                    let right_rowid = *v;
                    if let Some(payload) =
                        table::lookup_rowid_payload(pager, right_root, right_rowid, right_overflow)?
                        && let Some(right_row) =
                            right_scan.eval_payload_with_filters(payload, right_decoded, false)?
                    {
                        let right_out = right_meta.output_row(right_row.values_raw());
                        cb(JoinedRow {
                            left_rowid,
                            right_rowid,
                            left: left_out,
                            right: right_out,
                        })?;
                        matched = true;
                    }
                }
                RowIdList::Many(vs) => {
                    for &right_rowid in vs.iter() {
                        if let Some(payload) = table::lookup_rowid_payload(
                            pager,
                            right_root,
                            right_rowid,
                            right_overflow,
                        )? && let Some(right_row) =
                            right_scan.eval_payload_with_filters(payload, right_decoded, false)?
                        {
                            let right_out = right_meta.output_row(right_row.values_raw());
                            cb(JoinedRow {
                                left_rowid,
                                right_rowid,
                                left: left_out,
                                right: right_out,
                            })?;
                            matched = true;
                        }
                    }
                }
            }
            if !matched && let Some(len) = right_null_len {
                let right_out = null_right_row(right_nulls, len);
                cb(JoinedRow {
                    left_rowid,
                    right_rowid: NULL_ROWID,
                    left: left_out,
                    right: right_out,
                })?;
            }
        } else if let Some(len) = right_null_len {
            let right_out = null_right_row(right_nulls, len);
            cb(JoinedRow {
                left_rowid,
                right_rowid: NULL_ROWID,
                left: left_out,
                right: right_out,
            })?;
        }

        Ok(())
    })?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn nested_loop_scan<F>(
    left_scan: &mut CompiledScan<'_>,
    left_meta: &SideMeta,
    right_scan: &mut CompiledScan<'_>,
    right_meta: &SideMeta,
    left_scratch: &mut ScanScratch,
    right_scratch: &mut ScanScratch,
    right_nulls: &mut Vec<ValueRefRaw>,
    right_null_len: Option<usize>,
    cb: &mut F,
) -> Result<()>
where
    F: for<'row> FnMut(JoinedRow<'row>) -> Result<()>,
{
    left_scan.for_each(left_scratch, |left_rowid, left_row| {
        let left_out = left_meta.output_row(left_row.values_raw());
        let Some(left_key_value) = left_meta.join_key(left_rowid, &left_row)? else {
            return emit_left_only(left_rowid, left_out, right_nulls, right_null_len, cb);
        };
        let mut matched = false;

        right_scan.for_each(right_scratch, |right_rowid, right_row| {
            let Some(right_key_value) = right_meta.join_key(right_rowid, &right_row)? else {
                return Ok(());
            };
            if join_keys_equal(left_key_value, right_key_value) {
                let right_out = right_meta.output_row(right_row.values_raw());
                cb(JoinedRow { left_rowid, right_rowid, left: left_out, right: right_out })?;
                matched = true;
            }
            Ok(())
        })?;

        if !matched && let Some(len) = right_null_len {
            let right_out = null_right_row(right_nulls, len);
            cb(JoinedRow {
                left_rowid,
                right_rowid: NULL_ROWID,
                left: left_out,
                right: right_out,
            })?;
        }

        Ok(())
    })?;

    Ok(())
}

fn join_key_ref<'row>(
    side: &SideMeta,
    rowid: i64,
    row: &Row<'row>,
) -> Result<Option<HashKeyRef<'row>>> {
    let Some(value) = side.join_key(rowid, row)? else {
        return Ok(None);
    };
    hash_key_from_value(value)
}

fn join_keys_equal(left: ValueRef<'_>, right: ValueRef<'_>) -> bool {
    match (left, right) {
        (ValueRef::Null, _) | (_, ValueRef::Null) => false,
        (ValueRef::Integer(l), ValueRef::Integer(r)) => l == r,
        (ValueRef::Integer(l), ValueRef::Real(r)) => (l as f64) == r,
        (ValueRef::Real(l), ValueRef::Integer(r)) => l == (r as f64),
        (ValueRef::Real(l), ValueRef::Real(r)) => l == r,
        (ValueRef::TextBytes(l), ValueRef::TextBytes(r)) => l == r,
        (ValueRef::Blob(l), ValueRef::Blob(r)) => l == r,
        _ => false,
    }
}

#[derive(Debug, Clone, Copy)]
enum HashKeyRef<'a> {
    Number(u64),
    Text(&'a [u8]),
    Blob(&'a [u8]),
}

fn hash_key_from_value<'row>(value: ValueRef<'row>) -> Result<Option<HashKeyRef<'row>>> {
    Ok(match value {
        ValueRef::Null => None,
        ValueRef::Integer(value) => Some(HashKeyRef::Number((value as f64).to_bits())),
        ValueRef::Real(value) => Some(HashKeyRef::Number(value.to_bits())),
        ValueRef::TextBytes(bytes) => Some(HashKeyRef::Text(bytes)),
        ValueRef::Blob(bytes) => Some(HashKeyRef::Blob(bytes)),
    })
}

fn discover_index_for_join(
    pager: &Pager,
    table_root: PageId,
    join_col: u16,
) -> Result<Option<(PageId, u16)>> {
    let rows = table::read_table(pager, PageId::ROOT)?;
    let mut table_name: Option<String> = None;
    let mut table_sql: Option<String> = None;

    for row in &rows {
        if row.values.len() < 5 {
            continue;
        }
        let row_type = row.values[0].as_text();
        if row_type != Some("table") {
            continue;
        }
        let rootpage = row.values[3].as_integer();
        if rootpage != Some(table_root.into_inner() as i64) {
            continue;
        }
        table_name = row.values[1].as_text().map(|s| s.to_owned());
        table_sql = row.values[4].as_text().map(|s| s.to_owned());
        break;
    }

    let Some(table_name) = table_name else {
        return Ok(None);
    };
    let Some(table_sql) = table_sql else {
        return Ok(None);
    };

    let schema = parse_table_schema(&table_sql);
    if schema.without_rowid {
        return Ok(None);
    }

    let join_col_name = schema.columns.get(join_col as usize).cloned();
    let Some(join_col_name) = join_col_name else {
        return Ok(None);
    };

    for row in &rows {
        if row.values.len() < 5 {
            continue;
        }
        let row_type = row.values[0].as_text();
        if row_type != Some("index") {
            continue;
        }
        let tbl_name = row.values[2].as_text();
        if tbl_name.map(|s| s.eq_ignore_ascii_case(&table_name)) != Some(true) {
            continue;
        }
        let rootpage = row.values[3].as_integer().and_then(|v| u32::try_from(v).ok());
        let Some(rootpage) = rootpage else {
            continue;
        };
        let sql = row.values[4].as_text();
        let Some(sql) = sql else {
            continue;
        };

        let Some(index_cols) = parse_index_columns(sql) else {
            continue;
        };

        for (idx, col) in index_cols.iter().enumerate() {
            if col.eq_ignore_ascii_case(&join_col_name) {
                return Ok(Some((PageId::new(rootpage), idx as u16)));
            }
        }
    }

    discover_autoindex_for_join(pager, &rows, &table_name, &schema, &join_col_name)
}

struct TableSchema {
    columns: Vec<String>,
    unique_indexes: Vec<Vec<String>>,
    without_rowid: bool,
}

struct ColumnConstraintInfo {
    name: String,
    unique_index: bool,
}

fn parse_table_schema(sql: &str) -> TableSchema {
    let without_rowid = contains_token_sequence(sql, &["WITHOUT", "ROWID"]);
    let Some(inner) = extract_parenthesized(sql) else {
        return TableSchema { columns: Vec::new(), unique_indexes: Vec::new(), without_rowid };
    };
    let mut columns = Vec::new();
    let mut unique_indexes = Vec::new();
    for part in split_top_level(inner) {
        if is_table_constraint(part) {
            if let Some(cols) = parse_table_constraint_index_cols(part) {
                unique_indexes.push(cols);
            }
            continue;
        }
        if let Some(info) = parse_column_def(part) {
            columns.push(info.name.clone());
            if info.unique_index {
                unique_indexes.push(vec![info.name]);
            }
        }
    }
    TableSchema { columns, unique_indexes, without_rowid }
}

fn parse_index_columns(sql: &str) -> Option<Vec<String>> {
    if sql.to_ascii_uppercase().contains(" WHERE ") {
        return None;
    }

    let paren_start = find_on_paren(sql)?;
    let inner = extract_parenthesized_at(sql, paren_start)?;
    let mut cols = Vec::new();
    for part in split_top_level(inner) {
        let name = parse_identifier(part)?;
        cols.push(name.to_ascii_lowercase());
    }
    Some(cols)
}

fn discover_autoindex_for_join(
    pager: &Pager,
    rows: &[table::TableRow],
    table_name: &str,
    schema: &TableSchema,
    join_col_name: &str,
) -> Result<Option<(PageId, u16)>> {
    let mut autoindexes = Vec::new();
    for row in rows {
        if row.values.len() < 5 {
            continue;
        }
        let row_type = row.values[0].as_text();
        if row_type != Some("index") {
            continue;
        }
        let tbl_name = row.values[2].as_text();
        if tbl_name.map(|s| s.eq_ignore_ascii_case(table_name)) != Some(true) {
            continue;
        }
        match row.values[4] {
            Value::Null => {}
            _ => continue,
        }
        let rootpage = row.values[3].as_integer().and_then(|v| u32::try_from(v).ok());
        let Some(rootpage) = rootpage else {
            continue;
        };
        autoindexes.push(PageId::new(rootpage));
    }

    if autoindexes.is_empty() {
        return Ok(None);
    }

    let mut constraints = Vec::new();
    for cols in &schema.unique_indexes {
        let Some(pos) = cols.iter().position(|c| c.eq_ignore_ascii_case(join_col_name)) else {
            continue;
        };
        constraints.push((cols, pos));
    }

    if constraints.is_empty() {
        return Ok(None);
    }

    if autoindexes.len() == 1 && constraints.len() == 1 {
        return Ok(Some((autoindexes[0], constraints[0].1 as u16)));
    }

    let mut index_scratch = IndexScratch::new();
    let mut autoindex_lens = Vec::new();
    for root in &autoindexes {
        if let Some(len) = index::index_key_len(pager, *root, &mut index_scratch)? {
            autoindex_lens.push((*root, len));
        }
    }

    let mut matches = Vec::new();
    for (cols, pos) in constraints {
        let len = cols.len();
        let roots: Vec<PageId> =
            autoindex_lens.iter().filter(|(_, l)| *l == len).map(|(r, _)| *r).collect();
        if roots.len() == 1 {
            matches.push((roots[0], pos as u16));
        }
    }

    matches.sort_by_key(|(root, pos)| (root.into_inner(), *pos));
    matches.dedup();

    if matches.len() == 1 {
        return Ok(Some(matches[0]));
    }

    Ok(None)
}

fn parse_table_constraint_index_cols(part: &str) -> Option<Vec<String>> {
    let has_primary = contains_token_sequence(part, &["PRIMARY", "KEY"]);
    let has_unique = contains_token(part, "UNIQUE");
    if !has_primary && !has_unique {
        return None;
    }
    let inner = extract_parenthesized(part)?;
    let mut cols = Vec::new();
    for item in split_top_level(inner) {
        let name = parse_identifier(item)?;
        cols.push(name.to_ascii_lowercase());
    }
    if cols.is_empty() { None } else { Some(cols) }
}

fn is_table_constraint(part: &str) -> bool {
    let word = first_word(part);
    matches!(
        word.as_deref(),
        Some("CONSTRAINT") | Some("PRIMARY") | Some("UNIQUE") | Some("CHECK") | Some("FOREIGN")
    )
}

fn first_word(part: &str) -> Option<String> {
    let bytes = part.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i >= bytes.len() {
        return None;
    }
    if matches!(bytes[i], b'"' | b'`' | b'[' | b'\'') {
        return None;
    }
    let start = i;
    while i < bytes.len() && is_ident_char(bytes[i]) {
        i += 1;
    }
    if start == i {
        return None;
    }
    Some(part[start..i].to_ascii_uppercase())
}

fn parse_column_def(part: &str) -> Option<ColumnConstraintInfo> {
    let (name, end) = parse_identifier_span(part)?;
    let name = name.to_ascii_lowercase();
    let rest = part[end..].trim_start();
    let (type_name, rest) = parse_optional_type(rest);
    let has_primary = contains_token_sequence(rest, &["PRIMARY", "KEY"]);
    let has_unique = contains_token(rest, "UNIQUE");
    let integer_primary = type_name.as_deref() == Some("INTEGER") && has_primary;
    let unique_index = (has_primary || has_unique) && !integer_primary;
    Some(ColumnConstraintInfo { name, unique_index })
}

fn parse_optional_type(rest: &str) -> (Option<String>, &str) {
    let Some((token, end)) = parse_identifier_span(rest) else {
        return (None, rest);
    };
    let upper = token.to_ascii_uppercase();
    if is_constraint_keyword(&upper) {
        return (None, rest);
    }
    (Some(upper), rest[end..].trim_start())
}

fn is_constraint_keyword(token: &str) -> bool {
    matches!(
        token,
        "CONSTRAINT"
            | "PRIMARY"
            | "UNIQUE"
            | "NOT"
            | "NULL"
            | "CHECK"
            | "DEFAULT"
            | "COLLATE"
            | "REFERENCES"
            | "GENERATED"
            | "AS"
            | "STORED"
            | "VIRTUAL"
            | "ON"
            | "AUTOINCREMENT"
    )
}

fn parse_identifier(part: &str) -> Option<String> {
    parse_identifier_span(part).map(|(token, _)| token)
}

fn parse_identifier_span(part: &str) -> Option<(String, usize)> {
    let bytes = part.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i >= bytes.len() {
        return None;
    }
    let first = bytes[i];
    if first == b'(' {
        return None;
    }
    if first == b'"' || first == b'`' || first == b'[' {
        let (token, end) = parse_quoted(bytes, i)?;
        let token = strip_qualifier(&token);
        return Some((token, end));
    }
    let start = i;
    while i < bytes.len()
        && !bytes[i].is_ascii_whitespace()
        && !matches!(bytes[i], b'(' | b',' | b')')
    {
        i += 1;
    }
    if start == i {
        return None;
    }
    let token = &part[start..i];
    let token = strip_qualifier(token);
    Some((token.to_owned(), i))
}

fn parse_quoted(bytes: &[u8], start: usize) -> Option<(String, usize)> {
    let quote = bytes[start];
    let end_quote = if quote == b'[' { b']' } else { quote };
    let mut i = start + 1;
    let mut out = Vec::new();
    while i < bytes.len() {
        let b = bytes[i];
        if b == end_quote {
            if end_quote == b'"' && i + 1 < bytes.len() && bytes[i + 1] == end_quote {
                out.push(end_quote);
                i += 2;
                continue;
            }
            return Some((String::from_utf8_lossy(&out).into_owned(), i + 1));
        }
        out.push(b);
        i += 1;
    }
    None
}

fn strip_qualifier(name: &str) -> String {
    match name.rsplit_once('.') {
        Some((_prefix, suffix)) => suffix.to_owned(),
        None => name.to_owned(),
    }
}

fn split_top_level(input: &str) -> Vec<&str> {
    let bytes = input.as_bytes();
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut depth = 0u32;
    let mut i = 0usize;
    let mut in_single = false;
    let mut in_double = false;
    let mut in_backtick = false;
    let mut in_bracket = false;

    while i < bytes.len() {
        let b = bytes[i];
        if in_single {
            if b == b'\'' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                    i += 2;
                    continue;
                }
                in_single = false;
            }
            i += 1;
            continue;
        }
        if in_double {
            if b == b'"' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                    i += 2;
                    continue;
                }
                in_double = false;
            }
            i += 1;
            continue;
        }
        if in_backtick {
            if b == b'`' {
                in_backtick = false;
            }
            i += 1;
            continue;
        }
        if in_bracket {
            if b == b']' {
                in_bracket = false;
            }
            i += 1;
            continue;
        }

        match b {
            b'\'' => in_single = true,
            b'"' => in_double = true,
            b'`' => in_backtick = true,
            b'[' => in_bracket = true,
            b'(' => depth += 1,
            b')' => {
                depth = depth.saturating_sub(1);
            }
            b',' if depth == 0 => {
                parts.push(input[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    if start < input.len() {
        parts.push(input[start..].trim());
    }
    parts
}

fn contains_token(input: &str, token: &str) -> bool {
    contains_token_sequence(input, &[token])
}

fn contains_token_sequence(input: &str, seq: &[&str]) -> bool {
    if seq.is_empty() {
        return true;
    }
    let tokens = tokens_upper(input);
    if tokens.len() < seq.len() {
        return false;
    }
    for i in 0..=tokens.len() - seq.len() {
        if seq.iter().enumerate().all(|(j, s)| tokens[i + j] == *s) {
            return true;
        }
    }
    false
}

fn tokens_upper(input: &str) -> Vec<String> {
    let bytes = input.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if is_ident_start(bytes[i]) {
            let start = i;
            i += 1;
            while i < bytes.len() && is_ident_char(bytes[i]) {
                i += 1;
            }
            tokens.push(input[start..i].to_ascii_uppercase());
            continue;
        }
        i += 1;
    }
    tokens
}

fn extract_parenthesized(sql: &str) -> Option<&str> {
    let bytes = sql.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'(' {
            return extract_parenthesized_at(sql, i);
        }
        i += 1;
    }
    None
}

fn extract_parenthesized_at(sql: &str, start: usize) -> Option<&str> {
    let bytes = sql.as_bytes();
    if start >= bytes.len() || bytes[start] != b'(' {
        return None;
    }
    let mut depth = 0u32;
    let mut i = start;
    let mut in_single = false;
    let mut in_double = false;
    let mut in_backtick = false;
    let mut in_bracket = false;

    while i < bytes.len() {
        let b = bytes[i];
        if in_single {
            if b == b'\'' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                    i += 2;
                    continue;
                }
                in_single = false;
            }
            i += 1;
            continue;
        }
        if in_double {
            if b == b'"' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                    i += 2;
                    continue;
                }
                in_double = false;
            }
            i += 1;
            continue;
        }
        if in_backtick {
            if b == b'`' {
                in_backtick = false;
            }
            i += 1;
            continue;
        }
        if in_bracket {
            if b == b']' {
                in_bracket = false;
            }
            i += 1;
            continue;
        }

        match b {
            b'\'' => in_single = true,
            b'"' => in_double = true,
            b'`' => in_backtick = true,
            b'[' => in_bracket = true,
            b'(' => {
                depth += 1;
                if depth == 1 {
                    i += 1;
                    continue;
                }
            }
            b')' => {
                if depth == 1 {
                    return Some(&sql[start + 1..i]);
                }
                depth = depth.saturating_sub(1);
            }
            _ => {}
        }
        i += 1;
    }
    None
}

fn find_on_paren(sql: &str) -> Option<usize> {
    let bytes = sql.as_bytes();
    let mut i = 0usize;
    let mut in_single = false;
    let mut in_double = false;
    let mut in_backtick = false;
    let mut in_bracket = false;
    let mut seen_on = false;

    while i < bytes.len() {
        let b = bytes[i];
        if in_single {
            if b == b'\'' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                    i += 2;
                    continue;
                }
                in_single = false;
            }
            i += 1;
            continue;
        }
        if in_double {
            if b == b'"' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                    i += 2;
                    continue;
                }
                in_double = false;
            }
            i += 1;
            continue;
        }
        if in_backtick {
            if b == b'`' {
                in_backtick = false;
            }
            i += 1;
            continue;
        }
        if in_bracket {
            if b == b']' {
                in_bracket = false;
            }
            i += 1;
            continue;
        }

        match b {
            b'\'' => in_single = true,
            b'"' => in_double = true,
            b'`' => in_backtick = true,
            b'[' => in_bracket = true,
            b'(' if seen_on => return Some(i),
            _ => {}
        }

        if !seen_on && is_ident_start(b) {
            let start = i;
            i += 1;
            while i < bytes.len() && is_ident_char(bytes[i]) {
                i += 1;
            }
            if sql[start..i].eq_ignore_ascii_case("ON") {
                let prev = start.checked_sub(1).and_then(|p| bytes.get(p).copied());
                let next = bytes.get(i).copied();
                if prev.is_none_or(|c| !is_ident_char(c)) && next.is_none_or(|c| !is_ident_char(c))
                {
                    seen_on = true;
                }
            }
            continue;
        }

        i += 1;
    }
    None
}

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn resolve_right_null_len(
    right_scan: &mut CompiledScan<'_>,
    right_meta: &SideMeta,
    right_decoded: &mut Vec<ValueRefRaw>,
    right_overflow: &mut Vec<u8>,
) -> Result<usize> {
    if let Some(map) = right_meta.output_map.as_ref() {
        return Ok(map.len());
    }
    if let Some(map) = right_meta.proj_map.as_ref() {
        return Ok(map.len());
    }
    if let Some(count) = right_scan.column_count_hint() {
        return Ok(count);
    }

    let pager = right_scan.pager();
    let root = right_scan.root();
    let mut stack = Vec::new();
    let count = table::scan_table_cells_with_scratch_and_stack_until(
        pager,
        root,
        right_overflow,
        &mut stack,
        |_, payload| {
            let count = table::decode_record_project_into(payload, None, right_decoded)?;
            Ok(Some(count))
        },
    )?;
    right_decoded.clear();
    count.ok_or_else(|| JoinError::LeftJoinMissingRightColumns.into())
}

fn null_right_row<'row>(nulls: &'row mut Vec<ValueRefRaw>, len: usize) -> Row<'row> {
    if nulls.len() != len {
        nulls.resize(len, ValueRefRaw::Null);
    }
    Row::from_raw(nulls.as_slice(), None)
}

fn emit_left_only<'row, F>(
    left_rowid: i64,
    left: Row<'row>,
    right_nulls: &'row mut Vec<ValueRefRaw>,
    right_null_len: Option<usize>,
    cb: &mut F,
) -> Result<()>
where
    F: for<'r> FnMut(JoinedRow<'r>) -> Result<()>,
{
    if let Some(len) = right_null_len {
        let right_out = null_right_row(right_nulls, len);
        cb(JoinedRow { left_rowid, right_rowid: NULL_ROWID, left, right: right_out })?;
    }
    Ok(())
}

struct MemTracker {
    limit: Option<usize>,
    used: usize,
}

impl MemTracker {
    fn new(limit: Option<usize>) -> Self {
        Self { limit, used: 0 }
    }

    fn charge(&mut self, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let new_total = self.used.saturating_add(bytes);
        if let Some(limit) = self.limit
            && new_total > limit
        {
            return Err(JoinError::HashMemoryLimitExceeded.into());
        }
        self.used = new_total;
        Ok(())
    }

    fn charge_capacity_growth(&mut self, old_cap: usize, new_cap: usize) -> Result<()> {
        if new_cap > old_cap {
            let bytes = (new_cap - old_cap) * std::mem::size_of::<i64>();
            self.charge(bytes)?;
        }
        Ok(())
    }
}
