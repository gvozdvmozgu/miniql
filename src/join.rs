use std::borrow::Borrow;
use std::collections::hash_map::Entry;
use std::hash::{Hash, Hasher};
use std::ptr::NonNull;

use bumpalo::Bump;
use hashbrown::HashMap as HbHashMap;
use rustc_hash::{FxBuildHasher, FxHashMap};

use crate::index::{self, IndexCursor, IndexScratch};
use crate::pager::{PageId, Pager};
use crate::query::{OrderDir, PreparedScan, Row, Scan, ScanScratch};
use crate::schema::{TableSchema, parse_index_columns, parse_table_schema};
use crate::table::{self, BytesSpan, RawBytes, ValueRef, ValueSlot};

/// Result type for join operations.
pub type Result<T> = table::Result<T>;

/// Join-specific errors.
#[non_exhaustive]
#[derive(Debug)]
pub enum Error {
    UnsupportedJoinKeyType,
    IndexKeyNotComparable,
    MissingIndexRowId,
    HashMemoryLimitExceeded,
    MissingJoinCondition,
    InvalidOrderByColumn,
    UnsupportedJoinType,
    UnsupportedJoinStrategy,
    LeftJoinMissingRightColumns,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedJoinKeyType => f.write_str("Unsupported join key type"),
            Self::IndexKeyNotComparable => f.write_str("Index key is not comparable to join key"),
            Self::MissingIndexRowId => f.write_str("Index record does not end with a rowid"),
            Self::HashMemoryLimitExceeded => f.write_str("Hash join memory limit exceeded"),
            Self::MissingJoinCondition => f.write_str("Join condition is missing"),
            Self::InvalidOrderByColumn => f.write_str("ORDER BY column is not available"),
            Self::UnsupportedJoinType => f.write_str("Join type is not supported"),
            Self::UnsupportedJoinStrategy => f.write_str("Join strategy is not supported"),
            Self::LeftJoinMissingRightColumns => {
                f.write_str("Left join could not determine right-side column count")
            }
        }
    }
}

impl std::error::Error for Error {}

/// Alias for join errors.
pub type JoinError = Error;

/// Supported join types.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub enum JoinType {
    Inner,
    Left,
}

/// Join key selector for left/right sides.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub enum JoinKey {
    Col(u16),
    RowId,
}

/// Join side selector for `ORDER BY`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinSide {
    Left,
    Right,
}

/// Column + side + direction for join ordering.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JoinOrderBy {
    pub side: JoinSide,
    pub col: u16,
    pub dir: OrderDir,
}

impl JoinOrderBy {
    /// Order by a left-side column.
    pub fn left(col: u16, dir: OrderDir) -> Self {
        Self { side: JoinSide::Left, col, dir }
    }

    /// Order by a right-side column.
    pub fn right(col: u16, dir: OrderDir) -> Self {
        Self { side: JoinSide::Right, col, dir }
    }

    /// Order by a left-side column ascending.
    pub fn left_asc(col: u16) -> Self {
        Self::left(col, OrderDir::Asc)
    }

    /// Order by a left-side column descending.
    pub fn left_desc(col: u16) -> Self {
        Self::left(col, OrderDir::Desc)
    }

    /// Order by a right-side column ascending.
    pub fn right_asc(col: u16) -> Self {
        Self::right(col, OrderDir::Asc)
    }

    /// Order by a right-side column descending.
    pub fn right_desc(col: u16) -> Self {
        Self::right(col, OrderDir::Desc)
    }
}

/// Shorthand for `JoinOrderBy::left_asc`.
///
/// ```rust
/// use std::path::Path;
///
/// use miniql::{Db, Join, JoinKey, JoinScratch, left_asc, right_desc};
///
/// let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/join.db");
/// let db = Db::open(path).unwrap();
/// let left = db.table("users").unwrap();
/// let right = db.table("orders").unwrap();
/// let mut scratch = JoinScratch::with_capacity(4, 4, 0);
/// Join::inner(left.scan(), right.scan())
///     .on(JoinKey::RowId, JoinKey::Col(0))
///     .project_left([0])
///     .project_right([1])
///     .order_by([left_asc(0), right_desc(0)])
///     .for_each(&mut scratch, |_| Ok(()))
///     .unwrap();
/// ```
pub fn left_asc(col: u16) -> JoinOrderBy {
    JoinOrderBy::left_asc(col)
}

/// Shorthand for `JoinOrderBy::left_desc`.
pub fn left_desc(col: u16) -> JoinOrderBy {
    JoinOrderBy::left_desc(col)
}

/// Shorthand for `JoinOrderBy::right_asc`.
pub fn right_asc(col: u16) -> JoinOrderBy {
    JoinOrderBy::right_asc(col)
}

/// Shorthand for `JoinOrderBy::right_desc`.
pub fn right_desc(col: u16) -> JoinOrderBy {
    JoinOrderBy::right_desc(col)
}

/// Join strategy selector.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub enum JoinStrategy {
    Auto,
    Hash,
    IndexNestedLoop { index_root: PageId, index_key_col: u16 },
    IndexMerge { index_root: PageId, index_key_col: u16 },
    HashJoin,
    NestedLoopScan,
}

/// Join builder.
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
    order_by: Option<Vec<JoinOrderBy>>,
}

// For left joins without a match, right_rowid is set to 0 and right values are
// NULL.
const NULL_ROWID: i64 = 0;

/// Compiled join ready for execution.
pub struct PreparedJoin<'db> {
    left: PreparedScan<'db>,
    right: PreparedScan<'db>,
    left_meta: SideMeta,
    right_meta: SideMeta,
    plan: JoinPlan,
    hash_mem_limit: Option<usize>,
    right_null_len: Option<usize>,
    order_by: Option<Box<[ResolvedOrderBy]>>,
}

#[derive(Clone, Debug)]
enum JoinPlan {
    IndexNestedLoop { index_root: PageId, index_key_col: u16, index_cols: Option<Box<[u16]>> },
    IndexMerge { index_root: PageId, index_key_col: u16, index_cols: Option<Box<[u16]>> },
    HashJoin,
    NestedLoopScan,
    RowIdNestedLoop,
}

#[derive(Clone, Debug)]
struct IndexInfo {
    root: PageId,
    key_col: u16,
    cols: Option<Vec<u16>>,
    unique_by_key: bool,
}

impl<'db> Join<'db> {
    /// Create a new join builder.
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
            order_by: None,
        }
    }

    /// Create an inner join builder.
    pub fn inner(left: Scan<'db>, right: Scan<'db>) -> Self {
        Self::new(JoinType::Inner, left, right)
    }

    /// Create a left join builder.
    pub fn left(left: Scan<'db>, right: Scan<'db>) -> Self {
        Self::new(JoinType::Left, left, right)
    }

    /// Set the join key columns.
    pub fn on(mut self, left: JoinKey, right: JoinKey) -> Self {
        self.left_key = Some(left);
        self.right_key = Some(right);
        self
    }

    /// Set the join strategy.
    pub fn strategy(mut self, s: JoinStrategy) -> Self {
        self.strategy = s;
        self
    }

    /// Set the hash join memory limit in bytes.
    pub fn hash_mem_limit(mut self, bytes: usize) -> Self {
        self.hash_mem_limit = Some(bytes);
        self
    }

    /// Project columns from the left side.
    pub fn project_left<const N: usize>(mut self, cols: [u16; N]) -> Self {
        self.project_left = Some(cols.to_vec());
        self
    }

    /// Project columns from the right side.
    pub fn project_right<const N: usize>(mut self, cols: [u16; N]) -> Self {
        self.project_right = Some(cols.to_vec());
        self
    }

    /// Apply `ORDER BY` to join output.
    ///
    /// ```rust
    /// use std::path::Path;
    ///
    /// use miniql::{Db, Join, JoinKey, JoinScratch, left_asc, right_desc};
    ///
    /// let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/join.db");
    /// let db = Db::open(path).unwrap();
    /// let left = db.table("users").unwrap();
    /// let right = db.table("orders").unwrap();
    /// let mut scratch = JoinScratch::with_capacity(4, 4, 0);
    /// Join::inner(left.scan(), right.scan())
    ///     .on(JoinKey::RowId, JoinKey::Col(0))
    ///     .project_left([0])
    ///     .project_right([1])
    ///     .order_by([left_asc(0), right_desc(0)])
    ///     .for_each(&mut scratch, |_| Ok(()))
    ///     .unwrap();
    /// ```
    pub fn order_by<const N: usize>(mut self, cols: [JoinOrderBy; N]) -> Self {
        if N == 0 {
            self.order_by = None;
        } else {
            self.order_by = Some(cols.to_vec());
        }
        self
    }

    /// Compile the join into an executable plan.
    pub fn compile(self) -> Result<PreparedJoin<'db>> {
        let left_key = self.left_key.ok_or(Error::MissingJoinCondition)?;
        let right_key = self.right_key.ok_or(Error::MissingJoinCondition)?;

        let order_by = self.order_by.unwrap_or_default();
        let left_order_cols = collect_order_cols(&order_by, JoinSide::Left);
        let right_order_cols = collect_order_cols(&order_by, JoinSide::Right);
        let left_has_order = self.left.has_order_by();

        let ((left_scan, left_meta), (mut right_scan, right_meta)) = build_sides(
            self.left,
            self.right,
            left_key,
            right_key,
            self.project_left,
            self.project_right,
            &left_order_cols,
            &right_order_cols,
        )?;

        let plan = match self.strategy {
            JoinStrategy::IndexNestedLoop { index_root, index_key_col } => {
                if !matches!(right_key, JoinKey::Col(_)) {
                    return Err(Error::UnsupportedJoinKeyType.into());
                }
                let index_cols = discover_index_cols_for_root(
                    right_scan.pager(),
                    right_scan.root(),
                    index_root,
                )?;
                JoinPlan::IndexNestedLoop {
                    index_root,
                    index_key_col,
                    index_cols: index_cols.map(|cols| cols.into_boxed_slice()),
                }
            }
            JoinStrategy::IndexMerge { index_root, index_key_col } => {
                if !matches!(right_key, JoinKey::Col(_)) {
                    return Err(Error::UnsupportedJoinKeyType.into());
                }
                if !matches!(left_key, JoinKey::RowId) || left_has_order {
                    return Err(Error::UnsupportedJoinStrategy.into());
                }
                let index_cols = discover_index_cols_for_root(
                    right_scan.pager(),
                    right_scan.root(),
                    index_root,
                )?;
                JoinPlan::IndexMerge {
                    index_root,
                    index_key_col,
                    index_cols: index_cols.map(|cols| cols.into_boxed_slice()),
                }
            }
            JoinStrategy::Hash | JoinStrategy::HashJoin => JoinPlan::HashJoin,
            JoinStrategy::NestedLoopScan => JoinPlan::NestedLoopScan,
            JoinStrategy::Auto => {
                if matches!(right_key, JoinKey::RowId) {
                    JoinPlan::RowIdNestedLoop
                } else if let JoinKey::Col(col) = right_key {
                    let prefer_hash = should_prefer_hash_join(&left_scan, &right_scan);
                    if let Some(index) =
                        discover_index_for_join(right_scan.pager(), right_scan.root(), col)?
                    {
                        let index_cols = index.cols.map(|cols| cols.into_boxed_slice());
                        if index.unique_by_key {
                            JoinPlan::IndexNestedLoop {
                                index_root: index.root,
                                index_key_col: index.key_col,
                                index_cols,
                            }
                        } else if prefer_hash {
                            JoinPlan::HashJoin
                        } else if matches!(left_key, JoinKey::RowId) && !left_has_order {
                            JoinPlan::IndexMerge {
                                index_root: index.root,
                                index_key_col: index.key_col,
                                index_cols,
                            }
                        } else {
                            JoinPlan::IndexNestedLoop {
                                index_root: index.root,
                                index_key_col: index.key_col,
                                index_cols,
                            }
                        }
                    } else {
                        JoinPlan::HashJoin
                    }
                } else {
                    JoinPlan::HashJoin
                }
            }
        };

        let right_null_len = if matches!(self.join_type, JoinType::Left) {
            let mut right_values = Vec::new();
            let mut right_bytes = Vec::new();
            let mut right_serials = Vec::new();
            Some(resolve_right_null_len(
                &mut right_scan,
                &right_meta,
                &mut right_values,
                &mut right_bytes,
                &mut right_serials,
            )?)
        } else {
            None
        };

        let resolved_order_by = resolve_order_by(&order_by, &left_scan, &right_scan)?;

        Ok(PreparedJoin {
            left: left_scan,
            right: right_scan,
            left_meta,
            right_meta,
            plan,
            hash_mem_limit: self.hash_mem_limit,
            right_null_len,
            order_by: resolved_order_by,
        })
    }

    /// Execute the join and invoke `cb` for each joined row.
    pub fn for_each<F>(self, scratch: &mut JoinScratch, mut cb: F) -> Result<()>
    where
        F: for<'row> FnMut(JoinedRow<'row>) -> Result<()>,
    {
        let mut prepared = self.compile()?;
        prepared.for_each(scratch, &mut cb)
    }
}

impl<'db> PreparedJoin<'db> {
    /// Execute the prepared join and invoke `cb` for each joined row.
    pub fn for_each<F>(&mut self, scratch: &mut JoinScratch, mut cb: F) -> Result<()>
    where
        F: for<'row> FnMut(JoinedRow<'row>) -> Result<()>,
    {
        if let Some(order_by) = self.order_by.take() {
            if order_by.is_empty() {
                self.order_by = Some(order_by);
            } else {
                let result = self.for_each_ordered(scratch, order_by.as_ref(), &mut cb);
                self.order_by = Some(order_by);
                return result;
            }
        }

        self.for_each_plan(scratch, &mut cb)
    }

    fn for_each_plan<F>(&mut self, scratch: &mut JoinScratch, cb: &mut F) -> Result<()>
    where
        F: for<'row> FnMut(JoinedRow<'row>) -> Result<()>,
    {
        let (
            left_scan_scratch,
            right_scan_scratch,
            right_values,
            right_bytes,
            right_serials,
            index_scratch,
            hash_state,
            right_nulls,
        ) = scratch.split_mut();
        let right_null_len = self.right_null_len;

        match &self.plan {
            JoinPlan::IndexNestedLoop { index_root, index_key_col, index_cols } => {
                index_nested_loop(
                    &mut self.left,
                    &self.left_meta,
                    &mut self.right,
                    &self.right_meta,
                    *index_root,
                    *index_key_col,
                    index_cols.as_deref(),
                    left_scan_scratch,
                    right_values,
                    right_bytes,
                    right_serials,
                    index_scratch,
                    right_nulls,
                    right_null_len,
                    cb,
                )
            }
            JoinPlan::IndexMerge { index_root, index_key_col, index_cols } => index_merge_join(
                &mut self.left,
                &self.left_meta,
                &mut self.right,
                &self.right_meta,
                *index_root,
                *index_key_col,
                index_cols.as_deref(),
                left_scan_scratch,
                right_values,
                right_bytes,
                right_serials,
                index_scratch,
                right_nulls,
                right_null_len,
                cb,
            ),
            JoinPlan::HashJoin => hash_join(
                &mut self.left,
                &self.left_meta,
                &mut self.right,
                &self.right_meta,
                self.hash_mem_limit,
                left_scan_scratch,
                right_scan_scratch,
                right_values,
                right_bytes,
                right_serials,
                hash_state,
                right_nulls,
                right_null_len,
                cb,
            ),
            JoinPlan::NestedLoopScan => nested_loop_scan(
                &mut self.left,
                &self.left_meta,
                &mut self.right,
                &self.right_meta,
                left_scan_scratch,
                right_scan_scratch,
                right_nulls,
                right_null_len,
                cb,
            ),
            JoinPlan::RowIdNestedLoop => rowid_nested_loop(
                &mut self.left,
                &self.left_meta,
                &mut self.right,
                &self.right_meta,
                left_scan_scratch,
                right_values,
                right_bytes,
                right_serials,
                right_nulls,
                right_null_len,
                cb,
            ),
        }
    }

    fn for_each_ordered<F>(
        &mut self,
        scratch: &mut JoinScratch,
        order_by: &[ResolvedOrderBy],
        cb: &mut F,
    ) -> Result<()>
    where
        F: for<'row> FnMut(JoinedRow<'row>) -> Result<()>,
    {
        let mut entries = Vec::new();
        let mut seq = 0u64;

        self.for_each_plan(scratch, &mut |jr| {
            let left_values: Vec<OwnedValue> =
                jr.left.values_raw().iter().copied().map(owned_value_from_slot).collect();
            let right_values: Vec<OwnedValue> =
                jr.right.values_raw().iter().copied().map(owned_value_from_slot).collect();

            for order in order_by {
                let len = match order.side {
                    JoinSide::Left => left_values.len(),
                    JoinSide::Right => right_values.len(),
                };
                if order.idx >= len {
                    return Err(Error::InvalidOrderByColumn.into());
                }
            }

            entries.push(JoinSortEntry {
                left_rowid: jr.left_rowid,
                right_rowid: jr.right_rowid,
                left_values,
                right_values,
                seq,
            });
            seq = seq.wrapping_add(1);
            Ok(())
        })?;

        entries.sort_by(|left, right| compare_join_entries(left, right, order_by));

        let mut left_slots = Vec::new();
        let mut right_slots = Vec::new();

        for entry in entries {
            build_slots_from_owned(&entry.left_values, &mut left_slots);
            build_slots_from_owned(&entry.right_values, &mut right_slots);
            let left_row = self.left_meta.output_row(&left_slots);
            let right_row = self.right_meta.output_row(&right_slots);
            cb(JoinedRow {
                left_rowid: entry.left_rowid,
                right_rowid: entry.right_rowid,
                left: left_row,
                right: right_row,
            })?;
        }

        Ok(())
    }
}

/// Joined row containing left and right projections.
pub struct JoinedRow<'row> {
    pub left_rowid: i64,
    pub right_rowid: i64,
    pub left: Row<'row>,
    pub right: Row<'row>,
}

/// Scratch buffers for join execution.
#[derive(Debug)]
pub struct JoinScratch {
    left_scan: ScanScratch,
    right_scan: ScanScratch,
    right_values: Vec<ValueSlot>,
    right_bytes: Vec<u8>,
    right_serials: Vec<u64>,
    index: IndexScratch,
    hash: HashState,
    right_nulls: Vec<ValueSlot>,
}

type JoinScratchParts<'a> = (
    &'a mut ScanScratch,
    &'a mut ScanScratch,
    &'a mut Vec<ValueSlot>,
    &'a mut Vec<u8>,
    &'a mut Vec<u64>,
    &'a mut IndexScratch,
    &'a mut HashState,
    &'a mut Vec<ValueSlot>,
);

impl JoinScratch {
    /// Create an empty join scratch buffer.
    pub fn new() -> Self {
        Self {
            left_scan: ScanScratch::new(),
            right_scan: ScanScratch::new(),
            right_values: Vec::new(),
            right_bytes: Vec::new(),
            right_serials: Vec::new(),
            index: IndexScratch::new(),
            hash: HashState::new(),
            right_nulls: Vec::new(),
        }
    }

    /// Create a join scratch buffer with capacity hints.
    pub fn with_capacity(left_values: usize, right_values: usize, overflow: usize) -> Self {
        Self {
            left_scan: ScanScratch::with_capacity(left_values, overflow),
            right_scan: ScanScratch::with_capacity(right_values, overflow),
            right_values: Vec::with_capacity(right_values),
            right_bytes: Vec::with_capacity(overflow),
            right_serials: Vec::with_capacity(right_values),
            index: IndexScratch::with_capacity(right_values, overflow),
            hash: HashState::with_capacity(right_values, overflow),
            right_nulls: Vec::with_capacity(right_values),
        }
    }

    fn split_mut(&mut self) -> JoinScratchParts<'_> {
        (
            &mut self.left_scan,
            &mut self.right_scan,
            &mut self.right_values,
            &mut self.right_bytes,
            &mut self.right_serials,
            &mut self.index,
            &mut self.hash,
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
    fn output_row<'row>(&'row self, values: &'row [ValueSlot]) -> Row<'row> {
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
                let idx = self.join_key_index.ok_or(Error::UnsupportedJoinKeyType)?;
                match row.get(idx) {
                    Some(ValueRef::Null) | None => Ok(None),
                    Some(value) => Ok(Some(value)),
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ResolvedOrderBy {
    side: JoinSide,
    idx: usize,
    dir: OrderDir,
}

#[derive(Clone, Debug)]
enum OwnedValue {
    Null,
    Integer(i64),
    Real(f64),
    Text(Vec<u8>),
    Blob(Vec<u8>),
}

#[derive(Clone, Debug)]
struct JoinSortEntry {
    left_rowid: i64,
    right_rowid: i64,
    left_values: Vec<OwnedValue>,
    right_values: Vec<OwnedValue>,
    seq: u64,
}

#[allow(clippy::too_many_arguments)]
fn build_sides<'db>(
    left: Scan<'db>,
    right: Scan<'db>,
    left_key: JoinKey,
    right_key: JoinKey,
    project_left: Option<Vec<u16>>,
    project_right: Option<Vec<u16>>,
    left_order_cols: &[u16],
    right_order_cols: &[u16],
) -> Result<((PreparedScan<'db>, SideMeta), (PreparedScan<'db>, SideMeta))> {
    let left_state = build_side(left, left_key, project_left, left_order_cols)?;
    let right_state = build_side(right, right_key, project_right, right_order_cols)?;
    Ok((left_state, right_state))
}

fn build_side<'db>(
    scan: Scan<'db>,
    join_key: JoinKey,
    output_override: Option<Vec<u16>>,
    order_cols: &[u16],
) -> Result<(PreparedScan<'db>, SideMeta)> {
    let output_cols = output_override.or_else(|| scan.projection().map(|cols| cols.to_vec()));
    let mut scan_proj = output_cols.clone();
    if let (Some(cols), JoinKey::Col(col)) = (scan_proj.as_mut(), join_key)
        && !cols.contains(&col)
    {
        cols.push(col);
    }
    if let Some(cols) = scan_proj.as_mut() {
        for col in order_cols {
            if !cols.contains(col) {
                cols.push(*col);
            }
        }
    }

    let join_key_index = match join_key {
        JoinKey::RowId => None,
        JoinKey::Col(col) => match scan_proj.as_ref() {
            Some(cols) => Some(
                cols.iter().position(|value| *value == col).ok_or(Error::UnsupportedJoinKeyType)?,
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

fn collect_order_cols(order_by: &[JoinOrderBy], side: JoinSide) -> Vec<u16> {
    let mut cols = Vec::new();
    for order in order_by {
        if order.side != side {
            continue;
        }
        if !cols.contains(&order.col) {
            cols.push(order.col);
        }
    }
    cols
}

fn resolve_order_by(
    order_by: &[JoinOrderBy],
    left_scan: &PreparedScan<'_>,
    right_scan: &PreparedScan<'_>,
) -> Result<Option<Box<[ResolvedOrderBy]>>> {
    if order_by.is_empty() {
        return Ok(None);
    }

    let mut resolved = Vec::with_capacity(order_by.len());
    for order in order_by {
        let (scan, side) = match order.side {
            JoinSide::Left => (left_scan, JoinSide::Left),
            JoinSide::Right => (right_scan, JoinSide::Right),
        };
        if let Some(count) = scan.column_count_hint()
            && order.col as usize >= count
        {
            return Err(Error::InvalidOrderByColumn.into());
        }
        let idx = match scan.needed_cols() {
            Some(cols) => {
                cols.binary_search(&order.col).map_err(|_| Error::InvalidOrderByColumn)?
            }
            None => order.col as usize,
        };
        resolved.push(ResolvedOrderBy { side, idx, dir: order.dir });
    }

    Ok(Some(resolved.into_boxed_slice()))
}

fn owned_value_from_slot(slot: ValueSlot) -> OwnedValue {
    match unsafe { slot.as_value_ref() } {
        ValueRef::Null => OwnedValue::Null,
        ValueRef::Integer(value) => OwnedValue::Integer(value),
        ValueRef::Real(value) => OwnedValue::Real(value),
        ValueRef::Text(bytes) => OwnedValue::Text(bytes.to_vec()),
        ValueRef::Blob(bytes) => OwnedValue::Blob(bytes.to_vec()),
    }
}

fn build_slots_from_owned(values: &[OwnedValue], out: &mut Vec<ValueSlot>) {
    out.clear();
    out.reserve(values.len());
    for value in values {
        let slot = match value {
            OwnedValue::Null => ValueSlot::Null,
            OwnedValue::Integer(value) => ValueSlot::Integer(*value),
            OwnedValue::Real(value) => ValueSlot::Real(*value),
            OwnedValue::Text(bytes) => {
                ValueSlot::Text(BytesSpan::Scratch(RawBytes::from_slice(bytes)))
            }
            OwnedValue::Blob(bytes) => {
                ValueSlot::Blob(BytesSpan::Scratch(RawBytes::from_slice(bytes)))
            }
        };
        out.push(slot);
    }
}

fn compare_join_entries(
    left: &JoinSortEntry,
    right: &JoinSortEntry,
    order_by: &[ResolvedOrderBy],
) -> std::cmp::Ordering {
    for order in order_by {
        let (left_value, right_value) = match order.side {
            JoinSide::Left => (&left.left_values[order.idx], &right.left_values[order.idx]),
            JoinSide::Right => (&left.right_values[order.idx], &right.right_values[order.idx]),
        };
        let mut cmp = compare_owned_values(left_value, right_value);
        if matches!(order.dir, OrderDir::Desc) {
            cmp = cmp.reverse();
        }
        if cmp != std::cmp::Ordering::Equal {
            return cmp;
        }
    }
    left.seq.cmp(&right.seq)
}

fn compare_owned_values(left: &OwnedValue, right: &OwnedValue) -> std::cmp::Ordering {
    let rank = |value: &OwnedValue| match value {
        OwnedValue::Null => 0u8,
        OwnedValue::Integer(_) | OwnedValue::Real(_) => 1u8,
        OwnedValue::Text(_) => 2u8,
        OwnedValue::Blob(_) => 3u8,
    };

    let left_rank = rank(left);
    let right_rank = rank(right);
    if left_rank != right_rank {
        return left_rank.cmp(&right_rank);
    }

    match (left, right) {
        (OwnedValue::Null, OwnedValue::Null) => std::cmp::Ordering::Equal,
        (OwnedValue::Integer(l), OwnedValue::Integer(r)) => l.cmp(r),
        (OwnedValue::Integer(l), OwnedValue::Real(r)) => cmp_f64_total(*l as f64, *r),
        (OwnedValue::Real(l), OwnedValue::Integer(r)) => cmp_f64_total(*l, *r as f64),
        (OwnedValue::Real(l), OwnedValue::Real(r)) => cmp_f64_total(*l, *r),
        (OwnedValue::Text(l), OwnedValue::Text(r)) => l.cmp(r),
        (OwnedValue::Blob(l), OwnedValue::Blob(r)) => l.cmp(r),
        _ => std::cmp::Ordering::Equal,
    }
}

fn cmp_f64_total(left: f64, right: f64) -> std::cmp::Ordering {
    match (left.is_nan(), right.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => left.partial_cmp(&right).unwrap_or(std::cmp::Ordering::Equal),
    }
}

fn covering_map_for_index(
    right_scan: &PreparedScan<'_>,
    index_cols: Option<&[u16]>,
) -> Option<Vec<(u16, usize)>> {
    let needed = right_scan.needed_cols()?;
    let index_cols = index_cols?;
    let mut map = Vec::with_capacity(needed.len());
    for (out_idx, col) in needed.iter().enumerate() {
        let pos = index_cols.iter().position(|idx| idx == col)?;
        map.push((pos as u16, out_idx));
    }
    map.sort_by_key(|(pos, _)| *pos);
    Some(map)
}

#[allow(clippy::too_many_arguments)]
fn index_nested_loop<F>(
    left_scan: &mut PreparedScan<'_>,
    left_meta: &SideMeta,
    right_scan: &mut PreparedScan<'_>,
    right_meta: &SideMeta,
    index_root: PageId,
    index_key_col: u16,
    index_cols: Option<&[u16]>,
    left_scratch: &mut ScanScratch,
    right_values: &mut Vec<ValueSlot>,
    right_bytes: &mut Vec<u8>,
    right_serials: &mut Vec<u64>,
    index_scratch: &mut IndexScratch,
    right_nulls: &mut Vec<ValueSlot>,
    right_null_len: Option<usize>,
    cb: &mut F,
) -> Result<()>
where
    F: for<'row> FnMut(JoinedRow<'row>) -> Result<()>,
{
    let right_root = right_scan.root();
    let pager = right_scan.pager();
    let mut cursor = IndexCursor::new(pager, index_root, index_key_col, index_scratch);
    let covering_map = covering_map_for_index(right_scan, index_cols);

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
            if let Some(map) = covering_map.as_ref() {
                let emitted = cursor.with_current_payload_and_rowid(|payload, right_rowid| {
                    if let Some(right_row) = right_scan.eval_index_payload_with_map(
                        payload,
                        map,
                        right_values,
                        right_bytes,
                        right_serials,
                        true,
                    )? {
                        let right_out = right_meta.output_row(right_row.values_raw());
                        cb(JoinedRow {
                            left_rowid,
                            right_rowid,
                            left: left_out,
                            right: right_out,
                        })?;
                        return Ok(true);
                    }
                    Ok(false)
                })?;
                if emitted {
                    matched = true;
                }
            } else {
                let right_rowid = cursor.current_rowid()?;
                if let Some(cell) = table::lookup_rowid_cell(pager, right_root, right_rowid)?
                    && let Some(right_row) = right_scan.eval_payload(
                        cell.payload(),
                        right_values,
                        right_bytes,
                        right_serials,
                    )?
                {
                    let right_out = right_meta.output_row(right_row.values_raw());
                    cb(JoinedRow { left_rowid, right_rowid, left: left_out, right: right_out })?;
                    matched = true;
                }
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
fn index_merge_join<F>(
    left_scan: &mut PreparedScan<'_>,
    left_meta: &SideMeta,
    right_scan: &mut PreparedScan<'_>,
    right_meta: &SideMeta,
    index_root: PageId,
    index_key_col: u16,
    index_cols: Option<&[u16]>,
    left_scratch: &mut ScanScratch,
    right_values: &mut Vec<ValueSlot>,
    right_bytes: &mut Vec<u8>,
    right_serials: &mut Vec<u64>,
    index_scratch: &mut IndexScratch,
    right_nulls: &mut Vec<ValueSlot>,
    right_null_len: Option<usize>,
    cb: &mut F,
) -> Result<()>
where
    F: for<'row> FnMut(JoinedRow<'row>) -> Result<()>,
{
    let right_root = right_scan.root();
    let pager = right_scan.pager();
    let mut cursor = IndexCursor::new(pager, index_root, index_key_col, index_scratch);
    let covering_map = covering_map_for_index(right_scan, index_cols);
    let mut right_exhausted = false;

    left_scan.for_each(left_scratch, |left_rowid, left_row| {
        let left_out = left_meta.output_row(left_row.values_raw());
        let Some(left_key_value) = left_meta.join_key(left_rowid, &left_row)? else {
            return emit_left_only(left_rowid, left_out, right_nulls, right_null_len, cb);
        };

        if right_exhausted {
            return emit_left_only(left_rowid, left_out, right_nulls, right_null_len, cb);
        }

        if !cursor.advance_to_ge(left_key_value)? {
            right_exhausted = true;
            return emit_left_only(left_rowid, left_out, right_nulls, right_null_len, cb);
        }

        if !cursor.key_eq(left_key_value)? {
            return emit_left_only(left_rowid, left_out, right_nulls, right_null_len, cb);
        }

        let mut matched = false;
        loop {
            if let Some(map) = covering_map.as_ref() {
                let emitted = cursor.with_current_payload_and_rowid(|payload, right_rowid| {
                    if let Some(right_row) = right_scan.eval_index_payload_with_map(
                        payload,
                        map,
                        right_values,
                        right_bytes,
                        right_serials,
                        true,
                    )? {
                        let right_out = right_meta.output_row(right_row.values_raw());
                        cb(JoinedRow {
                            left_rowid,
                            right_rowid,
                            left: left_out,
                            right: right_out,
                        })?;
                        return Ok(true);
                    }
                    Ok(false)
                })?;
                if emitted {
                    matched = true;
                }
            } else {
                let right_rowid = cursor.current_rowid()?;
                if let Some(cell) = table::lookup_rowid_cell(pager, right_root, right_rowid)?
                    && let Some(right_row) = right_scan.eval_payload(
                        cell.payload(),
                        right_values,
                        right_bytes,
                        right_serials,
                    )?
                {
                    let right_out = right_meta.output_row(right_row.values_raw());
                    cb(JoinedRow { left_rowid, right_rowid, left: left_out, right: right_out })?;
                    matched = true;
                }
            }

            if !cursor.next()? {
                right_exhausted = true;
                break;
            }
            if !cursor.key_eq(left_key_value)? {
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
    left_scan: &mut PreparedScan<'_>,
    left_meta: &SideMeta,
    right_scan: &mut PreparedScan<'_>,
    right_meta: &SideMeta,
    left_scratch: &mut ScanScratch,
    right_values: &mut Vec<ValueSlot>,
    right_bytes: &mut Vec<u8>,
    right_serials: &mut Vec<u64>,
    right_nulls: &mut Vec<ValueSlot>,
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
        if let Some(cell) = table::lookup_rowid_cell(pager, right_root, target_rowid)?
            && let Some(right_row) =
                right_scan.eval_payload(cell.payload(), right_values, right_bytes, right_serials)?
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

#[derive(Debug, Clone, Copy)]
enum NumericKey {
    Integer(i64),
    Real(f64),
}

#[derive(Debug, Clone)]
struct RightRow {
    rowid: i64,
    page_id: PageId,
    cell_offset: u16,
    numeric_key: Option<NumericKey>,
}

#[derive(Debug, Clone)]
enum RightRowList {
    One(RightRow),
    Many(Vec<RightRow>),
}

impl RightRowList {
    #[inline(always)]
    fn new(row: RightRow) -> Self {
        RightRowList::One(row)
    }

    #[inline(always)]
    fn push(&mut self, row: RightRow) -> (usize, usize) {
        match self {
            RightRowList::One(_) => {
                let first = match std::mem::replace(self, RightRowList::Many(Vec::new())) {
                    RightRowList::One(first) => first,
                    _ => unreachable!("RightRowList variant swap failed"),
                };
                let vec = vec![first, row];
                let new = vec.capacity();
                *self = RightRowList::Many(vec);
                (0, new)
            }
            RightRowList::Many(vec) => {
                let old = vec.capacity();
                vec.push(row);
                (old, vec.capacity())
            }
        }
    }
}

// BytesKey points into HashState::arena. Keys are only valid until
// HashState::clear() which clears the maps before resetting the arena.
#[derive(Clone, Copy)]
struct BytesKey {
    ptr: NonNull<u8>,
    len: u32,
}

impl BytesKey {
    fn from_slice_in(arena: &Bump, bytes: &[u8]) -> Self {
        if bytes.is_empty() {
            return Self { ptr: NonNull::dangling(), len: 0 };
        }
        debug_assert!(bytes.len() <= u32::MAX as usize);
        let stored = arena.alloc_slice_copy(bytes);
        // SAFETY: bump allocation returns a valid, non-null pointer for the slice.
        let ptr = unsafe { NonNull::new_unchecked(stored.as_mut_ptr()) };
        let len = bytes.len() as u32;
        Self { ptr, len }
    }

    fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: pointer comes from the arena and is valid until HashState::clear.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len as usize) }
    }
}

impl Borrow<[u8]> for BytesKey {
    fn borrow(&self) -> &[u8] {
        self.as_slice()
    }
}

impl PartialEq for BytesKey {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for BytesKey {}

impl Hash for BytesKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

struct HashState {
    numeric: FxHashMap<u64, RightRowList>,
    text: HbHashMap<BytesKey, RightRowList, FxBuildHasher>,
    blob: HbHashMap<BytesKey, RightRowList, FxBuildHasher>,
    arena: Bump,
}

impl HashState {
    fn new() -> Self {
        Self {
            numeric: FxHashMap::default(),
            text: HbHashMap::with_hasher(FxBuildHasher),
            blob: HbHashMap::with_hasher(FxBuildHasher),
            arena: Bump::new(),
        }
    }

    fn with_capacity(keys: usize, bytes: usize) -> Self {
        let arena = if bytes > 0 { Bump::with_capacity(bytes) } else { Bump::new() };
        Self {
            numeric: FxHashMap::with_capacity_and_hasher(keys, FxBuildHasher),
            text: HbHashMap::with_capacity_and_hasher(keys, FxBuildHasher),
            blob: HbHashMap::with_capacity_and_hasher(keys, FxBuildHasher),
            arena,
        }
    }

    fn clear(&mut self) {
        self.numeric.clear();
        self.text.clear();
        self.blob.clear();
        self.arena.reset();
    }
}

impl std::fmt::Debug for HashState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HashState")
            .field("numeric_len", &self.numeric.len())
            .field("text_len", &self.text.len())
            .field("blob_len", &self.blob.len())
            .finish()
    }
}

#[allow(clippy::too_many_arguments)]
fn hash_join<F>(
    left_scan: &mut PreparedScan<'_>,
    left_meta: &SideMeta,
    right_scan: &mut PreparedScan<'_>,
    right_meta: &SideMeta,
    mem_limit: Option<usize>,
    left_scratch: &mut ScanScratch,
    right_scratch: &mut ScanScratch,
    right_values: &mut Vec<ValueSlot>,
    right_bytes: &mut Vec<u8>,
    right_serials: &mut Vec<u64>,
    hash_state: &mut HashState,
    right_nulls: &mut Vec<ValueSlot>,
    right_null_len: Option<usize>,
    cb: &mut F,
) -> Result<()>
where
    F: for<'row> FnMut(JoinedRow<'row>) -> Result<()>,
{
    let mut mem = MemTracker::new(mem_limit);
    hash_state.clear();
    let arena = &hash_state.arena;
    let numeric = &mut hash_state.numeric;
    let text = &mut hash_state.text;
    let blob = &mut hash_state.blob;

    let pager = right_scan.pager();
    let right_root = right_scan.root();
    let (values, bytes, serials, stack) = right_scratch.split_mut();
    let mut seen = 0usize;
    let limit = right_scan.limit();

    table::scan_table_cells_with_scratch_and_stack_until(pager, right_root, stack, |cell| {
        if let Some(limit) = limit
            && seen >= limit
        {
            return Ok(Some(()));
        }
        let rowid = cell.rowid();
        let payload = cell.payload();
        let payload_len = payload.len();
        let Some(row) =
            right_scan.eval_payload_with_filters(payload, values, bytes, serials, true)?
        else {
            return Ok(None::<()>);
        };
        seen += 1;
        let Some(key_value) = right_meta.join_key(rowid, &row)? else {
            return Ok(None::<()>);
        };
        let Some(key_ref) = hash_key_from_value(key_value)? else {
            return Ok(None::<()>);
        };
        let numeric_key = numeric_key_from_value(key_value);
        mem.charge(payload_len)?;

        let right_row = RightRow {
            rowid,
            page_id: cell.page_id(),
            cell_offset: cell.cell_offset(),
            numeric_key,
        };

        match key_ref {
            HashKeyRef::Number(bits) => match numeric.entry(bits) {
                Entry::Occupied(mut e) => {
                    let (old, new) = e.get_mut().push(right_row);
                    mem.charge_capacity_growth(old, new, std::mem::size_of::<RightRow>())?;
                }
                Entry::Vacant(e) => {
                    mem.charge(std::mem::size_of::<u64>())?;
                    e.insert(RightRowList::new(right_row));
                }
            },
            HashKeyRef::Text(bytes) => {
                if let Some(list) = text.get_mut(bytes) {
                    let (old, new) = list.push(right_row);
                    mem.charge_capacity_growth(old, new, std::mem::size_of::<RightRow>())?;
                } else {
                    mem.charge(bytes.len())?;
                    let key = BytesKey::from_slice_in(arena, bytes);
                    text.insert(key, RightRowList::new(right_row));
                }
            }
            HashKeyRef::Blob(bytes) => {
                if let Some(list) = blob.get_mut(bytes) {
                    let (old, new) = list.push(right_row);
                    mem.charge_capacity_growth(old, new, std::mem::size_of::<RightRow>())?;
                } else {
                    mem.charge(bytes.len())?;
                    let key = BytesKey::from_slice_in(arena, bytes);
                    blob.insert(key, RightRowList::new(right_row));
                }
            }
        }

        Ok(None::<()>)
    })?;

    left_scan.for_each(left_scratch, |left_rowid, left_row| {
        let left_out = left_meta.output_row(left_row.values_raw());
        let Some(left_key_value) = left_meta.join_key(left_rowid, &left_row)? else {
            return emit_left_only(left_rowid, left_out, right_nulls, right_null_len, cb);
        };
        let Some(key_ref) = hash_key_from_value(left_key_value)? else {
            return emit_left_only(left_rowid, left_out, right_nulls, right_null_len, cb);
        };
        let rows = match key_ref {
            HashKeyRef::Number(bits) => numeric.get(&bits),
            HashKeyRef::Text(bytes) => text.get(bytes),
            HashKeyRef::Blob(bytes) => blob.get(bytes),
        };

        if let Some(rows) = rows {
            let mut matched = false;
            match rows {
                RightRowList::One(row) => {
                    if let Some(numeric_key) = row.numeric_key
                        && !numeric_key_equals(left_key_value, numeric_key)
                    {
                        // Hash collision on numeric key: ignore.
                    } else {
                        let cell = table::read_table_cell_ref_from_bytes(
                            pager,
                            row.page_id,
                            row.cell_offset,
                        )?;
                        let payload = cell.payload();
                        if let Some(right_row) = right_scan.eval_payload_with_filters(
                            payload,
                            right_values,
                            right_bytes,
                            right_serials,
                            false,
                        )? {
                            let right_out = right_meta.output_row(right_row.values_raw());
                            cb(JoinedRow {
                                left_rowid,
                                right_rowid: row.rowid,
                                left: left_out,
                                right: right_out,
                            })?;
                            matched = true;
                        }
                    }
                }
                RightRowList::Many(rows) => {
                    for row in rows.iter() {
                        if let Some(numeric_key) = row.numeric_key
                            && !numeric_key_equals(left_key_value, numeric_key)
                        {
                            continue;
                        }
                        let cell = table::read_table_cell_ref_from_bytes(
                            pager,
                            row.page_id,
                            row.cell_offset,
                        )?;
                        let payload = cell.payload();
                        if let Some(right_row) = right_scan.eval_payload_with_filters(
                            payload,
                            right_values,
                            right_bytes,
                            right_serials,
                            false,
                        )? {
                            let right_out = right_meta.output_row(right_row.values_raw());
                            cb(JoinedRow {
                                left_rowid,
                                right_rowid: row.rowid,
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
    left_scan: &mut PreparedScan<'_>,
    left_meta: &SideMeta,
    right_scan: &mut PreparedScan<'_>,
    right_meta: &SideMeta,
    left_scratch: &mut ScanScratch,
    right_scratch: &mut ScanScratch,
    right_nulls: &mut Vec<ValueSlot>,
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

fn join_keys_equal(left: ValueRef<'_>, right: ValueRef<'_>) -> bool {
    match (left, right) {
        (ValueRef::Null, _) | (_, ValueRef::Null) => false,
        (ValueRef::Integer(l), ValueRef::Integer(r)) => l == r,
        (ValueRef::Integer(l), ValueRef::Real(r)) => (l as f64) == r,
        (ValueRef::Real(l), ValueRef::Integer(r)) => l == (r as f64),
        (ValueRef::Real(l), ValueRef::Real(r)) => l == r,
        (ValueRef::Text(l), ValueRef::Text(r)) => l == r,
        (ValueRef::Blob(l), ValueRef::Blob(r)) => l == r,
        _ => false,
    }
}

fn numeric_key_from_value(value: ValueRef<'_>) -> Option<NumericKey> {
    match value {
        ValueRef::Integer(value) => Some(NumericKey::Integer(value)),
        ValueRef::Real(value) => Some(NumericKey::Real(value)),
        _ => None,
    }
}

fn numeric_key_equals(left: ValueRef<'_>, right: NumericKey) -> bool {
    match right {
        NumericKey::Integer(value) => join_keys_equal(left, ValueRef::Integer(value)),
        NumericKey::Real(value) => join_keys_equal(left, ValueRef::Real(value)),
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
        ValueRef::Real(value) => {
            let value = if value == 0.0 { 0.0 } else { value };
            Some(HashKeyRef::Number(value.to_bits()))
        }
        ValueRef::Text(bytes) => Some(HashKeyRef::Text(bytes)),
        ValueRef::Blob(bytes) => Some(HashKeyRef::Blob(bytes)),
    })
}

fn should_prefer_hash_join(left_scan: &PreparedScan<'_>, right_scan: &PreparedScan<'_>) -> bool {
    let (Some(left_limit), Some(right_limit)) = (left_scan.limit(), right_scan.limit()) else {
        return false;
    };
    left_limit >= right_limit.saturating_mul(4)
}

fn index_cols_to_indices(schema: &TableSchema, index_cols: &[String]) -> Option<Vec<u16>> {
    let mut mapped = Vec::with_capacity(index_cols.len());
    for col in index_cols {
        let idx = schema.columns.iter().position(|name| name.eq_ignore_ascii_case(col))?;
        mapped.push(idx as u16);
    }
    Some(mapped)
}

fn row_text<'row>(row: table::RowView<'row>, idx: usize) -> Option<&'row str> {
    match row.get(idx)? {
        ValueRef::Text(bytes) => std::str::from_utf8(bytes).ok(),
        _ => None,
    }
}

fn row_integer(row: table::RowView<'_>, idx: usize) -> Option<i64> {
    match row.get(idx)? {
        ValueRef::Integer(value) => Some(value),
        _ => None,
    }
}

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn index_sql_is_unique(sql: &str) -> bool {
    let bytes = sql.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if is_ident_start(bytes[i]) {
            let start = i;
            i += 1;
            while i < bytes.len() && is_ident_char(bytes[i]) {
                i += 1;
            }
            if sql[start..i].eq_ignore_ascii_case("UNIQUE") {
                return true;
            }
            continue;
        }
        i += 1;
    }
    false
}

fn find_table_info(
    pager: &Pager,
    table_root: PageId,
    scratch: &mut table::RowScratch,
) -> Result<Option<(String, String)>> {
    let target_root = table_root.into_inner() as i64;
    table::scan_table_ref_until_with_scratch(pager, PageId::ROOT, scratch, |_, row| {
        if row.len() < 5 {
            return Ok(None);
        }
        let row_type = match row_text(row, 0) {
            Some(value) => value,
            None => return Ok(None),
        };
        if !row_type.eq_ignore_ascii_case("table") {
            return Ok(None);
        }
        let rootpage = match row_integer(row, 3) {
            Some(value) => value,
            None => return Ok(None),
        };
        if rootpage != target_root {
            return Ok(None);
        }
        let name = match row_text(row, 1) {
            Some(value) => value.to_owned(),
            None => return Ok(None),
        };
        let sql = match row_text(row, 4) {
            Some(value) => value.to_owned(),
            None => return Ok(None),
        };
        Ok(Some((name, sql)))
    })
}

fn discover_index_for_join(
    pager: &Pager,
    table_root: PageId,
    join_col: u16,
) -> Result<Option<IndexInfo>> {
    let mut scratch = table::RowScratch::with_capacity(8, table::DEFAULT_SPILL_CAPACITY);
    let Some((table_name, table_sql)) = find_table_info(pager, table_root, &mut scratch)? else {
        return Ok(None);
    };

    let schema = parse_table_schema(&table_sql);
    if schema.without_rowid {
        return Ok(None);
    }

    let join_col_name = match schema.columns.get(join_col as usize) {
        Some(name) => name.as_str(),
        None => {
            return Ok(None);
        }
    };

    let mut autoindexes = Vec::new();
    let mut best: Option<IndexInfo> = None;
    let found =
        table::scan_table_ref_until_with_scratch(pager, PageId::ROOT, &mut scratch, |_, row| {
            if row.len() < 5 {
                return Ok(None);
            }
            let row_type = match row_text(row, 0) {
                Some(value) => value,
                None => return Ok(None),
            };
            if !row_type.eq_ignore_ascii_case("index") {
                return Ok(None);
            }
            let tbl_name = match row_text(row, 2) {
                Some(value) => value,
                None => return Ok(None),
            };
            if !tbl_name.eq_ignore_ascii_case(&table_name) {
                return Ok(None);
            }
            let rootpage = match row_integer(row, 3).and_then(|value| u32::try_from(value).ok()) {
                Some(value) => value,
                None => return Ok(None),
            };
            match row.get(4) {
                Some(ValueRef::Text(bytes)) => {
                    let sql = match std::str::from_utf8(bytes) {
                        Ok(value) => value,
                        Err(_) => return Ok(None),
                    };
                    let Some(index_cols) = parse_index_columns(sql) else {
                        return Ok(None);
                    };
                    let unique_by_key = index_sql_is_unique(sql)
                        && index_cols.len() == 1
                        && index_cols[0].eq_ignore_ascii_case(join_col_name);
                    let Some(index_cols) = index_cols_to_indices(&schema, &index_cols) else {
                        return Ok(None);
                    };
                    if index_cols.first().copied() == Some(join_col) {
                        let info = IndexInfo {
                            root: PageId::new(rootpage),
                            key_col: 0,
                            cols: Some(index_cols),
                            unique_by_key,
                        };
                        if unique_by_key {
                            return Ok(Some(info));
                        }
                        if best.is_none() {
                            best = Some(info);
                        }
                    }
                }
                Some(ValueRef::Null) | None => {
                    autoindexes.push(PageId::new(rootpage));
                }
                _ => {}
            }
            Ok(None)
        })?;

    if let Some(index) = found {
        return Ok(Some(index));
    }

    if let Some(auto) = discover_autoindex_for_join(pager, &autoindexes, &schema, join_col_name)?
        && (auto.unique_by_key || best.is_none())
    {
        return Ok(Some(auto));
    }

    Ok(best)
}

fn discover_index_cols_for_root(
    pager: &Pager,
    table_root: PageId,
    index_root: PageId,
) -> Result<Option<Vec<u16>>> {
    let mut scratch = table::RowScratch::with_capacity(8, table::DEFAULT_SPILL_CAPACITY);
    let Some((table_name, table_sql)) = find_table_info(pager, table_root, &mut scratch)? else {
        return Ok(None);
    };

    let schema = parse_table_schema(&table_sql);
    if schema.without_rowid {
        return Ok(None);
    }

    enum IndexColsOutcome {
        Explicit(Vec<u16>),
        Autoindex,
        Unsupported,
    }

    let outcome =
        table::scan_table_ref_until_with_scratch(pager, PageId::ROOT, &mut scratch, |_, row| {
            if row.len() < 5 {
                return Ok(None);
            }
            let row_type = match row_text(row, 0) {
                Some(value) => value,
                None => return Ok(None),
            };
            if !row_type.eq_ignore_ascii_case("index") {
                return Ok(None);
            }
            let tbl_name = match row_text(row, 2) {
                Some(value) => value,
                None => return Ok(None),
            };
            if !tbl_name.eq_ignore_ascii_case(&table_name) {
                return Ok(None);
            }
            let rootpage = match row_integer(row, 3).and_then(|value| u32::try_from(value).ok()) {
                Some(value) => value,
                None => return Ok(None),
            };
            if PageId::new(rootpage) != index_root {
                return Ok(None);
            }
            match row.get(4) {
                Some(ValueRef::Text(bytes)) => {
                    let sql = match std::str::from_utf8(bytes) {
                        Ok(value) => value,
                        Err(_) => return Ok(Some(IndexColsOutcome::Unsupported)),
                    };
                    let Some(index_cols) = parse_index_columns(sql) else {
                        return Ok(Some(IndexColsOutcome::Unsupported));
                    };
                    let Some(mapped) = index_cols_to_indices(&schema, &index_cols) else {
                        return Ok(Some(IndexColsOutcome::Unsupported));
                    };
                    Ok(Some(IndexColsOutcome::Explicit(mapped)))
                }
                Some(ValueRef::Null) | None => Ok(Some(IndexColsOutcome::Autoindex)),
                _ => Ok(Some(IndexColsOutcome::Unsupported)),
            }
        })?;

    match outcome {
        Some(IndexColsOutcome::Explicit(cols)) => return Ok(Some(cols)),
        Some(IndexColsOutcome::Unsupported) => return Ok(None),
        Some(IndexColsOutcome::Autoindex) | None => {}
    }

    let mut index_scratch = IndexScratch::new();
    let Some(len) = index::index_key_len(pager, index_root, &mut index_scratch)? else {
        return Ok(None);
    };
    let mut candidates = Vec::new();
    for cols in &schema.unique_indexes {
        if cols.len() == len
            && let Some(mapped) = index_cols_to_indices(&schema, cols)
        {
            candidates.push(mapped);
        }
    }
    if candidates.len() == 1 {
        return Ok(Some(candidates.remove(0)));
    }

    Ok(None)
}

fn discover_autoindex_for_join(
    pager: &Pager,
    autoindexes: &[PageId],
    schema: &TableSchema,
    join_col_name: &str,
) -> Result<Option<IndexInfo>> {
    if autoindexes.is_empty() {
        return Ok(None);
    }

    let mut constraints = Vec::new();
    for cols in &schema.unique_indexes {
        let Some(pos) = cols.iter().position(|c| c.eq_ignore_ascii_case(join_col_name)) else {
            continue;
        };
        if pos == 0 {
            constraints.push((cols, pos));
        }
    }

    if constraints.is_empty() {
        return Ok(None);
    }

    if autoindexes.len() == 1 && constraints.len() == 1 {
        let Some(cols) = index_cols_to_indices(schema, constraints[0].0) else {
            return Ok(None);
        };
        let unique_by_key = constraints[0].0.len() == 1;
        return Ok(Some(IndexInfo {
            root: autoindexes[0],
            key_col: 0,
            cols: Some(cols),
            unique_by_key,
        }));
    }

    let mut index_scratch = IndexScratch::new();
    let mut autoindex_lens = Vec::new();
    for root in autoindexes {
        if let Some(len) = index::index_key_len(pager, *root, &mut index_scratch)? {
            autoindex_lens.push((*root, len));
        }
    }

    let mut matches = Vec::new();
    for (cols, pos) in constraints {
        let len = cols.len();
        let roots: Vec<PageId> =
            autoindex_lens.iter().filter(|(_, l)| *l == len).map(|(r, _)| *r).collect();
        if roots.len() == 1
            && let Some(mapped) = index_cols_to_indices(schema, cols)
        {
            matches.push(IndexInfo {
                root: roots[0],
                key_col: pos as u16,
                cols: Some(mapped),
                unique_by_key: cols.len() == 1,
            });
        }
    }

    matches.sort_by_key(|info| (info.root.into_inner(), info.key_col));
    matches.dedup_by_key(|info| (info.root.into_inner(), info.key_col));

    if matches.len() == 1 {
        return Ok(Some(matches[0].clone()));
    }

    Ok(None)
}

fn resolve_right_null_len(
    right_scan: &mut PreparedScan<'_>,
    right_meta: &SideMeta,
    right_values: &mut Vec<ValueSlot>,
    right_bytes: &mut Vec<u8>,
    right_serials: &mut Vec<u64>,
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
    let mut stack = Vec::with_capacity(64);
    let count =
        table::scan_table_cells_with_scratch_and_stack_until(pager, root, &mut stack, |cell| {
            let count = table::decode_record_project_into(
                cell.payload(),
                None,
                right_values,
                right_bytes,
                right_serials,
            )?;
            Ok(Some(count))
        })?;
    right_values.clear();
    count.ok_or_else(|| Error::LeftJoinMissingRightColumns.into())
}

fn null_right_row<'row>(nulls: &'row mut Vec<ValueSlot>, len: usize) -> Row<'row> {
    if nulls.len() != len {
        nulls.resize(len, ValueSlot::Null);
    }
    Row::from_raw(nulls.as_slice(), None)
}

fn emit_left_only<'row, F>(
    left_rowid: i64,
    left: Row<'row>,
    right_nulls: &'row mut Vec<ValueSlot>,
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
            return Err(Error::HashMemoryLimitExceeded.into());
        }
        self.used = new_total;
        Ok(())
    }

    fn charge_capacity_growth(
        &mut self,
        old_cap: usize,
        new_cap: usize,
        elem_size: usize,
    ) -> Result<()> {
        if new_cap > old_cap {
            let bytes = (new_cap - old_cap) * elem_size;
            self.charge(bytes)?;
        }
        Ok(())
    }
}
