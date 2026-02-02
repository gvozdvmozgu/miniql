use std::cell::OnceCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::hash::{Hash, Hasher};

use bumpalo::Bump;
use hashbrown::HashMap;
use hashbrown::hash_map::RawEntryMut;
use rustc_hash::{FxBuildHasher, FxHasher};
use smallvec::SmallVec;

use crate::compare::compare_value_refs;
use crate::pager::{PageId, Pager};
use crate::table::{
    self, BytesSpan, Corruption, QueryError, RawBytes, ValueKind, ValueRef, ValueSlot,
};

/// Filter expression tree used by `Scan::filter`.
#[derive(Debug, Clone)]
pub enum Expr {
    Col(u16),
    Lit(ValueLit),

    Eq(Box<Expr>, Box<Expr>),
    Ne(Box<Expr>, Box<Expr>),
    Lt(Box<Expr>, Box<Expr>),
    Le(Box<Expr>, Box<Expr>),
    Gt(Box<Expr>, Box<Expr>),
    Ge(Box<Expr>, Box<Expr>),

    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Not(Box<Expr>),

    IsNull(Box<Expr>),
    IsNotNull(Box<Expr>),
}

#[derive(Debug, Clone)]
enum CompiledExpr {
    Col { col: u16, idx: usize },
    Lit(ValueLit),
    CmpColLit { idx: usize, op: CmpOp, lit: ValueLit },

    Eq(Box<CompiledExpr>, Box<CompiledExpr>),
    Ne(Box<CompiledExpr>, Box<CompiledExpr>),
    Lt(Box<CompiledExpr>, Box<CompiledExpr>),
    Le(Box<CompiledExpr>, Box<CompiledExpr>),
    Gt(Box<CompiledExpr>, Box<CompiledExpr>),
    Ge(Box<CompiledExpr>, Box<CompiledExpr>),

    And(Box<CompiledExpr>, Box<CompiledExpr>),
    Or(Box<CompiledExpr>, Box<CompiledExpr>),
    Not(Box<CompiledExpr>),

    IsNull(Box<CompiledExpr>),
    IsNotNull(Box<CompiledExpr>),
}

/// Literal value used in filter expressions.
#[derive(Debug, Clone)]
pub enum ValueLit {
    Null,
    Integer(i64),
    Real(f64),
    Text(Vec<u8>),
}

/// Aggregate expression used in `Aggregate` projections.
#[derive(Debug, Clone)]
pub enum AggExpr {
    /// Non-aggregate expression (must appear in GROUP BY).
    Value(Expr),
    /// COUNT(*)
    CountStar,
    /// COUNT(expr)
    Count(Expr),
    /// SUM(expr)
    Sum(Expr),
    /// AVG(expr)
    Avg(Expr),
    /// MIN(expr)
    Min(Expr),
    /// MAX(expr)
    Max(Expr),
}

impl AggExpr {
    /// Use a non-aggregate expression in the projection (must appear in GROUP
    /// BY).
    pub fn value(expr: Expr) -> Self {
        Self::Value(expr)
    }

    /// COUNT(*)
    pub fn count_star() -> Self {
        Self::CountStar
    }

    /// COUNT(expr)
    pub fn count(expr: Expr) -> Self {
        Self::Count(expr)
    }

    /// SUM(expr)
    pub fn sum(expr: Expr) -> Self {
        Self::Sum(expr)
    }

    /// AVG(expr)
    pub fn avg(expr: Expr) -> Self {
        Self::Avg(expr)
    }

    /// MIN(expr)
    pub fn min(expr: Expr) -> Self {
        Self::Min(expr)
    }

    /// MAX(expr)
    pub fn max(expr: Expr) -> Self {
        Self::Max(expr)
    }
}

/// Shorthand for `AggExpr::value`.
pub fn group(expr: Expr) -> AggExpr {
    AggExpr::value(expr)
}

/// Shorthand for `AggExpr::count_star`.
pub fn count_star() -> AggExpr {
    AggExpr::count_star()
}

/// Shorthand for `AggExpr::count`.
pub fn count(expr: Expr) -> AggExpr {
    AggExpr::count(expr)
}

/// Shorthand for `AggExpr::sum`.
pub fn sum(expr: Expr) -> AggExpr {
    AggExpr::sum(expr)
}

/// Shorthand for `AggExpr::avg`.
pub fn avg(expr: Expr) -> AggExpr {
    AggExpr::avg(expr)
}

/// Shorthand for `AggExpr::min`.
pub fn min(expr: Expr) -> AggExpr {
    AggExpr::min(expr)
}

/// Shorthand for `AggExpr::max`.
pub fn max(expr: Expr) -> AggExpr {
    AggExpr::max(expr)
}

/// Ordering direction for `ORDER BY`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderDir {
    Asc,
    Desc,
}

/// Column + direction for `ORDER BY`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrderBy {
    pub col: u16,
    pub dir: OrderDir,
}

impl OrderBy {
    /// Order by a column ascending.
    pub fn asc(col: u16) -> Self {
        Self { col, dir: OrderDir::Asc }
    }

    /// Order by a column descending.
    pub fn desc(col: u16) -> Self {
        Self { col, dir: OrderDir::Desc }
    }
}

/// Shorthand for `OrderBy::asc`.
///
/// ```rust
/// use std::path::Path;
///
/// use miniql::{Db, ScanScratch, asc};
///
/// let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/users.db");
/// let db = Db::open(path).unwrap();
/// let table = db.table("users").unwrap();
/// let mut scratch = ScanScratch::with_capacity(3, 0);
/// table.scan().order_by([asc(1)]).for_each(&mut scratch, |_, _| Ok(())).unwrap();
/// ```
pub fn asc(col: u16) -> OrderBy {
    OrderBy::asc(col)
}

/// Shorthand for `OrderBy::desc`.
///
/// ```rust
/// use std::path::Path;
///
/// use miniql::{Db, ScanScratch, desc};
///
/// let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/users.db");
/// let db = Db::open(path).unwrap();
/// let table = db.table("users").unwrap();
/// let mut scratch = ScanScratch::with_capacity(3, 0);
/// table.scan().order_by([desc(2)]).for_each(&mut scratch, |_, _| Ok(())).unwrap();
/// ```
pub fn desc(col: u16) -> OrderBy {
    OrderBy::desc(col)
}

/// Create a column reference expression.
pub fn col(i: u16) -> Expr {
    Expr::Col(i)
}

/// Create an integer literal expression.
pub fn lit_i64(v: i64) -> Expr {
    Expr::Lit(ValueLit::Integer(v))
}

/// Create a real literal expression.
pub fn lit_f64(v: f64) -> Expr {
    Expr::Lit(ValueLit::Real(v))
}

/// Create a text literal expression.
pub fn lit_bytes(v: impl Into<Vec<u8>>) -> Expr {
    Expr::Lit(ValueLit::Text(v.into()))
}

/// Create a NULL literal expression.
pub fn lit_null() -> Expr {
    Expr::Lit(ValueLit::Null)
}

impl Expr {
    /// Compare two expressions for equality.
    pub fn eq(self, rhs: Expr) -> Expr {
        Expr::Eq(Box::new(self), Box::new(rhs))
    }

    /// Compare two expressions for inequality.
    pub fn ne(self, rhs: Expr) -> Expr {
        Expr::Ne(Box::new(self), Box::new(rhs))
    }

    /// Compare two expressions with `<`.
    pub fn lt(self, rhs: Expr) -> Expr {
        Expr::Lt(Box::new(self), Box::new(rhs))
    }

    /// Compare two expressions with `<=`.
    pub fn le(self, rhs: Expr) -> Expr {
        Expr::Le(Box::new(self), Box::new(rhs))
    }

    /// Compare two expressions with `>`.
    pub fn gt(self, rhs: Expr) -> Expr {
        Expr::Gt(Box::new(self), Box::new(rhs))
    }

    /// Compare two expressions with `>=`.
    pub fn ge(self, rhs: Expr) -> Expr {
        Expr::Ge(Box::new(self), Box::new(rhs))
    }

    /// Logical AND of two expressions.
    pub fn and(self, rhs: Expr) -> Expr {
        Expr::And(Box::new(self), Box::new(rhs))
    }

    /// Logical OR of two expressions.
    pub fn or(self, rhs: Expr) -> Expr {
        Expr::Or(Box::new(self), Box::new(rhs))
    }

    #[allow(clippy::should_implement_trait)]
    /// Logical NOT of an expression.
    pub fn not(self) -> Expr {
        Expr::Not(Box::new(self))
    }

    /// Check whether the expression evaluates to NULL.
    pub fn is_null(self) -> Expr {
        Expr::IsNull(Box::new(self))
    }

    /// Check whether the expression evaluates to NOT NULL.
    pub fn is_not_null(self) -> Expr {
        Expr::IsNotNull(Box::new(self))
    }

    fn collect_cols(&self, out: &mut SmallVec<[u16; 8]>) {
        match self {
            Expr::Col(idx) => out.push(*idx),
            Expr::Lit(_) => {}
            Expr::Eq(lhs, rhs)
            | Expr::Ne(lhs, rhs)
            | Expr::Lt(lhs, rhs)
            | Expr::Le(lhs, rhs)
            | Expr::Gt(lhs, rhs)
            | Expr::Ge(lhs, rhs)
            | Expr::And(lhs, rhs)
            | Expr::Or(lhs, rhs) => {
                lhs.collect_cols(out);
                rhs.collect_cols(out);
            }
            Expr::Not(inner) | Expr::IsNull(inner) | Expr::IsNotNull(inner) => {
                inner.collect_cols(out);
            }
        }
    }
}

impl std::ops::Not for Expr {
    type Output = Expr;

    fn not(self) -> Self::Output {
        Expr::Not(Box::new(self))
    }
}

/// Scratch buffers for scans.
#[derive(Debug)]
pub struct ScanScratch {
    stack: Vec<PageId>,
    values: Vec<ValueSlot>,
    bytes: Vec<u8>,
    serials: Vec<u64>,
    offsets: Vec<u32>,
}

impl Default for ScanScratch {
    fn default() -> Self {
        Self::with_capacity(0, 0)
    }
}

impl ScanScratch {
    /// Create an empty scratch buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a scratch buffer with capacity hints.
    pub fn with_capacity(values: usize, overflow: usize) -> Self {
        Self {
            stack: Vec::with_capacity(64),
            values: Vec::with_capacity(values),
            bytes: Vec::with_capacity(overflow),
            serials: Vec::with_capacity(values),
            offsets: Vec::with_capacity(values),
        }
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn split_mut(
        &mut self,
    ) -> (&mut Vec<ValueSlot>, &mut Vec<u8>, &mut Vec<u64>, &mut Vec<u32>, &mut Vec<PageId>) {
        (&mut self.values, &mut self.bytes, &mut self.serials, &mut self.offsets, &mut self.stack)
    }
}

/// Row view returned by scans.
#[derive(Clone, Copy)]
pub struct Row<'row> {
    values: &'row [ValueSlot],
    proj_map: Option<&'row [usize]>,
}

impl<'row> Row<'row> {
    pub(crate) fn from_raw(values: &'row [ValueSlot], proj_map: Option<&'row [usize]>) -> Self {
        Self { values, proj_map }
    }

    pub(crate) fn values_raw(&self) -> &'row [ValueSlot] {
        self.values
    }

    /// Number of columns in the projected row.
    #[inline]
    pub fn len(&self) -> usize {
        self.proj_map.map_or(self.values.len(), |map| map.len())
    }

    /// Returns true when there are no columns.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return a value reference by column index.
    #[inline]
    pub fn get(&self, i: usize) -> Option<ValueRef<'row>> {
        let idx = match self.proj_map {
            Some(map) => *map.get(i)?,
            None => i,
        };
        self.values.get(idx).copied().map(raw_to_ref)
    }

    /// Return an `i64` value or a type mismatch error.
    pub fn get_i64(&self, i: usize) -> table::Result<i64> {
        match self.get(i) {
            Some(ValueRef::Integer(value)) => Ok(value),
            Some(other) => Err(table::Error::TypeMismatch {
                col: i,
                expected: ValueKind::Integer,
                got: value_kind(other),
            }),
            None => Err(table::Error::TypeMismatch {
                col: i,
                expected: ValueKind::Integer,
                got: ValueKind::Missing,
            }),
        }
    }

    /// Return an `f64` value or a type mismatch error.
    pub fn get_f64(&self, i: usize) -> table::Result<f64> {
        match self.get(i) {
            Some(ValueRef::Real(value)) => Ok(value),
            Some(other) => Err(table::Error::TypeMismatch {
                col: i,
                expected: ValueKind::Real,
                got: value_kind(other),
            }),
            None => Err(table::Error::TypeMismatch {
                col: i,
                expected: ValueKind::Real,
                got: ValueKind::Missing,
            }),
        }
    }

    /// Return a UTF-8 string or a type mismatch error.
    pub fn get_text(&self, i: usize) -> table::Result<&'row str> {
        match self.get(i) {
            Some(ValueRef::Text(bytes)) => Ok(std::str::from_utf8(bytes)?),
            Some(other) => Err(table::Error::TypeMismatch {
                col: i,
                expected: ValueKind::Text,
                got: value_kind(other),
            }),
            None => Err(table::Error::TypeMismatch {
                col: i,
                expected: ValueKind::Text,
                got: ValueKind::Missing,
            }),
        }
    }

    /// Return text/blob bytes or a type mismatch error.
    pub fn get_bytes(&self, i: usize) -> table::Result<&'row [u8]> {
        match self.get(i) {
            Some(ValueRef::Text(bytes)) => Ok(bytes),
            Some(ValueRef::Blob(bytes)) => Ok(bytes),
            Some(other) => Err(table::Error::TypeMismatch {
                col: i,
                expected: ValueKind::Bytes,
                got: value_kind(other),
            }),
            None => Err(table::Error::TypeMismatch {
                col: i,
                expected: ValueKind::Bytes,
                got: ValueKind::Missing,
            }),
        }
    }
}

type FilterFn<'db> = Box<dyn for<'row> FnMut(&Row<'row>) -> table::Result<bool> + 'db>;

/// Builder for table scans.
pub struct Scan<'db> {
    pager: &'db Pager,
    root: PageId,
    col_count_hint: Option<usize>,
    projection: Option<Vec<u16>>,
    filter_expr: Option<Expr>,
    filter_fn: Option<FilterFn<'db>>,
    order_by: Option<Vec<OrderBy>>,
    limit: Option<usize>,
}

/// Compiled scan ready for execution.
pub struct PreparedScan<'db> {
    pager: &'db Pager,
    root: PageId,
    needed_cols: Option<Box<[u16]>>,
    proj_map: Option<Box<[usize]>>,
    /// Maps ORDER BY column position -> values array index (None = decode all)
    order_val_map: Option<Box<[usize]>>,
    referenced_cols: Box<[u16]>,
    compiled_expr: Option<CompiledExpr>,
    filter_fn: Option<FilterFn<'db>>,
    order_by: Option<Box<[OrderBy]>>,
    limit: Option<usize>,
    column_count_hint: OnceCell<usize>,
}

/// Deprecated alias for `PreparedScan`.
#[deprecated(note = "use PreparedScan")]
pub type CompiledScan<'db> = PreparedScan<'db>;

impl<'db> Scan<'db> {
    /// Create a scan over a table root.
    pub fn table(pager: &'db Pager, root: PageId) -> Self {
        Self {
            pager,
            root,
            col_count_hint: None,
            projection: None,
            filter_expr: None,
            filter_fn: None,
            order_by: None,
            limit: None,
        }
    }

    pub(crate) fn from_root_with_hint(
        pager: &'db Pager,
        root: PageId,
        col_count_hint: Option<usize>,
    ) -> Self {
        Self {
            pager,
            root,
            col_count_hint,
            projection: None,
            filter_expr: None,
            filter_fn: None,
            order_by: None,
            limit: None,
        }
    }

    /// Project specific columns by index.
    pub fn project<const N: usize>(mut self, cols: [u16; N]) -> Self {
        self.projection = Some(cols.to_vec());
        self
    }

    /// Apply a filter expression.
    pub fn filter(mut self, expr: Expr) -> Self {
        self.filter_expr = Some(expr);
        self.filter_fn = None;
        self
    }

    /// Apply a custom filter function (disables predicate compilation).
    pub fn filter_fn_slow<F>(mut self, f: F) -> Self
    where
        F: for<'row> FnMut(&Row<'row>) -> table::Result<bool> + 'db,
    {
        self.filter_expr = None;
        self.filter_fn = Some(Box::new(f));
        self
    }

    /// Apply a custom filter function.
    pub fn filter_fn<F>(self, f: F) -> Self
    where
        F: for<'row> FnMut(&Row<'row>) -> table::Result<bool> + 'db,
    {
        self.filter_fn_slow(f)
    }

    /// Apply `ORDER BY` to a scan.
    ///
    /// ```rust
    /// use std::path::Path;
    ///
    /// use miniql::{Db, ScanScratch, asc, desc};
    ///
    /// let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/users.db");
    /// let db = Db::open(path).unwrap();
    /// let table = db.table("users").unwrap();
    /// let mut scratch = ScanScratch::with_capacity(3, 0);
    /// table.scan().order_by([asc(1), desc(2)]).for_each(&mut scratch, |_, _| Ok(())).unwrap();
    /// ```
    pub fn order_by<const N: usize>(mut self, cols: [OrderBy; N]) -> Self {
        if N == 0 {
            self.order_by = None;
        } else {
            self.order_by = Some(cols.to_vec());
        }
        self
    }

    /// Limit the number of rows returned.
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Build an aggregate query projection over this scan.
    pub fn aggregate<const N: usize>(self, exprs: [AggExpr; N]) -> Aggregate<'db> {
        Aggregate { scan: self, group_by: Vec::new(), select: exprs.to_vec(), having: None }
    }

    pub(crate) fn projection(&self) -> Option<&[u16]> {
        self.projection.as_deref()
    }

    pub(crate) fn has_order_by(&self) -> bool {
        matches!(self.order_by.as_deref(), Some(cols) if !cols.is_empty())
    }

    /// Check if scan output is sorted ascending by a specific column.
    pub(crate) fn sorted_asc_by_col(&self, col: u16) -> bool {
        match self.order_by.as_deref() {
            Some([first, ..]) => first.col == col && matches!(first.dir, OrderDir::Asc),
            _ => false,
        }
    }

    pub(crate) fn with_projection_override(mut self, projection: Option<Vec<u16>>) -> Self {
        self.projection = projection;
        self
    }

    /// Compile the scan into an executable plan.
    pub fn compile(self) -> table::Result<PreparedScan<'db>> {
        let Scan {
            pager,
            root,
            col_count_hint,
            projection,
            filter_expr,
            filter_fn,
            order_by,
            limit,
        } = self;

        let mut pred_cols = SmallVec::<[u16; 8]>::new();
        if let Some(expr) = &filter_expr {
            expr.collect_cols(&mut pred_cols);
        }

        let decode_all = filter_fn.is_some();
        let plan = build_plan(projection.as_ref(), &pred_cols, order_by.as_deref(), decode_all);
        let compiled_expr = filter_expr
            .as_ref()
            .map(|expr| compile_expr(expr, plan.needed_cols.as_deref()))
            .transpose()?;

        if let Some(count) = col_count_hint
            && let Some(err) = validate_columns(&plan.referenced_cols, count)
        {
            return Err(err);
        }

        let Plan { needed_cols, proj_map, order_val_map, referenced_cols } = plan;
        let column_count_hint = OnceCell::new();
        if let Some(count) = col_count_hint {
            let _ = column_count_hint.set(count);
        }

        Ok(PreparedScan {
            pager,
            root,
            needed_cols: needed_cols.map(|cols| cols.into_boxed_slice()),
            proj_map: proj_map.map(|cols| cols.into_boxed_slice()),
            order_val_map: order_val_map.map(|cols| cols.into_boxed_slice()),
            referenced_cols: referenced_cols.into_boxed_slice(),
            compiled_expr,
            filter_fn,
            order_by: order_by.map(|cols| cols.into_boxed_slice()),
            limit,
            column_count_hint,
        })
    }

    /// Execute the scan and invoke `cb` for each row with on-demand decoding.
    ///
    /// Values are decoded on-demand when `RowView::get()` is called. For
    /// repeated column access, use `RowView::cached` with `RowCache` to
    /// avoid re-scanning headers.
    /// Supports filters, ORDER BY, and LIMIT.
    pub fn for_each<F>(self, scratch: &mut ScanScratch, mut cb: F) -> table::Result<()>
    where
        F: for<'row> FnMut(i64, table::RowView<'row>) -> table::Result<()>,
    {
        let mut prepared = self.compile()?;
        prepared.for_each(scratch, &mut cb)
    }
}

impl<'db> PreparedScan<'db> {
    pub(crate) fn pager(&self) -> &'db Pager {
        self.pager
    }

    pub(crate) fn root(&self) -> PageId {
        self.root
    }

    pub(crate) fn needed_cols(&self) -> Option<&[u16]> {
        self.needed_cols.as_deref()
    }

    pub(crate) fn proj_map(&self) -> Option<&[usize]> {
        self.proj_map.as_deref()
    }

    pub(crate) fn column_count_hint(&self) -> Option<usize> {
        self.column_count_hint.get().copied()
    }

    pub(crate) fn limit(&self) -> Option<usize> {
        self.limit
    }

    pub(crate) fn has_filters(&self) -> bool {
        self.compiled_expr.is_some() || self.filter_fn.is_some()
    }

    pub(crate) fn eval_payload<'row>(
        &'row mut self,
        payload: table::PayloadRef<'row>,
        values: &'row mut Vec<ValueSlot>,
        bytes: &'row mut Vec<u8>,
        serials: &'row mut Vec<u64>,
        offsets: &'row mut Vec<u32>,
    ) -> table::Result<Option<Row<'row>>> {
        self.eval_payload_with_filters(payload, values, bytes, serials, offsets, true)
    }

    pub(crate) fn eval_payload_with_filters<'row>(
        &'row mut self,
        payload: table::PayloadRef<'row>,
        values: &'row mut Vec<ValueSlot>,
        bytes: &'row mut Vec<u8>,
        serials: &'row mut Vec<u64>,
        offsets: &'row mut Vec<u32>,
        apply_filters: bool,
    ) -> table::Result<Option<Row<'row>>> {
        let needed_cols = self.needed_cols.as_deref();
        let count = table::decode_record_project_into(
            payload,
            needed_cols,
            values,
            bytes,
            serials,
            offsets,
        )?;

        if let Some(expected) = self.column_count_hint.get().copied() {
            if expected != count {
                return Err(table::Error::Corrupted(Corruption::RowColumnCountMismatch));
            }
        } else {
            if let Some(err) = validate_columns(&self.referenced_cols, count) {
                return Err(err);
            }
            let _ = self.column_count_hint.set(count);
        }

        let values = values.as_slice();
        let scratch_bytes = bytes.as_slice();

        if apply_filters {
            if let Some(expr) = self.compiled_expr.as_ref()
                && eval_compiled_expr(expr, values, scratch_bytes)? != Truth::True
            {
                return Ok(None);
            }

            if let Some(filter_fn) = self.filter_fn.as_mut() {
                let row = Row { values, proj_map: None };
                if !filter_fn(&row)? {
                    return Ok(None);
                }
            }
        }

        let row = Row { values, proj_map: self.proj_map.as_deref() };
        Ok(Some(row))
    }

    pub(crate) fn decode_and_filter_payload<'row>(
        &mut self,
        payload: table::PayloadRef<'row>,
        values: &'row mut Vec<ValueSlot>,
        bytes: &'row mut Vec<u8>,
        serials: &'row mut Vec<u64>,
        offsets: &'row mut Vec<u32>,
    ) -> table::Result<bool> {
        let needed_cols = self.needed_cols.as_deref();
        let count = table::decode_record_project_into(
            payload,
            needed_cols,
            values,
            bytes,
            serials,
            offsets,
        )?;

        if let Some(expected) = self.column_count_hint.get().copied() {
            if expected != count {
                return Err(table::Error::Corrupted(Corruption::RowColumnCountMismatch));
            }
        } else {
            if let Some(err) = validate_columns(&self.referenced_cols, count) {
                return Err(err);
            }
            let _ = self.column_count_hint.set(count);
        }

        let values = values.as_slice();
        let scratch_bytes = bytes.as_slice();

        if let Some(expr) = self.compiled_expr.as_ref()
            && eval_compiled_expr(expr, values, scratch_bytes)? != Truth::True
        {
            return Ok(false);
        }
        if let Some(filter_fn) = self.filter_fn.as_mut() {
            let row = Row { values, proj_map: None };
            if !filter_fn(&row)? {
                return Ok(false);
            }
        }

        Ok(true)
    }

    pub(crate) fn build_sort_keys(
        &self,
        order_by: &[OrderBy],
        values: &[ValueSlot],
        scratch_bytes: &[u8],
        key_arena: &Bump,
    ) -> SmallVec<[SortKey; 4]> {
        let mut keys = SmallVec::<[SortKey; 4]>::with_capacity(order_by.len());
        let order_val_map = self.order_val_map.as_deref();
        for (i, order) in order_by.iter().enumerate() {
            let val_idx = match order_val_map {
                Some(map) => map[i],
                None => order.col as usize,
            };
            let slot = values.get(val_idx).copied().unwrap_or(ValueSlot::Null);
            let value = stabilize_sort_key_value(slot, scratch_bytes, key_arena);
            keys.push(SortKey { value, dir: order.dir });
        }
        keys
    }

    #[inline]
    pub(crate) fn row_view_from_payload<'row>(
        payload: table::PayloadRef<'row>,
    ) -> table::Result<table::RowView<'row>> {
        match payload {
            table::PayloadRef::Inline(data) => table::RowView::from_inline(data),
            table::PayloadRef::Overflow(ovf) => table::RowView::from_inline(ovf.local()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn eval_index_payload_with_map<'row>(
        &'row mut self,
        payload: table::PayloadRef<'row>,
        needed_map: &[(u16, usize)],
        values: &'row mut Vec<ValueSlot>,
        bytes: &'row mut Vec<u8>,
        serials: &'row mut Vec<u64>,
        offsets: &'row mut Vec<u32>,
        apply_filters: bool,
    ) -> table::Result<Option<Row<'row>>> {
        let Some(needed_cols) = self.needed_cols.as_deref() else {
            return Err(table::Error::Corrupted(Corruption::CoveringIndexRequiresColumnProjection));
        };
        if needed_cols.len() != needed_map.len() {
            return Err(table::Error::Corrupted(Corruption::CoveringIndexColumnMapMismatch));
        }

        let _count = table::decode_record_project_into_mapped(
            payload, needed_map, values, bytes, serials, offsets,
        )?;

        let values = values.as_slice();
        let scratch_bytes = bytes.as_slice();

        if apply_filters {
            if let Some(expr) = self.compiled_expr.as_ref()
                && eval_compiled_expr(expr, values, scratch_bytes)? != Truth::True
            {
                return Ok(None);
            }

            if let Some(filter_fn) = self.filter_fn.as_mut() {
                let row = Row { values, proj_map: None };
                if !filter_fn(&row)? {
                    return Ok(None);
                }
            }
        }

        let row = Row { values, proj_map: self.proj_map.as_deref() };
        Ok(Some(row))
    }

    /// Execute the prepared scan with on-demand row decoding.
    ///
    /// Values are decoded on-demand when `RowView::get()` is called. For
    /// repeated column access, use `RowView::cached` with `RowCache` to
    /// avoid re-scanning headers. This is more efficient when you only
    /// access a subset of columns.
    ///
    /// Supports filters, ORDER BY, and LIMIT. For filters and ORDER BY,
    /// only the required columns are decoded; the callback receives a
    /// `RowView` for on-demand access to other columns.
    ///
    /// # Example
    /// ```rust
    /// use std::path::Path;
    ///
    /// use miniql::{Db, ScanScratch};
    ///
    /// let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/users.db");
    /// let db = Db::open(path).unwrap();
    /// let table = db.table("users").unwrap();
    /// let mut scratch = ScanScratch::new();
    /// let mut count = 0;
    /// table
    ///     .scan()
    ///     .compile()
    ///     .unwrap()
    ///     .for_each(&mut scratch, |_, _| {
    ///         count += 1;
    ///         Ok(())
    ///     })
    ///     .unwrap();
    /// assert_eq!(count, 2);
    /// ```
    pub fn for_each<F>(&mut self, scratch: &mut ScanScratch, mut cb: F) -> table::Result<()>
    where
        F: for<'row> FnMut(i64, table::RowView<'row>) -> table::Result<()>,
    {
        if self.limit == Some(0) {
            return Ok(());
        }

        let has_order = matches!(self.order_by.as_deref(), Some(o) if !o.is_empty());
        let has_filter = self.compiled_expr.is_some() || self.filter_fn.is_some();

        if has_order {
            let order_by = self.order_by.take().unwrap();
            let result = self.for_each_ordered(scratch, &order_by, &mut cb);
            self.order_by = Some(order_by);
            return result;
        }

        if has_filter {
            return self.for_each_filtered(scratch, &mut cb);
        }

        // Simple case: no filter, no ORDER BY
        let pager = self.pager;
        let root = self.root;
        let limit = self.limit;
        let (_, _, _, _, btree_stack) = scratch.split_mut();

        match limit {
            Some(limit) => {
                let mut seen = 0usize;
                btree_stack.clear();
                btree_stack.push(root);
                table::scan_table_with_stack(pager, btree_stack, &mut |rowid, row| {
                    cb(rowid, row)?;
                    seen += 1;
                    if seen >= limit {
                        return Err(table::Error::Corrupted(Corruption::LimitReached));
                    }
                    Ok(())
                })
                .or_else(|e| {
                    if matches!(&e, table::Error::Corrupted(Corruption::LimitReached)) {
                        Ok(())
                    } else {
                        Err(e)
                    }
                })
            }
            None => {
                btree_stack.clear();
                btree_stack.push(root);
                table::scan_table_with_stack(pager, btree_stack, &mut cb)
            }
        }
    }

    fn for_each_filtered<F>(&mut self, scratch: &mut ScanScratch, cb: &mut F) -> table::Result<()>
    where
        F: for<'row> FnMut(i64, table::RowView<'row>) -> table::Result<()>,
    {
        let pager = self.pager;
        let root = self.root;
        let limit = self.limit;
        let (values, bytes, serials, offsets, btree_stack) = scratch.split_mut();

        match limit {
            Some(limit) => {
                let mut seen = 0usize;
                table::scan_table_cells_with_scratch_and_stack_until(
                    pager,
                    root,
                    btree_stack,
                    |cell| {
                        let rowid = cell.rowid();
                        let payload = cell.payload();

                        if !self
                            .decode_and_filter_payload(payload, values, bytes, serials, offsets)?
                        {
                            return Ok(None);
                        }

                        let row_view = Self::row_view_from_payload(payload)?;
                        cb(rowid, row_view)?;
                        seen += 1;

                        if seen >= limit {
                            return Ok(Some(()));
                        }
                        Ok(None)
                    },
                )?;
            }
            None => {
                table::scan_table_cells_with_scratch_and_stack(pager, root, btree_stack, |cell| {
                    let rowid = cell.rowid();
                    let payload = cell.payload();

                    if !self.decode_and_filter_payload(payload, values, bytes, serials, offsets)? {
                        return Ok(());
                    }

                    let row_view = Self::row_view_from_payload(payload)?;
                    cb(rowid, row_view)?;
                    Ok(())
                })?;
            }
        }

        Ok(())
    }

    fn for_each_ordered<F>(
        &mut self,
        scratch: &mut ScanScratch,
        order_by: &[OrderBy],
        cb: &mut F,
    ) -> table::Result<()>
    where
        F: for<'row> FnMut(i64, table::RowView<'row>) -> table::Result<()>,
    {
        let pager = self.pager;
        let root = self.root;
        let limit = self.limit;

        let (values, bytes, serials, offsets, btree_stack) = scratch.split_mut();
        let key_arena = Bump::new();
        let mut seq = 0u64;

        match limit {
            Some(limit) => {
                let mut heap = BinaryHeap::with_capacity(limit.saturating_add(1));
                table::scan_table_cells_with_scratch_and_stack(pager, root, btree_stack, |cell| {
                    let rowid = cell.rowid();
                    let page_id = cell.page_id();
                    let cell_offset = cell.cell_offset();
                    let payload = cell.payload();

                    if !self.decode_and_filter_payload(payload, values, bytes, serials, offsets)? {
                        return Ok(());
                    }

                    let keys = self.build_sort_keys(
                        order_by,
                        values.as_slice(),
                        bytes.as_slice(),
                        &key_arena,
                    );

                    let entry = SortEntry { rowid, page_id, cell_offset, keys, seq };
                    seq = seq.wrapping_add(1);

                    if heap.len() < limit {
                        heap.push(entry);
                    } else if let Some(top) = heap.peek()
                        && entry.cmp(top) == Ordering::Less
                    {
                        heap.pop();
                        heap.push(entry);
                    }

                    Ok(())
                })?;

                let mut rows = heap.into_vec();
                rows.sort();

                for entry in rows {
                    let cell = table::read_table_cell_ref_from_bytes(
                        pager,
                        entry.page_id,
                        entry.cell_offset,
                    )?;
                    let payload = cell.payload();
                    let row_view = Self::row_view_from_payload(payload)?;
                    cb(entry.rowid, row_view)?;
                }
            }
            None => {
                let mut entries = Vec::with_capacity(256);
                table::scan_table_cells_with_scratch_and_stack(pager, root, btree_stack, |cell| {
                    let rowid = cell.rowid();
                    let page_id = cell.page_id();
                    let cell_offset = cell.cell_offset();
                    let payload = cell.payload();

                    if !self.decode_and_filter_payload(payload, values, bytes, serials, offsets)? {
                        return Ok(());
                    }

                    let keys = self.build_sort_keys(
                        order_by,
                        values.as_slice(),
                        bytes.as_slice(),
                        &key_arena,
                    );

                    entries.push(SortEntry { rowid, page_id, cell_offset, keys, seq });
                    seq = seq.wrapping_add(1);

                    Ok(())
                })?;

                entries.sort();

                for entry in entries {
                    let cell = table::read_table_cell_ref_from_bytes(
                        pager,
                        entry.page_id,
                        entry.cell_offset,
                    )?;
                    let payload = cell.payload();
                    let row_view = Self::row_view_from_payload(payload)?;
                    cb(entry.rowid, row_view)?;
                }
            }
        }

        Ok(())
    }

    /// Execute the scan with eager (pre-decoded) rows.
    ///
    /// Internal method used by join operations that need access to raw values.
    /// Supports filters and LIMIT, but not ORDER BY - use `for_each` for ORDER
    /// BY.
    pub(crate) fn for_each_eager<F>(
        &mut self,
        scratch: &mut ScanScratch,
        mut cb: F,
    ) -> table::Result<()>
    where
        F: for<'row> FnMut(i64, Row<'row>) -> table::Result<()>,
    {
        if self.limit == Some(0) {
            return Ok(());
        }

        let pager = self.pager;
        let root = self.root;
        let limit = self.limit;

        let (values, bytes, serials, offsets, btree_stack) = scratch.split_mut();

        match limit {
            Some(limit) => {
                let mut seen = 0usize;
                table::scan_table_cells_with_scratch_and_stack_until(
                    pager,
                    root,
                    btree_stack,
                    |cell| {
                        let rowid = cell.rowid();
                        let Some(row) =
                            self.eval_payload(cell.payload(), values, bytes, serials, offsets)?
                        else {
                            return Ok(None);
                        };
                        cb(rowid, row)?;
                        seen += 1;
                        if seen >= limit {
                            return Ok(Some(()));
                        }
                        Ok(None)
                    },
                )?;
            }
            None => {
                table::scan_table_cells_with_scratch_and_stack(pager, root, btree_stack, |cell| {
                    let rowid = cell.rowid();
                    let Some(row) =
                        self.eval_payload(cell.payload(), values, bytes, serials, offsets)?
                    else {
                        return Ok(());
                    };
                    cb(rowid, row)?;
                    Ok(())
                })?;
            }
        }

        Ok(())
    }
}

/// Builder for aggregate queries over a scan.
pub struct Aggregate<'db> {
    scan: Scan<'db>,
    group_by: Vec<Expr>,
    select: Vec<AggExpr>,
    having: Option<Expr>,
}

/// Compiled aggregate query ready for execution.
pub struct PreparedAggregate<'db> {
    scan: PreparedScan<'db>,
    group_by: Vec<ValueExpr>,
    select: Vec<SelectItemPlan>,
    agg_template: Vec<AggState>,
    having: Option<CompiledExpr>,
    has_agg: bool,
}

impl<'db> Aggregate<'db> {
    /// Apply GROUP BY expressions.
    pub fn group_by<const N: usize>(mut self, exprs: [Expr; N]) -> Self {
        self.group_by = exprs.to_vec();
        self
    }

    /// Apply a HAVING predicate evaluated against the aggregate output row.
    pub fn having(mut self, expr: Expr) -> Self {
        self.having = Some(expr);
        self
    }

    /// Compile the aggregate query into an executable plan.
    pub fn compile(self) -> table::Result<PreparedAggregate<'db>> {
        let Aggregate { scan, group_by, select, having } = self;

        if select.is_empty() {
            return Err(table::Error::Query(QueryError::AggregateProjectionEmpty));
        }

        let mut group_by_values = Vec::with_capacity(group_by.len());
        for expr in &group_by {
            group_by_values.push(parse_value_expr(expr)?);
        }

        let mut select_plan = Vec::with_capacity(select.len());
        let mut agg_template = Vec::new();
        let mut has_agg = false;

        for expr in select {
            match expr {
                AggExpr::Value(inner) => {
                    let value_expr = parse_value_expr(&inner)?;
                    if group_by_values.is_empty() {
                        return Err(table::Error::Query(QueryError::NonAggregateRequiresGroupBy));
                    }
                    let Some(idx) = group_by_values.iter().position(|value| value == &value_expr)
                    else {
                        return Err(table::Error::Query(
                            QueryError::NonAggregateMustAppearInGroupBy,
                        ));
                    };
                    select_plan.push(SelectItemPlan::GroupKey(idx));
                }
                AggExpr::CountStar => {
                    has_agg = true;
                    let idx = agg_template.len();
                    agg_template.push(AggState::Count { expr: None, n: 0 });
                    select_plan.push(SelectItemPlan::Agg(idx));
                }
                AggExpr::Count(inner) => {
                    has_agg = true;
                    let value_expr = parse_value_expr(&inner)?;
                    let idx = agg_template.len();
                    agg_template.push(AggState::Count { expr: Some(value_expr), n: 0 });
                    select_plan.push(SelectItemPlan::Agg(idx));
                }
                AggExpr::Sum(inner) => {
                    has_agg = true;
                    let value_expr = parse_value_expr(&inner)?;
                    let idx = agg_template.len();
                    agg_template.push(AggState::Sum {
                        expr: value_expr,
                        count: 0,
                        int_sum: 0,
                        real_sum: 0.0,
                        use_real: false,
                    });
                    select_plan.push(SelectItemPlan::Agg(idx));
                }
                AggExpr::Avg(inner) => {
                    has_agg = true;
                    let value_expr = parse_value_expr(&inner)?;
                    let idx = agg_template.len();
                    agg_template.push(AggState::Avg { expr: value_expr, count: 0, sum: 0.0 });
                    select_plan.push(SelectItemPlan::Agg(idx));
                }
                AggExpr::Min(inner) => {
                    has_agg = true;
                    let value_expr = parse_value_expr(&inner)?;
                    let idx = agg_template.len();
                    agg_template.push(AggState::Min { expr: value_expr, value: None });
                    select_plan.push(SelectItemPlan::Agg(idx));
                }
                AggExpr::Max(inner) => {
                    has_agg = true;
                    let value_expr = parse_value_expr(&inner)?;
                    let idx = agg_template.len();
                    agg_template.push(AggState::Max { expr: value_expr, value: None });
                    select_plan.push(SelectItemPlan::Agg(idx));
                }
            }
        }

        if !has_agg && group_by_values.is_empty() {
            return Err(table::Error::Query(QueryError::AggregateRequiresGroupByOrFunctions));
        }

        let having = if let Some(expr) = having {
            let mut cols = SmallVec::<[u16; 8]>::new();
            expr.collect_cols(&mut cols);
            cols.sort_unstable();
            cols.dedup();
            if let Some(err) = validate_columns(&cols, select_plan.len()) {
                return Err(err);
            }
            Some(compile_expr(&expr, None)?)
        } else {
            None
        };

        let scan = scan.with_projection_override(None);
        let scan = scan.compile()?;

        Ok(PreparedAggregate {
            scan,
            group_by: group_by_values,
            select: select_plan,
            agg_template,
            having,
            has_agg,
        })
    }

    /// Execute the aggregate query and invoke `cb` for each output row.
    pub fn for_each<F>(self, scratch: &mut ScanScratch, mut cb: F) -> table::Result<()>
    where
        F: for<'row> FnMut(Row<'row>) -> table::Result<()>,
    {
        let mut prepared = self.compile()?;
        prepared.for_each(scratch, &mut cb)
    }
}

impl<'db> PreparedAggregate<'db> {
    /// Execute the prepared aggregate query and invoke `cb` for each output
    /// row.
    pub fn for_each<F>(&mut self, scratch: &mut ScanScratch, mut cb: F) -> table::Result<()>
    where
        F: for<'row> FnMut(Row<'row>) -> table::Result<()>,
    {
        let mut groups: Vec<GroupState> = Vec::new();
        let mut group_map: HashMap<GroupKeyPtr, usize, FxBuildHasher> = HashMap::default();

        let (scan, group_by, agg_template) = (&mut self.scan, &self.group_by, &self.agg_template);
        let group_len = group_by.len();

        // Aggregate without GROUP BY always has exactly one group (even for empty
        // input).
        if group_len == 0 && self.has_agg {
            groups.push(GroupState::new(Vec::new(), agg_template.clone()));
        }

        scan.for_each_eager(scratch, |_, row| {
            let group_idx = if group_len == 0 {
                0
            } else {
                let mut key_refs: SmallVec<[ValueRef<'_>; 8]> = SmallVec::with_capacity(group_len);
                let mut hasher = FxHasher::default();
                for expr in group_by {
                    let value = eval_value_expr(expr, &row)?;
                    let value_ref = match value {
                        EvalValue::Value(value) => value,
                        EvalValue::Lit(lit) => lit.as_value_ref(),
                    };
                    key_refs.push(value_ref);
                    hash_value_ref(value_ref, &mut hasher);
                }
                let hash = hasher.finish();
                match group_map
                    .raw_entry_mut()
                    .from_hash(hash, |key| key.matches_refs(key_refs.as_slice()))
                {
                    RawEntryMut::Occupied(entry) => *entry.get(),
                    RawEntryMut::Vacant(entry) => {
                        let idx = groups.len();
                        let mut owned_keys = Vec::with_capacity(group_len);
                        for value_ref in key_refs.iter().copied() {
                            owned_keys.push(OwnedValue::from_value_ref(value_ref));
                        }
                        groups.push(GroupState::new(owned_keys, agg_template.clone()));
                        let key_ptr = GroupKeyPtr::new(&groups[idx].keys);
                        entry.insert(key_ptr, idx);
                        idx
                    }
                }
            };

            let group = groups
                .get_mut(group_idx)
                .ok_or(table::Error::Corrupted(Corruption::AggregateGroupIndexOutOfBounds))?;
            for state in &mut group.aggs {
                state.step(&row)?;
            }
            Ok(())
        })?;
        let mut output_slots: Vec<ValueSlot> = Vec::with_capacity(self.select.len());

        for group in &groups {
            output_slots.clear();

            for item in &self.select {
                match item {
                    SelectItemPlan::GroupKey(idx) => {
                        let value = group.keys.get(*idx).ok_or(table::Error::Corrupted(
                            Corruption::AggregateGroupKeyIndexOutOfBounds,
                        ))?;
                        output_slots.push(value.to_slot());
                    }
                    SelectItemPlan::Agg(idx) => {
                        let slot = group
                            .aggs
                            .get(*idx)
                            .ok_or(table::Error::Corrupted(
                                Corruption::AggregateStateIndexOutOfBounds,
                            ))?
                            .finalize_slot();
                        output_slots.push(slot);
                    }
                }
            }

            if let Some(expr) = self.having.as_ref()
                && eval_compiled_expr(expr, output_slots.as_slice(), &[])? != Truth::True
            {
                continue;
            }

            let row = Row::from_raw(output_slots.as_slice(), None);
            cb(row)?;
        }

        Ok(())
    }
}

#[derive(Clone, Debug)]
enum ValueExpr {
    Col(u16),
    Lit(ValueLit),
}

impl PartialEq for ValueExpr {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Col(l), Self::Col(r)) => l == r,
            (Self::Lit(l), Self::Lit(r)) => value_lit_eq(l, r),
            _ => false,
        }
    }
}

impl Eq for ValueExpr {}

fn value_lit_eq(left: &ValueLit, right: &ValueLit) -> bool {
    match (left, right) {
        (ValueLit::Null, ValueLit::Null) => true,
        (ValueLit::Integer(l), ValueLit::Integer(r)) => l == r,
        (ValueLit::Real(l), ValueLit::Real(r)) => {
            if l.is_nan() && r.is_nan() {
                true
            } else {
                l.to_bits() == r.to_bits()
            }
        }
        (ValueLit::Text(l), ValueLit::Text(r)) => l == r,
        _ => false,
    }
}

#[derive(Clone, Debug)]
enum SelectItemPlan {
    GroupKey(usize),
    Agg(usize),
}

#[derive(Clone, Debug)]
enum AggState {
    Count { expr: Option<ValueExpr>, n: i64 },
    Sum { expr: ValueExpr, count: i64, int_sum: i64, real_sum: f64, use_real: bool },
    Avg { expr: ValueExpr, count: i64, sum: f64 },
    Min { expr: ValueExpr, value: Option<OwnedValue> },
    Max { expr: ValueExpr, value: Option<OwnedValue> },
}

impl AggState {
    fn step<'row>(&mut self, row: &Row<'row>) -> table::Result<()> {
        match self {
            Self::Count { expr, n } => {
                if let Some(expr) = expr {
                    let value = eval_value_expr(expr, row)?;
                    if !value.is_null() {
                        *n += 1;
                    }
                } else {
                    *n += 1;
                }
            }
            Self::Sum { expr, count, int_sum, real_sum, use_real } => {
                let value = eval_value_expr(expr, row)?;
                let Some(num) = numeric_value_from_eval(value) else {
                    return Ok(());
                };
                *count += 1;
                match num {
                    NumericValue::Integer(value) => {
                        if *use_real {
                            *real_sum += value as f64;
                        } else if let Some(sum) = int_sum.checked_add(value) {
                            *int_sum = sum;
                        } else {
                            *use_real = true;
                            *real_sum = (*int_sum as f64) + (value as f64);
                        }
                    }
                    NumericValue::Real(value) => {
                        if !*use_real {
                            *use_real = true;
                            *real_sum = *int_sum as f64;
                        }
                        *real_sum += value;
                    }
                }
            }
            Self::Avg { expr, count, sum } => {
                let value = eval_value_expr(expr, row)?;
                let Some(num) = numeric_value_from_eval(value) else {
                    return Ok(());
                };
                *count += 1;
                *sum += num.as_f64();
            }
            Self::Min { expr, value } => {
                let current = eval_value_expr(expr, row)?;
                if current.is_null() {
                    return Ok(());
                }
                match value.as_ref() {
                    None => {
                        *value = Some(OwnedValue::from_eval_value(current));
                    }
                    Some(existing) => {
                        let current_ref = match &current {
                            EvalValue::Value(value) => *value,
                            EvalValue::Lit(lit) => lit.as_value_ref(),
                        };
                        if compare_value_refs(current_ref, existing.as_value_ref())
                            == Ordering::Less
                        {
                            *value = Some(OwnedValue::from_eval_value(current));
                        }
                    }
                }
            }
            Self::Max { expr, value } => {
                let current = eval_value_expr(expr, row)?;
                if current.is_null() {
                    return Ok(());
                }
                match value.as_ref() {
                    None => {
                        *value = Some(OwnedValue::from_eval_value(current));
                    }
                    Some(existing) => {
                        let current_ref = match &current {
                            EvalValue::Value(value) => *value,
                            EvalValue::Lit(lit) => lit.as_value_ref(),
                        };
                        if compare_value_refs(current_ref, existing.as_value_ref())
                            == Ordering::Greater
                        {
                            *value = Some(OwnedValue::from_eval_value(current));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn finalize_slot(&self) -> ValueSlot {
        match self {
            Self::Count { n, .. } => ValueSlot::Integer(*n),
            Self::Sum { count, int_sum, real_sum, use_real, .. } => {
                if *count == 0 {
                    ValueSlot::Null
                } else if *use_real {
                    ValueSlot::Real(*real_sum)
                } else {
                    ValueSlot::Integer(*int_sum)
                }
            }
            Self::Avg { count, sum, .. } => {
                if *count == 0 {
                    ValueSlot::Null
                } else {
                    ValueSlot::Real(*sum / (*count as f64))
                }
            }
            Self::Min { value, .. } | Self::Max { value, .. } => match value.as_ref() {
                None => ValueSlot::Null,
                Some(value) => value.to_slot(),
            },
        }
    }
}

#[derive(Clone, Debug)]
struct GroupState {
    keys: Vec<OwnedValue>,
    aggs: Vec<AggState>,
}

impl GroupState {
    fn new(keys: Vec<OwnedValue>, aggs: Vec<AggState>) -> Self {
        Self { keys, aggs }
    }
}

/// HashMap key for GROUP BY: points at the stable heap buffer of
/// `GroupState.keys`. This avoids storing the key twice (once in the map and
/// once in the group).
#[derive(Clone, Copy, Debug)]
struct GroupKeyPtr {
    ptr: *const OwnedValue,
    len: usize,
}

impl GroupKeyPtr {
    #[inline]
    fn new(values: &[OwnedValue]) -> Self {
        Self { ptr: values.as_ptr(), len: values.len() }
    }

    #[inline]
    fn as_slice(&self) -> &[OwnedValue] {
        // SAFETY: `ptr/len` always reference a `Vec<OwnedValue>` owned by `groups`.
        // `groups` outlives `group_map` for the duration of
        // `PreparedAggregate::for_each`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    #[inline]
    fn matches_refs(&self, refs: &[ValueRef<'_>]) -> bool {
        if self.len != refs.len() {
            return false;
        }
        self.as_slice()
            .iter()
            .zip(refs.iter())
            .all(|(owned, r)| compare_value_refs(owned.as_value_ref(), *r) == Ordering::Equal)
    }
}

impl PartialEq for GroupKeyPtr {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        self.as_slice().iter().zip(other.as_slice().iter()).all(|(left, right)| {
            compare_value_refs(left.as_value_ref(), right.as_value_ref()) == Ordering::Equal
        })
    }
}

impl Eq for GroupKeyPtr {}

impl Hash for GroupKeyPtr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for value in self.as_slice() {
            hash_owned_value(value, state);
        }
    }
}

#[derive(Clone, Debug)]
enum OwnedValue {
    Null,
    Integer(i64),
    Real(f64),
    Text(Vec<u8>),
    Blob(Vec<u8>),
}

impl OwnedValue {
    fn from_value_ref(value: ValueRef<'_>) -> Self {
        match value {
            ValueRef::Null => Self::Null,
            ValueRef::Integer(value) => Self::Integer(value),
            ValueRef::Real(value) => Self::Real(value),
            ValueRef::Text(bytes) => Self::Text(bytes.to_vec()),
            ValueRef::Blob(bytes) => Self::Blob(bytes.to_vec()),
        }
    }

    fn from_lit(lit: &ValueLit) -> Self {
        match lit {
            ValueLit::Null => Self::Null,
            ValueLit::Integer(value) => Self::Integer(*value),
            ValueLit::Real(value) => Self::Real(*value),
            ValueLit::Text(bytes) => Self::Text(bytes.clone()),
        }
    }

    fn from_eval_value(value: EvalValue<'_, '_>) -> Self {
        match value {
            EvalValue::Value(value) => Self::from_value_ref(value),
            EvalValue::Lit(lit) => Self::from_lit(lit),
        }
    }

    fn as_value_ref(&self) -> ValueRef<'_> {
        match self {
            Self::Null => ValueRef::Null,
            Self::Integer(value) => ValueRef::Integer(*value),
            Self::Real(value) => ValueRef::Real(*value),
            Self::Text(bytes) => ValueRef::Text(bytes),
            Self::Blob(bytes) => ValueRef::Blob(bytes),
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

#[derive(Clone, Copy, Debug)]
enum NumericValue {
    Integer(i64),
    Real(f64),
}

impl NumericValue {
    fn as_f64(self) -> f64 {
        match self {
            Self::Integer(value) => value as f64,
            Self::Real(value) => value,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum EvalValue<'row, 'expr> {
    Value(ValueRef<'row>),
    Lit(&'expr ValueLit),
}

impl<'row, 'expr> EvalValue<'row, 'expr> {
    fn is_null(&self) -> bool {
        matches!(self, Self::Value(ValueRef::Null) | Self::Lit(ValueLit::Null))
    }
}

fn numeric_value(value: ValueRef<'_>) -> Option<NumericValue> {
    match value {
        ValueRef::Integer(value) => Some(NumericValue::Integer(value)),
        ValueRef::Real(value) => Some(NumericValue::Real(value)),
        _ => None,
    }
}

fn numeric_value_from_eval(value: EvalValue<'_, '_>) -> Option<NumericValue> {
    match value {
        EvalValue::Value(value) => numeric_value(value),
        EvalValue::Lit(lit) => match lit {
            ValueLit::Integer(value) => Some(NumericValue::Integer(*value)),
            ValueLit::Real(value) => Some(NumericValue::Real(*value)),
            _ => None,
        },
    }
}

fn hash_owned_value<H: Hasher>(value: &OwnedValue, state: &mut H) {
    match value {
        OwnedValue::Null => {
            0u8.hash(state);
        }
        OwnedValue::Integer(value) => {
            1u8.hash(state);
            hash_numeric(*value as f64, state);
        }
        OwnedValue::Real(value) => {
            1u8.hash(state);
            hash_numeric(*value, state);
        }
        OwnedValue::Text(bytes) => {
            2u8.hash(state);
            bytes.hash(state);
        }
        OwnedValue::Blob(bytes) => {
            3u8.hash(state);
            bytes.hash(state);
        }
    }
}

#[inline(always)]
fn hash_value_ref<H: Hasher>(value: ValueRef<'_>, state: &mut H) {
    match value {
        ValueRef::Null => {
            0u8.hash(state);
        }
        ValueRef::Integer(value) => {
            1u8.hash(state);
            hash_numeric(value as f64, state);
        }
        ValueRef::Real(value) => {
            1u8.hash(state);
            hash_numeric(value, state);
        }
        ValueRef::Text(bytes) => {
            2u8.hash(state);
            bytes.hash(state);
        }
        ValueRef::Blob(bytes) => {
            3u8.hash(state);
            bytes.hash(state);
        }
    }
}

fn hash_numeric<H: Hasher>(value: f64, state: &mut H) {
    let normalized = if value.is_nan() {
        f64::NAN
    } else if value == 0.0 {
        0.0
    } else {
        value
    };
    normalized.to_bits().hash(state);
}

fn parse_value_expr(expr: &Expr) -> table::Result<ValueExpr> {
    match expr {
        Expr::Col(col) => Ok(ValueExpr::Col(*col)),
        Expr::Lit(lit) => Ok(ValueExpr::Lit(lit.clone())),
        _ => Err(table::Error::Query(QueryError::AggregateExprOnlyColOrLit)),
    }
}

fn eval_value_expr<'row, 'expr>(
    expr: &'expr ValueExpr,
    row: &Row<'row>,
) -> table::Result<EvalValue<'row, 'expr>> {
    match expr {
        ValueExpr::Col(col) => row_value(row, *col).map(EvalValue::Value),
        ValueExpr::Lit(lit) => Ok(EvalValue::Lit(lit)),
    }
}

fn row_value<'row>(row: &Row<'row>, col: u16) -> table::Result<ValueRef<'row>> {
    let idx = col as usize;
    match row.get(idx) {
        Some(value) => Ok(value),
        None => Err(table::Error::InvalidColumnIndex { col, column_count: row.len() }),
    }
}

#[derive(Clone)]
pub(crate) struct SortKey {
    value: ValueSlot,
    dir: OrderDir,
}

#[derive(Clone)]
struct SortEntry {
    rowid: i64,
    page_id: PageId,
    cell_offset: u16,
    keys: SmallVec<[SortKey; 4]>,
    seq: u64,
}

impl SortEntry {
    fn cmp_keys(&self, other: &Self) -> Ordering {
        for (left, right) in self.keys.iter().zip(other.keys.iter()) {
            let mut ord = compare_value_slots(left.value, right.value);
            if matches!(left.dir, OrderDir::Desc) {
                ord = ord.reverse();
            }
            if ord != Ordering::Equal {
                return ord;
            }
        }
        self.seq.cmp(&other.seq)
    }
}

impl PartialEq for SortEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp_keys(other) == Ordering::Equal
    }
}

impl Eq for SortEntry {}

impl PartialOrd for SortEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(std::cmp::Ord::cmp(self, other))
    }
}

impl Ord for SortEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cmp_keys(other)
    }
}

fn stabilize_sort_key_value(value: ValueSlot, scratch_bytes: &[u8], arena: &Bump) -> ValueSlot {
    match value {
        ValueSlot::Text(span) => match span {
            BytesSpan::Scratch(_) => {
                let bytes =
                    match unsafe { ValueSlot::Text(span).as_value_ref_with_scratch(scratch_bytes) }
                    {
                        ValueRef::Text(bytes) => bytes,
                        _ => &[],
                    };
                let stored = arena.alloc_slice_copy(bytes);
                ValueSlot::Text(BytesSpan::Scratch(RawBytes::from_slice(stored)))
            }
            _ => ValueSlot::Text(span),
        },
        ValueSlot::Blob(span) => match span {
            BytesSpan::Scratch(_) => {
                let bytes =
                    match unsafe { ValueSlot::Blob(span).as_value_ref_with_scratch(scratch_bytes) }
                    {
                        ValueRef::Blob(bytes) => bytes,
                        _ => &[],
                    };
                let stored = arena.alloc_slice_copy(bytes);
                ValueSlot::Blob(BytesSpan::Scratch(RawBytes::from_slice(stored)))
            }
            _ => ValueSlot::Blob(span),
        },
        other => other,
    }
}

fn compare_value_slots(left: ValueSlot, right: ValueSlot) -> Ordering {
    let left = unsafe { left.as_value_ref() };
    let right = unsafe { right.as_value_ref() };
    compare_value_refs(left, right)
}

struct Plan {
    needed_cols: Option<Vec<u16>>,
    proj_map: Option<Vec<usize>>,
    order_val_map: Option<Vec<usize>>,
    referenced_cols: Vec<u16>,
}

fn build_plan(
    projection: Option<&Vec<u16>>,
    pred_cols: &[u16],
    order_by: Option<&[OrderBy]>,
    decode_all: bool,
) -> Plan {
    let decode_all = decode_all || projection.is_none();
    let proj_len = projection.map_or(0, |p| p.len());
    let order_len = order_by.map_or(0, |o| o.len());
    let mut referenced_cols = Vec::with_capacity(pred_cols.len() + proj_len + order_len);
    referenced_cols.extend_from_slice(pred_cols);
    if let Some(proj) = projection {
        referenced_cols.extend(proj.iter().copied());
    }
    if let Some(order_by) = order_by {
        referenced_cols.extend(order_by.iter().map(|order| order.col));
    }
    referenced_cols.sort_unstable();
    referenced_cols.dedup();

    let needed_cols = if decode_all {
        None
    } else {
        let mut needed = Vec::with_capacity(pred_cols.len() + proj_len + order_len);
        needed.extend_from_slice(pred_cols);
        if let Some(proj) = projection {
            needed.extend(proj.iter().copied());
        }
        // Include ORDER BY columns so they're decoded together (avoids
        // re-reading overflow pages for large blobs)
        if let Some(order_by) = order_by {
            needed.extend(order_by.iter().map(|order| order.col));
        }
        needed.sort_unstable();
        needed.dedup();
        Some(needed)
    };

    let proj_map = projection.map(|proj| {
        if let Some(needed) = needed_cols.as_ref() {
            proj.iter()
                .map(|col| {
                    needed
                        .binary_search(col)
                        .unwrap_or_else(|_| panic!("missing projection column in needed set"))
                })
                .collect()
        } else {
            proj.iter().map(|col| *col as usize).collect()
        }
    });

    // Build ORDER BY column → values index mapping (avoids re-reading overflow)
    let order_val_map = order_by.map(|order_by| {
        if let Some(needed) = needed_cols.as_ref() {
            order_by
                .iter()
                .map(|order| {
                    needed
                        .binary_search(&order.col)
                        .unwrap_or_else(|_| panic!("missing ORDER BY column in needed set"))
                })
                .collect()
        } else {
            order_by.iter().map(|order| order.col as usize).collect()
        }
    });

    Plan { needed_cols, proj_map, order_val_map, referenced_cols }
}

fn compile_expr(expr: &Expr, needed_cols: Option<&[u16]>) -> table::Result<CompiledExpr> {
    let col_to_idx = |col: u16| -> table::Result<usize> {
        match needed_cols {
            Some(cols) => cols
                .binary_search(&col)
                .map_err(|_| table::Error::Corrupted(Corruption::PredicateColumnNotDecoded)),
            None => Ok(col as usize),
        }
    };

    let compile_col = |col: u16| -> table::Result<CompiledExpr> {
        let idx = col_to_idx(col)?;
        Ok(CompiledExpr::Col { col, idx })
    };

    // Helper for compiling comparison operators with Col/Lit optimization
    let compile_cmp = |lhs: &Expr,
                       rhs: &Expr,
                       op: CmpOp,
                       make_fallback: fn(Box<CompiledExpr>, Box<CompiledExpr>) -> CompiledExpr|
     -> table::Result<CompiledExpr> {
        match (lhs, rhs) {
            (Expr::Col(c), Expr::Lit(lit)) => {
                Ok(CompiledExpr::CmpColLit { idx: col_to_idx(*c)?, op, lit: lit.clone() })
            }
            (Expr::Lit(lit), Expr::Col(c)) => Ok(CompiledExpr::CmpColLit {
                idx: col_to_idx(*c)?,
                op: swap_cmp(op),
                lit: lit.clone(),
            }),
            _ => Ok(make_fallback(
                Box::new(compile_expr(lhs, needed_cols)?),
                Box::new(compile_expr(rhs, needed_cols)?),
            )),
        }
    };

    Ok(match expr {
        Expr::Col(idx) => compile_col(*idx)?,
        Expr::Lit(lit) => CompiledExpr::Lit(lit.clone()),
        Expr::Eq(lhs, rhs) => compile_cmp(lhs, rhs, CmpOp::Eq, CompiledExpr::Eq)?,
        Expr::Ne(lhs, rhs) => compile_cmp(lhs, rhs, CmpOp::Ne, CompiledExpr::Ne)?,
        Expr::Lt(lhs, rhs) => compile_cmp(lhs, rhs, CmpOp::Lt, CompiledExpr::Lt)?,
        Expr::Le(lhs, rhs) => compile_cmp(lhs, rhs, CmpOp::Le, CompiledExpr::Le)?,
        Expr::Gt(lhs, rhs) => compile_cmp(lhs, rhs, CmpOp::Gt, CompiledExpr::Gt)?,
        Expr::Ge(lhs, rhs) => compile_cmp(lhs, rhs, CmpOp::Ge, CompiledExpr::Ge)?,
        Expr::And(lhs, rhs) => CompiledExpr::And(
            Box::new(compile_expr(lhs, needed_cols)?),
            Box::new(compile_expr(rhs, needed_cols)?),
        ),
        Expr::Or(lhs, rhs) => CompiledExpr::Or(
            Box::new(compile_expr(lhs, needed_cols)?),
            Box::new(compile_expr(rhs, needed_cols)?),
        ),
        Expr::Not(inner) => CompiledExpr::Not(Box::new(compile_expr(inner, needed_cols)?)),
        Expr::IsNull(inner) => CompiledExpr::IsNull(Box::new(compile_expr(inner, needed_cols)?)),
        Expr::IsNotNull(inner) => {
            CompiledExpr::IsNotNull(Box::new(compile_expr(inner, needed_cols)?))
        }
    })
}

#[inline(always)]
fn swap_cmp(op: CmpOp) -> CmpOp {
    match op {
        CmpOp::Eq => CmpOp::Eq,
        CmpOp::Ne => CmpOp::Ne,
        CmpOp::Lt => CmpOp::Gt,
        CmpOp::Le => CmpOp::Ge,
        CmpOp::Gt => CmpOp::Lt,
        CmpOp::Ge => CmpOp::Le,
    }
}

fn validate_columns(referenced: &[u16], column_count: usize) -> Option<table::Error> {
    referenced
        .iter()
        .find(|&&col| col as usize >= column_count)
        .map(|&col| table::Error::InvalidColumnIndex { col, column_count })
}

#[inline]
fn raw_to_ref<'row>(value: ValueSlot) -> ValueRef<'row> {
    // SAFETY: raw values point into the current row payload/overflow buffer and
    // are only materialized for the duration of the row callback.
    unsafe { value.as_value_ref() }
}

#[inline(always)]
fn raw_to_ref_with_scratch<'row>(value: ValueSlot, scratch_bytes: &'row [u8]) -> ValueRef<'row> {
    // SAFETY: raw values point into the current row payload/overflow buffer and
    // are only materialized for the duration of the row callback.
    unsafe { value.as_value_ref_with_scratch(scratch_bytes) }
}

#[inline]
fn value_kind(value: ValueRef<'_>) -> ValueKind {
    match value {
        ValueRef::Null => ValueKind::Null,
        ValueRef::Integer(_) => ValueKind::Integer,
        ValueRef::Real(_) => ValueKind::Real,
        ValueRef::Text(_) => ValueKind::Text,
        ValueRef::Blob(_) => ValueKind::Blob,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Truth {
    True,
    False,
    Null,
}

#[inline]
fn truth_and(left: Truth, right: Truth) -> Truth {
    match (left, right) {
        (Truth::False, _) | (_, Truth::False) => Truth::False,
        (Truth::True, Truth::True) => Truth::True,
        (Truth::True, Truth::Null) | (Truth::Null, Truth::True) | (Truth::Null, Truth::Null) => {
            Truth::Null
        }
    }
}

#[inline]
fn truth_or(left: Truth, right: Truth) -> Truth {
    match (left, right) {
        (Truth::True, _) | (_, Truth::True) => Truth::True,
        (Truth::False, Truth::False) => Truth::False,
        (Truth::False, Truth::Null) | (Truth::Null, Truth::False) | (Truth::Null, Truth::Null) => {
            Truth::Null
        }
    }
}

#[inline]
fn truth_not(value: Truth) -> Truth {
    match value {
        Truth::True => Truth::False,
        Truth::False => Truth::True,
        Truth::Null => Truth::Null,
    }
}

#[derive(Clone, Copy, Debug)]
enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

enum Operand<'row, 'expr> {
    Value(ValueRef<'row>),
    Literal(ValueRef<'expr>),
    Null,
}

struct EvalContext<'row> {
    values: &'row [ValueSlot],
    scratch_bytes: &'row [u8],
}

impl<'row> EvalContext<'row> {
    /// Get value at index. idx is validated during compile phase.
    #[inline(always)]
    fn value_by_idx(&self, idx: usize, col: u16) -> table::Result<ValueRef<'row>> {
        // SAFETY: idx is computed during compile and verified against column count.
        // At runtime the values slice has the same number of elements.
        if idx < self.values.len() {
            let raw = unsafe { *self.values.get_unchecked(idx) };
            Ok(raw_to_ref_with_scratch(raw, self.scratch_bytes))
        } else {
            Err(table::Error::InvalidColumnIndex { col, column_count: self.values.len() })
        }
    }
}

#[inline(always)]
fn eval_compiled_expr(
    expr: &CompiledExpr,
    values: &[ValueSlot],
    scratch_bytes: &[u8],
) -> table::Result<Truth> {
    let ctx = EvalContext { values, scratch_bytes };
    eval_compiled_expr_inner(expr, &ctx)
}

#[inline]
fn eval_compiled_expr_inner(expr: &CompiledExpr, ctx: &EvalContext<'_>) -> table::Result<Truth> {
    match expr {
        CompiledExpr::CmpColLit { idx, op, lit } => {
            let raw = unsafe { *ctx.values.get_unchecked(*idx) };
            let left = raw_to_ref_with_scratch(raw, ctx.scratch_bytes);
            let right = lit.as_value_ref();
            Ok(compare(*op, left, right))
        }
        CompiledExpr::Col { .. } | CompiledExpr::Lit(_) => Ok(Truth::Null),
        CompiledExpr::Eq(lhs, rhs) => eval_cmp(CmpOp::Eq, lhs, rhs, ctx),
        CompiledExpr::Ne(lhs, rhs) => eval_cmp(CmpOp::Ne, lhs, rhs, ctx),
        CompiledExpr::Lt(lhs, rhs) => eval_cmp(CmpOp::Lt, lhs, rhs, ctx),
        CompiledExpr::Le(lhs, rhs) => eval_cmp(CmpOp::Le, lhs, rhs, ctx),
        CompiledExpr::Gt(lhs, rhs) => eval_cmp(CmpOp::Gt, lhs, rhs, ctx),
        CompiledExpr::Ge(lhs, rhs) => eval_cmp(CmpOp::Ge, lhs, rhs, ctx),
        CompiledExpr::And(lhs, rhs) => {
            let left = eval_compiled_expr_inner(lhs, ctx)?;
            if left == Truth::False {
                return Ok(Truth::False);
            }
            let right = eval_compiled_expr_inner(rhs, ctx)?;
            Ok(truth_and(left, right))
        }
        CompiledExpr::Or(lhs, rhs) => {
            let left = eval_compiled_expr_inner(lhs, ctx)?;
            if left == Truth::True {
                return Ok(Truth::True);
            }
            let right = eval_compiled_expr_inner(rhs, ctx)?;
            Ok(truth_or(left, right))
        }
        CompiledExpr::Not(inner) => Ok(truth_not(eval_compiled_expr_inner(inner, ctx)?)),
        CompiledExpr::IsNull(inner) => eval_is_null(inner, ctx),
        CompiledExpr::IsNotNull(inner) => eval_is_not_null(inner, ctx),
    }
}

#[inline]
fn eval_is_null(inner: &CompiledExpr, ctx: &EvalContext<'_>) -> table::Result<Truth> {
    match inner {
        CompiledExpr::Col { col, idx } => match ctx.value_by_idx(*idx, *col)? {
            ValueRef::Null => Ok(Truth::True),
            _ => Ok(Truth::False),
        },
        CompiledExpr::Lit(ValueLit::Null) => Ok(Truth::True),
        CompiledExpr::Lit(_) => Ok(Truth::False),
        _ => Ok(if matches!(eval_compiled_expr_inner(inner, ctx)?, Truth::Null) {
            Truth::True
        } else {
            Truth::False
        }),
    }
}

#[inline]
fn eval_is_not_null(inner: &CompiledExpr, ctx: &EvalContext<'_>) -> table::Result<Truth> {
    match eval_is_null(inner, ctx)? {
        Truth::True => Ok(Truth::False),
        Truth::False => Ok(Truth::True),
        Truth::Null => Ok(Truth::Null),
    }
}

#[inline]
fn eval_cmp(
    op: CmpOp,
    lhs: &CompiledExpr,
    rhs: &CompiledExpr,
    ctx: &EvalContext<'_>,
) -> table::Result<Truth> {
    let left = eval_operand(lhs, ctx)?;
    let right = eval_operand(rhs, ctx)?;

    Ok(match (left, right) {
        (Operand::Null, _) | (_, Operand::Null) => Truth::Null,
        (Operand::Value(l), Operand::Value(r)) => compare(op, l, r),
        (Operand::Value(l), Operand::Literal(r)) => compare(op, l, r),
        (Operand::Literal(l), Operand::Value(r)) => compare(op, l, r),
        (Operand::Literal(l), Operand::Literal(r)) => compare(op, l, r),
    })
}

#[inline]
fn eval_operand<'row, 'expr>(
    expr: &'expr CompiledExpr,
    ctx: &EvalContext<'row>,
) -> table::Result<Operand<'row, 'expr>> {
    match expr {
        CompiledExpr::Col { col, idx } => ctx.value_by_idx(*idx, *col).map(Operand::Value),
        CompiledExpr::Lit(lit) => Ok(Operand::Literal(lit.as_value_ref())),
        _ => Ok(Operand::Null),
    }
}

impl ValueLit {
    fn as_value_ref(&self) -> ValueRef<'_> {
        match self {
            ValueLit::Null => ValueRef::Null,
            ValueLit::Integer(value) => ValueRef::Integer(*value),
            ValueLit::Real(value) => ValueRef::Real(*value),
            ValueLit::Text(bytes) => ValueRef::Text(bytes.as_slice()),
        }
    }
}

#[inline(always)]
fn compare<'a, 'b>(op: CmpOp, left: ValueRef<'a>, right: ValueRef<'b>) -> Truth {
    match (left, right) {
        (ValueRef::Null, _) | (_, ValueRef::Null) => Truth::Null,
        (ValueRef::Integer(l), ValueRef::Integer(r)) => cmp_order(op, l.cmp(&r)),
        (ValueRef::Integer(l), ValueRef::Real(r)) => cmp_f64(op, l as f64, r),
        (ValueRef::Real(l), ValueRef::Integer(r)) => cmp_f64(op, l, r as f64),
        (ValueRef::Real(l), ValueRef::Real(r)) => cmp_f64(op, l, r),
        (ValueRef::Text(l), ValueRef::Text(r)) => cmp_order(op, l.cmp(r)),
        (ValueRef::Blob(l), ValueRef::Blob(r)) => cmp_order(op, l.cmp(r)),
        _ => Truth::False,
    }
}

#[inline(always)]
fn cmp_f64(op: CmpOp, left: f64, right: f64) -> Truth {
    match left.partial_cmp(&right) {
        Some(order) => cmp_order(op, order),
        None => Truth::False,
    }
}

#[inline(always)]
fn cmp_order(op: CmpOp, order: std::cmp::Ordering) -> Truth {
    let matches = match op {
        CmpOp::Eq => order == std::cmp::Ordering::Equal,
        CmpOp::Ne => order != std::cmp::Ordering::Equal,
        CmpOp::Lt => order == std::cmp::Ordering::Less,
        CmpOp::Le => order != std::cmp::Ordering::Greater,
        CmpOp::Gt => order == std::cmp::Ordering::Greater,
        CmpOp::Ge => order != std::cmp::Ordering::Less,
    };

    if matches { Truth::True } else { Truth::False }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(expr: &Expr) -> Vec<u16> {
        let mut cols = SmallVec::<[u16; 8]>::new();
        expr.collect_cols(&mut cols);
        cols.sort_unstable();
        cols.dedup();
        cols.to_vec()
    }

    #[test]
    fn collects_predicate_columns() {
        let expr = col(2).gt(lit_i64(10)).and(col(1).eq(lit_bytes(b"alice"))).or(col(3).is_null());
        assert_eq!(collect(&expr), vec![1, 2, 3]);
    }

    #[test]
    fn truth_table_and_or_not() {
        assert_eq!(truth_and(Truth::True, Truth::Null), Truth::Null);
        assert_eq!(truth_and(Truth::False, Truth::Null), Truth::False);
        assert_eq!(truth_or(Truth::True, Truth::Null), Truth::True);
        assert_eq!(truth_or(Truth::False, Truth::Null), Truth::Null);
        assert_eq!(truth_not(Truth::Null), Truth::Null);
    }

    #[test]
    fn projection_mapping_remaps_columns() {
        let projection = vec![3u16, 1u16];
        let plan = build_plan(Some(&projection), &[], None, false);
        assert_eq!(plan.needed_cols, Some(vec![1, 3]));
        assert_eq!(plan.proj_map, Some(vec![1, 0]));
    }

    #[test]
    fn projection_skips_unneeded_decoding() {
        let payload =
            build_record(&[ValueRef::Integer(1), ValueRef::Integer(2), ValueRef::Integer(3)]);
        let needed = vec![0u16, 2u16];
        let mut decoded = Vec::with_capacity(needed.len());
        let mut bytes = Vec::new();
        let mut serials = Vec::with_capacity(needed.len());
        let mut offsets = Vec::with_capacity(needed.len());
        let _ = table::decode_record_project_into(
            table::PayloadRef::Inline(&payload),
            Some(&needed),
            &mut decoded,
            &mut bytes,
            &mut serials,
            &mut offsets,
        )
        .expect("decode");
        assert_eq!(decoded.len(), 2);
    }

    fn build_record(values: &[ValueRef<'_>]) -> Vec<u8> {
        let mut header = Vec::new();
        for value in values {
            let serial = match value {
                ValueRef::Null => 0u64,
                ValueRef::Integer(v) => match v {
                    -128..=127 => 1,
                    -32768..=32767 => 2,
                    -8_388_608..=8_388_607 => 3,
                    -2_147_483_648..=2_147_483_647 => 4,
                    _ => 6,
                },
                ValueRef::Real(_) => 7,
                ValueRef::Text(bytes) => 13 + (bytes.len() as u64) * 2,
                ValueRef::Blob(bytes) => 12 + (bytes.len() as u64) * 2,
            };
            push_varint(&mut header, serial);
        }

        let serial_len = header.len();
        let mut header_len = serial_len + varint_len(serial_len as u64);
        loop {
            let new_len = serial_len + varint_len(header_len as u64);
            if new_len == header_len {
                break;
            }
            header_len = new_len;
        }
        let mut record = Vec::new();
        push_varint(&mut record, header_len as u64);
        record.extend_from_slice(&header);

        for value in values {
            match value {
                ValueRef::Null => {}
                ValueRef::Integer(v) => {
                    let bytes = v.to_be_bytes();
                    let len = match v {
                        -128..=127 => 1,
                        -32768..=32767 => 2,
                        -8_388_608..=8_388_607 => 3,
                        -2_147_483_648..=2_147_483_647 => 4,
                        _ => 8,
                    };
                    record.extend_from_slice(&bytes[8 - len..]);
                }
                ValueRef::Real(v) => record.extend_from_slice(&v.to_bits().to_be_bytes()),
                ValueRef::Text(bytes) | ValueRef::Blob(bytes) => record.extend_from_slice(bytes),
            }
        }

        record
    }

    fn push_varint(out: &mut Vec<u8>, mut value: u64) {
        let mut buf = [0u8; 9];
        let mut idx = 8;
        buf[8] = (value & 0x7F) as u8;
        value >>= 7;

        while value > 0 {
            idx -= 1;
            buf[idx] = ((value & 0x7F) as u8) | 0x80;
            value >>= 7;
        }

        out.extend_from_slice(&buf[idx..=8]);
    }

    fn varint_len(mut value: u64) -> usize {
        if value <= 0x7F {
            return 1;
        }
        let mut len = 1;
        value >>= 7;
        while value > 0 {
            len += 1;
            value >>= 7;
        }
        len
    }
}
