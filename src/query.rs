use std::cell::OnceCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use bumpalo::Bump;
use smallvec::SmallVec;

use crate::pager::{PageId, Pager};
use crate::table::{self, BytesSpan, RawBytes, ValueRef, ValueSlot};

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

    fn collect_cols(&self, out: &mut Vec<u16>) {
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
#[derive(Debug, Default)]
pub struct ScanScratch {
    stack: Vec<PageId>,
    values: Vec<ValueSlot>,
    bytes: Vec<u8>,
    serials: Vec<u64>,
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
        }
    }

    pub(crate) fn split_mut(
        &mut self,
    ) -> (&mut Vec<ValueSlot>, &mut Vec<u8>, &mut Vec<u64>, &mut Vec<PageId>) {
        (&mut self.values, &mut self.bytes, &mut self.serials, &mut self.stack)
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
    pub fn len(&self) -> usize {
        self.proj_map.map_or(self.values.len(), |map| map.len())
    }

    /// Returns true when there are no columns.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return a value reference by column index.
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
                expected: "INTEGER",
                got: value_kind(other),
            }),
            None => {
                Err(table::Error::TypeMismatch { col: i, expected: "INTEGER", got: "<missing>" })
            }
        }
    }

    /// Return an `f64` value or a type mismatch error.
    pub fn get_f64(&self, i: usize) -> table::Result<f64> {
        match self.get(i) {
            Some(ValueRef::Real(value)) => Ok(value),
            Some(other) => {
                Err(table::Error::TypeMismatch { col: i, expected: "REAL", got: value_kind(other) })
            }
            None => Err(table::Error::TypeMismatch { col: i, expected: "REAL", got: "<missing>" }),
        }
    }

    /// Return a UTF-8 string or a type mismatch error.
    pub fn get_text(&self, i: usize) -> table::Result<&'row str> {
        match self.get(i) {
            Some(ValueRef::Text(bytes)) => Ok(std::str::from_utf8(bytes)?),
            Some(other) => {
                Err(table::Error::TypeMismatch { col: i, expected: "TEXT", got: value_kind(other) })
            }
            None => Err(table::Error::TypeMismatch { col: i, expected: "TEXT", got: "<missing>" }),
        }
    }

    /// Return text/blob bytes or a type mismatch error.
    pub fn get_bytes(&self, i: usize) -> table::Result<&'row [u8]> {
        match self.get(i) {
            Some(ValueRef::Text(bytes)) => Ok(bytes),
            Some(ValueRef::Blob(bytes)) => Ok(bytes),
            Some(other) => Err(table::Error::TypeMismatch {
                col: i,
                expected: "BYTES",
                got: value_kind(other),
            }),
            None => Err(table::Error::TypeMismatch { col: i, expected: "BYTES", got: "<missing>" }),
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

    pub(crate) fn projection(&self) -> Option<&[u16]> {
        self.projection.as_deref()
    }

    pub(crate) fn has_order_by(&self) -> bool {
        matches!(self.order_by.as_deref(), Some(cols) if !cols.is_empty())
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

        let mut pred_cols = Vec::new();
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

        let Plan { needed_cols, proj_map, referenced_cols } = plan;
        let column_count_hint = OnceCell::new();
        if let Some(count) = col_count_hint {
            let _ = column_count_hint.set(count);
        }

        Ok(PreparedScan {
            pager,
            root,
            needed_cols: needed_cols.map(|cols| cols.into_boxed_slice()),
            proj_map: proj_map.map(|cols| cols.into_boxed_slice()),
            referenced_cols: referenced_cols.into_boxed_slice(),
            compiled_expr,
            filter_fn,
            order_by: order_by.map(|cols| cols.into_boxed_slice()),
            limit,
            column_count_hint,
        })
    }

    /// Execute the scan and invoke `cb` for each row.
    pub fn for_each<F>(self, scratch: &mut ScanScratch, mut cb: F) -> table::Result<()>
    where
        F: for<'row> FnMut(i64, Row<'row>) -> table::Result<()>,
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

    pub(crate) fn eval_payload<'row>(
        &'row mut self,
        payload: table::PayloadRef<'row>,
        values: &'row mut Vec<ValueSlot>,
        bytes: &'row mut Vec<u8>,
        serials: &'row mut Vec<u64>,
    ) -> table::Result<Option<Row<'row>>> {
        self.eval_payload_with_filters(payload, values, bytes, serials, true)
    }

    pub(crate) fn eval_payload_with_filters<'row>(
        &'row mut self,
        payload: table::PayloadRef<'row>,
        values: &'row mut Vec<ValueSlot>,
        bytes: &'row mut Vec<u8>,
        serials: &'row mut Vec<u64>,
        apply_filters: bool,
    ) -> table::Result<Option<Row<'row>>> {
        let needed_cols = self.needed_cols.as_deref();
        let count =
            table::decode_record_project_into(payload, needed_cols, values, bytes, serials)?;

        if let Some(expected) = self.column_count_hint.get().copied() {
            if expected != count {
                return Err(table::Error::Corrupted(
                    "row column count does not match table schema",
                ));
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

    pub(crate) fn eval_index_payload_with_map<'row>(
        &'row mut self,
        payload: table::PayloadRef<'row>,
        needed_map: &[(u16, usize)],
        values: &'row mut Vec<ValueSlot>,
        bytes: &'row mut Vec<u8>,
        serials: &'row mut Vec<u64>,
        apply_filters: bool,
    ) -> table::Result<Option<Row<'row>>> {
        let Some(needed_cols) = self.needed_cols.as_deref() else {
            return Err(table::Error::Corrupted("covering index requires column projection"));
        };
        if needed_cols.len() != needed_map.len() {
            return Err(table::Error::Corrupted(
                "covering index column map does not match needed columns",
            ));
        }

        let _count =
            table::decode_record_project_into_mapped(payload, needed_map, values, bytes, serials)?;

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

    /// Execute the prepared scan and invoke `cb` for each row.
    pub fn for_each<F>(&mut self, scratch: &mut ScanScratch, mut cb: F) -> table::Result<()>
    where
        F: for<'row> FnMut(i64, Row<'row>) -> table::Result<()>,
    {
        if self.limit == Some(0) {
            return Ok(());
        }

        if let Some(order_by) = self.order_by.take() {
            if order_by.is_empty() {
                self.order_by = Some(order_by);
            } else {
                let result = self.for_each_ordered(scratch, order_by.as_ref(), &mut cb);
                self.order_by = Some(order_by);
                return result;
            }
        }

        let pager = self.pager;
        let root = self.root;
        let limit = self.limit;
        let mut seen = 0usize;

        let (values, bytes, serials, btree_stack) = scratch.split_mut();

        table::scan_table_cells_with_scratch_and_stack_until(pager, root, btree_stack, |cell| {
            let rowid = cell.rowid();
            let Some(row) = self.eval_payload(cell.payload(), values, bytes, serials)? else {
                return Ok(None);
            };

            cb(rowid, row)?;
            seen += 1;

            if let Some(limit) = limit
                && seen >= limit
            {
                return Ok(Some(()));
            }

            Ok(None)
        })?;

        Ok(())
    }

    fn for_each_ordered<F>(
        &mut self,
        scratch: &mut ScanScratch,
        order_by: &[OrderBy],
        cb: &mut F,
    ) -> table::Result<()>
    where
        F: for<'row> FnMut(i64, Row<'row>) -> table::Result<()>,
    {
        let pager = self.pager;
        let root = self.root;
        let limit = self.limit;

        let (values, bytes, serials, btree_stack) = scratch.split_mut();
        let mut order_bytes = Vec::new();
        let key_arena = Bump::new();
        let mut seq = 0u64;

        let mut heap = BinaryHeap::new();
        let mut entries = Vec::new();

        table::scan_table_cells_with_scratch_and_stack(pager, root, btree_stack, |cell| {
            let rowid = cell.rowid();
            let page_id = cell.page_id();
            let cell_offset = cell.cell_offset();
            let payload = cell.payload();
            let Some(_row) = self.eval_payload(payload, values, bytes, serials)? else {
                return Ok(());
            };

            let mut keys = SmallVec::<[SortKey; 4]>::with_capacity(order_by.len());
            for order in order_by {
                let Some(value) =
                    table::decode_record_column(payload, order.col, &mut order_bytes)?
                else {
                    return Err(table::Error::Corrupted("ORDER BY column missing"));
                };
                let value = stabilize_sort_key_value(value, order_bytes.as_slice(), &key_arena);
                keys.push(SortKey { value, dir: order.dir });
            }

            let entry = SortEntry { rowid, page_id, cell_offset, keys, seq };
            seq = seq.wrapping_add(1);

            if let Some(limit) = limit {
                if heap.len() < limit {
                    heap.push(entry);
                } else if let Some(top) = heap.peek()
                    && entry.cmp(top) == Ordering::Less
                {
                    heap.pop();
                    heap.push(entry);
                }
            } else {
                entries.push(entry);
            }

            Ok(())
        })?;

        let mut rows = if limit.is_some() { heap.into_vec() } else { entries };
        rows.sort();

        for entry in rows {
            let cell =
                table::read_table_cell_ref_from_bytes(pager, entry.page_id, entry.cell_offset)?;
            if let Some(row) =
                self.eval_payload_with_filters(cell.payload(), values, bytes, serials, false)?
            {
                cb(entry.rowid, row)?;
            }
        }

        Ok(())
    }
}

#[derive(Clone)]
struct SortKey {
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

fn compare_value_refs(left: ValueRef<'_>, right: ValueRef<'_>) -> Ordering {
    let rank = |value: ValueRef<'_>| match value {
        ValueRef::Null => 0u8,
        ValueRef::Integer(_) | ValueRef::Real(_) => 1u8,
        ValueRef::Text(_) => 2u8,
        ValueRef::Blob(_) => 3u8,
    };

    let left_rank = rank(left);
    let right_rank = rank(right);
    if left_rank != right_rank {
        return left_rank.cmp(&right_rank);
    }

    match (left, right) {
        (ValueRef::Null, ValueRef::Null) => Ordering::Equal,
        (ValueRef::Integer(l), ValueRef::Integer(r)) => l.cmp(&r),
        (ValueRef::Integer(l), ValueRef::Real(r)) => cmp_f64_total(l as f64, r),
        (ValueRef::Real(l), ValueRef::Integer(r)) => cmp_f64_total(l, r as f64),
        (ValueRef::Real(l), ValueRef::Real(r)) => cmp_f64_total(l, r),
        (ValueRef::Text(l), ValueRef::Text(r)) => l.cmp(r),
        (ValueRef::Blob(l), ValueRef::Blob(r)) => l.cmp(r),
        _ => Ordering::Equal,
    }
}

fn cmp_f64_total(left: f64, right: f64) -> Ordering {
    match (left.is_nan(), right.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => left.partial_cmp(&right).unwrap_or(Ordering::Equal),
    }
}

struct Plan {
    needed_cols: Option<Vec<u16>>,
    proj_map: Option<Vec<usize>>,
    referenced_cols: Vec<u16>,
}

fn build_plan(
    projection: Option<&Vec<u16>>,
    pred_cols: &[u16],
    order_by: Option<&[OrderBy]>,
    decode_all: bool,
) -> Plan {
    let decode_all = decode_all || projection.is_none();
    let mut referenced_cols = pred_cols.to_vec();
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
        let mut needed = pred_cols.to_vec();
        if let Some(proj) = projection {
            needed.extend(proj.iter().copied());
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

    Plan { needed_cols, proj_map, referenced_cols }
}

fn compile_expr(expr: &Expr, needed_cols: Option<&[u16]>) -> table::Result<CompiledExpr> {
    let col_to_idx = |col: u16| -> table::Result<usize> {
        Ok(match needed_cols {
            Some(cols) => cols
                .binary_search(&col)
                .map_err(|_| table::Error::Corrupted("predicate column not decoded"))?,
            None => col as usize,
        })
    };

    let compile_col = |col: u16| -> table::Result<CompiledExpr> {
        let idx = col_to_idx(col)?;
        Ok(CompiledExpr::Col { col, idx })
    };

    Ok(match expr {
        Expr::Col(idx) => compile_col(*idx)?,
        Expr::Lit(lit) => CompiledExpr::Lit(lit.clone()),
        Expr::Eq(lhs, rhs) => match (&**lhs, &**rhs) {
            (Expr::Col(c), Expr::Lit(lit)) => {
                CompiledExpr::CmpColLit { idx: col_to_idx(*c)?, op: CmpOp::Eq, lit: lit.clone() }
            }
            (Expr::Lit(lit), Expr::Col(c)) => {
                CompiledExpr::CmpColLit { idx: col_to_idx(*c)?, op: CmpOp::Eq, lit: lit.clone() }
            }
            _ => CompiledExpr::Eq(
                Box::new(compile_expr(lhs, needed_cols)?),
                Box::new(compile_expr(rhs, needed_cols)?),
            ),
        },
        Expr::Ne(lhs, rhs) => match (&**lhs, &**rhs) {
            (Expr::Col(c), Expr::Lit(lit)) => {
                CompiledExpr::CmpColLit { idx: col_to_idx(*c)?, op: CmpOp::Ne, lit: lit.clone() }
            }
            (Expr::Lit(lit), Expr::Col(c)) => CompiledExpr::CmpColLit {
                idx: col_to_idx(*c)?,
                op: swap_cmp(CmpOp::Ne),
                lit: lit.clone(),
            },
            _ => CompiledExpr::Ne(
                Box::new(compile_expr(lhs, needed_cols)?),
                Box::new(compile_expr(rhs, needed_cols)?),
            ),
        },
        Expr::Lt(lhs, rhs) => match (&**lhs, &**rhs) {
            (Expr::Col(c), Expr::Lit(lit)) => {
                CompiledExpr::CmpColLit { idx: col_to_idx(*c)?, op: CmpOp::Lt, lit: lit.clone() }
            }
            (Expr::Lit(lit), Expr::Col(c)) => CompiledExpr::CmpColLit {
                idx: col_to_idx(*c)?,
                op: swap_cmp(CmpOp::Lt),
                lit: lit.clone(),
            },
            _ => CompiledExpr::Lt(
                Box::new(compile_expr(lhs, needed_cols)?),
                Box::new(compile_expr(rhs, needed_cols)?),
            ),
        },
        Expr::Le(lhs, rhs) => match (&**lhs, &**rhs) {
            (Expr::Col(c), Expr::Lit(lit)) => {
                CompiledExpr::CmpColLit { idx: col_to_idx(*c)?, op: CmpOp::Le, lit: lit.clone() }
            }
            (Expr::Lit(lit), Expr::Col(c)) => CompiledExpr::CmpColLit {
                idx: col_to_idx(*c)?,
                op: swap_cmp(CmpOp::Le),
                lit: lit.clone(),
            },
            _ => CompiledExpr::Le(
                Box::new(compile_expr(lhs, needed_cols)?),
                Box::new(compile_expr(rhs, needed_cols)?),
            ),
        },
        Expr::Gt(lhs, rhs) => match (&**lhs, &**rhs) {
            (Expr::Col(c), Expr::Lit(lit)) => {
                CompiledExpr::CmpColLit { idx: col_to_idx(*c)?, op: CmpOp::Gt, lit: lit.clone() }
            }
            (Expr::Lit(lit), Expr::Col(c)) => CompiledExpr::CmpColLit {
                idx: col_to_idx(*c)?,
                op: swap_cmp(CmpOp::Gt),
                lit: lit.clone(),
            },
            _ => CompiledExpr::Gt(
                Box::new(compile_expr(lhs, needed_cols)?),
                Box::new(compile_expr(rhs, needed_cols)?),
            ),
        },
        Expr::Ge(lhs, rhs) => match (&**lhs, &**rhs) {
            (Expr::Col(c), Expr::Lit(lit)) => {
                CompiledExpr::CmpColLit { idx: col_to_idx(*c)?, op: CmpOp::Ge, lit: lit.clone() }
            }
            (Expr::Lit(lit), Expr::Col(c)) => CompiledExpr::CmpColLit {
                idx: col_to_idx(*c)?,
                op: swap_cmp(CmpOp::Ge),
                lit: lit.clone(),
            },
            _ => CompiledExpr::Ge(
                Box::new(compile_expr(lhs, needed_cols)?),
                Box::new(compile_expr(rhs, needed_cols)?),
            ),
        },
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
    for col in referenced {
        if *col as usize >= column_count {
            return Some(table::Error::InvalidColumnIndex { col: *col, column_count });
        }
    }
    None
}

fn raw_to_ref<'row>(value: ValueSlot) -> ValueRef<'row> {
    // SAFETY: raw values point into the current row payload/overflow buffer and
    // are only materialized for the duration of the row callback.
    unsafe { value.as_value_ref() }
}

fn raw_to_ref_with_scratch<'row>(value: ValueSlot, scratch_bytes: &'row [u8]) -> ValueRef<'row> {
    // SAFETY: raw values point into the current row payload/overflow buffer and
    // are only materialized for the duration of the row callback.
    unsafe { value.as_value_ref_with_scratch(scratch_bytes) }
}

fn value_kind(value: ValueRef<'_>) -> &'static str {
    match value {
        ValueRef::Null => "NULL",
        ValueRef::Integer(_) => "INTEGER",
        ValueRef::Real(_) => "REAL",
        ValueRef::Text(_) => "TEXT",
        ValueRef::Blob(_) => "BLOB",
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Truth {
    True,
    False,
    Null,
}

fn truth_and(left: Truth, right: Truth) -> Truth {
    match (left, right) {
        (Truth::False, _) | (_, Truth::False) => Truth::False,
        (Truth::True, Truth::True) => Truth::True,
        (Truth::True, Truth::Null) | (Truth::Null, Truth::True) | (Truth::Null, Truth::Null) => {
            Truth::Null
        }
    }
}

fn truth_or(left: Truth, right: Truth) -> Truth {
    match (left, right) {
        (Truth::True, _) | (_, Truth::True) => Truth::True,
        (Truth::False, Truth::False) => Truth::False,
        (Truth::False, Truth::Null) | (Truth::Null, Truth::False) | (Truth::Null, Truth::Null) => {
            Truth::Null
        }
    }
}

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
    fn value_by_idx(&self, idx: usize, col: u16) -> table::Result<ValueRef<'row>> {
        self.values
            .get(idx)
            .copied()
            .map(|raw| raw_to_ref_with_scratch(raw, self.scratch_bytes))
            .ok_or(table::Error::InvalidColumnIndex { col, column_count: self.values.len() })
    }
}

fn eval_compiled_expr(
    expr: &CompiledExpr,
    values: &[ValueSlot],
    scratch_bytes: &[u8],
) -> table::Result<Truth> {
    let ctx = EvalContext { values, scratch_bytes };
    eval_compiled_expr_inner(expr, &ctx)
}

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

fn eval_is_not_null(inner: &CompiledExpr, ctx: &EvalContext<'_>) -> table::Result<Truth> {
    match eval_is_null(inner, ctx)? {
        Truth::True => Ok(Truth::False),
        Truth::False => Ok(Truth::True),
        Truth::Null => Ok(Truth::Null),
    }
}

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

fn cmp_f64(op: CmpOp, left: f64, right: f64) -> Truth {
    match left.partial_cmp(&right) {
        Some(order) => cmp_order(op, order),
        None => Truth::False,
    }
}

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
        let mut cols = Vec::new();
        expr.collect_cols(&mut cols);
        cols.sort_unstable();
        cols.dedup();
        cols
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
        let _ = table::decode_record_project_into(
            table::PayloadRef::Inline(&payload),
            Some(&needed),
            &mut decoded,
            &mut bytes,
            &mut serials,
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
