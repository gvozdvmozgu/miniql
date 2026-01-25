use std::{fmt, ptr, str};

use crate::decoder::Decoder;
use crate::join::JoinError;
use crate::pager::{PageId, PageRef, Pager};

pub type Result<T> = std::result::Result<T, Error>;

const MAX_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug)]
pub enum Error {
    Pager(crate::pager::Error),
    UnsupportedPageType(u8),
    UnsupportedSerialType(u64),
    Corrupted(&'static str),
    InvalidColumnIndex { col: u16, column_count: usize },
    TypeMismatch { col: usize, expected: &'static str, got: &'static str },
    Utf8(str::Utf8Error),
    TableNotFound(String),
    PayloadTooLarge(usize),
    OverflowChainTruncated,
    OverflowLoopDetected,
    Join(JoinError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pager(err) => write!(f, "{err}"),
            Self::UnsupportedPageType(kind) => write!(f, "Unsupported page type: 0x{kind:02X}"),
            Self::UnsupportedSerialType(serial) => {
                write!(f, "Unsupported record serial type: {serial}")
            }
            Self::Corrupted(msg) => write!(f, "Corrupted table page: {msg}"),
            Self::InvalidColumnIndex { col, column_count } => {
                write!(f, "Invalid column index {col} (column count {column_count})")
            }
            Self::TypeMismatch { col, expected, got } => {
                write!(f, "Type mismatch at column {col}: expected {expected}, got {got}")
            }
            Self::Utf8(err) => write!(f, "{err}"),
            Self::TableNotFound(name) => {
                write!(f, "Table '{name}' not found in sqlite_schema")
            }
            Self::PayloadTooLarge(size) => write!(f, "Payload too large: {size} bytes"),
            Self::OverflowChainTruncated => f.write_str("Overflow chain is truncated"),
            Self::OverflowLoopDetected => f.write_str("Overflow chain contains a loop"),
            Self::Join(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Pager(err) => Some(err),
            Self::Utf8(err) => Some(err),
            Self::Join(err) => Some(err),
            _ => None,
        }
    }
}

impl From<crate::pager::Error> for Error {
    fn from(err: crate::pager::Error) -> Self {
        Self::Pager(err)
    }
}

impl From<str::Utf8Error> for Error {
    fn from(err: str::Utf8Error) -> Self {
        Self::Utf8(err)
    }
}

impl From<JoinError> for Error {
    fn from(err: JoinError) -> Self {
        Self::Join(err)
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    Null,
    Integer(i64),
    Real(f64),
    Text(String),
    Blob(Vec<u8>),
}

impl Value {
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text.as_str()),
            _ => None,
        }
    }

    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Self::Integer(value) => Some(*value),
            _ => None,
        }
    }

    fn display_blob(bytes: &[u8], f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("x'")?;
        for byte in bytes {
            write!(f, "{byte:02x}")?;
        }
        f.write_str("'")
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => f.write_str("NULL"),
            Self::Integer(value) => write!(f, "{value}"),
            Self::Real(value) => write!(f, "{value}"),
            Self::Text(value) => f.write_str(value),
            Self::Blob(bytes) => Self::display_blob(bytes, f),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ValueRef<'row> {
    Null,
    Integer(i64),
    Real(f64),
    TextBytes(&'row [u8]),
    Blob(&'row [u8]),
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RawBytes {
    ptr: *const u8,
    len: usize,
}

impl RawBytes {
    #[inline]
    fn from_slice(bytes: &[u8]) -> Self {
        Self { ptr: bytes.as_ptr(), len: bytes.len() }
    }

    #[inline]
    unsafe fn as_slice<'row>(self) -> &'row [u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ValueRefRaw {
    Null,
    Integer(i64),
    Real(f64),
    TextBytes(RawBytes),
    Blob(RawBytes),
}

#[derive(Clone, Copy)]
pub enum RecordPayload<'row> {
    Inline(&'row [u8]),
    Overflow(OverflowPayload<'row>),
}

#[derive(Clone, Copy)]
pub struct OverflowPayload<'row> {
    pager: &'row Pager,
    total_len: usize,
    local: &'row [u8],
    first_overflow: u32,
}

impl<'row> OverflowPayload<'row> {
    pub(crate) fn new(
        pager: &'row Pager,
        total_len: usize,
        local: &'row [u8],
        first_overflow: u32,
    ) -> Self {
        Self { pager, total_len, local, first_overflow }
    }
}

impl<'row> RecordPayload<'row> {
    pub fn to_vec(&self) -> Result<Vec<u8>> {
        match self {
            RecordPayload::Inline(bytes) => Ok(bytes.to_vec()),
            RecordPayload::Overflow(payload) => {
                let mut out = Vec::with_capacity(payload.total_len);
                assemble_overflow_payload(
                    payload.pager,
                    payload.total_len,
                    payload.local.len(),
                    payload.first_overflow,
                    payload.local,
                    &mut out,
                )?;
                Ok(out)
            }
        }
    }
}

impl ValueRefRaw {
    #[inline]
    pub(crate) unsafe fn as_value_ref<'row>(self) -> ValueRef<'row> {
        match self {
            Self::Null => ValueRef::Null,
            Self::Integer(value) => ValueRef::Integer(value),
            Self::Real(value) => ValueRef::Real(value),
            Self::TextBytes(bytes) => ValueRef::TextBytes(unsafe { bytes.as_slice() }),
            Self::Blob(bytes) => ValueRef::Blob(unsafe { bytes.as_slice() }),
        }
    }
}

fn raw_to_ref<'row>(value: ValueRefRaw) -> ValueRef<'row> {
    // SAFETY: raw values point into the current row payload/overflow buffer and
    // are only materialized for the duration of the row callback.
    unsafe { value.as_value_ref() }
}

impl<'row> ValueRef<'row> {
    pub fn as_text(&self) -> Option<&'row str> {
        match self {
            Self::TextBytes(bytes) => str::from_utf8(bytes).ok(),
            _ => None,
        }
    }

    pub fn text_bytes(&self) -> Option<&'row [u8]> {
        match self {
            Self::TextBytes(bytes) => Some(*bytes),
            _ => None,
        }
    }

    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Self::Integer(value) => Some(*value),
            _ => None,
        }
    }
}

impl fmt::Display for ValueRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => f.write_str("NULL"),
            Self::Integer(value) => write!(f, "{value}"),
            Self::Real(value) => write!(f, "{value}"),
            Self::TextBytes(bytes) => match str::from_utf8(bytes) {
                Ok(value) => f.write_str(value),
                Err(_) => f.write_str("<invalid utf8>"),
            },
            Self::Blob(bytes) => Value::display_blob(bytes, f),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RowView<'row> {
    values: &'row [ValueRefRaw],
}

impl<'row> RowView<'row> {
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn get(&self, i: usize) -> Option<ValueRef<'row>> {
        self.values.get(i).copied().map(raw_to_ref)
    }

    pub fn iter(&self) -> impl Iterator<Item = ValueRef<'row>> + '_ {
        self.values.iter().copied().map(raw_to_ref)
    }
}

impl TryFrom<ValueRef<'_>> for Value {
    type Error = Error;

    fn try_from(value: ValueRef<'_>) -> Result<Self> {
        match value {
            ValueRef::Null => Ok(Value::Null),
            ValueRef::Integer(value) => Ok(Value::Integer(value)),
            ValueRef::Real(value) => Ok(Value::Real(value)),
            ValueRef::TextBytes(bytes) => Ok(Value::Text(str::from_utf8(bytes)?.to_owned())),
            ValueRef::Blob(bytes) => Ok(Value::Blob(bytes.to_owned())),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TableRow {
    pub rowid: i64,
    pub values: Vec<Value>,
}

#[derive(Debug, Default)]
pub struct RowScratch {
    values: Vec<ValueRefRaw>,
    overflow_buf: Vec<u8>,
    btree_stack: Vec<PageId>,
}

impl RowScratch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(values: usize, overflow: usize) -> Self {
        Self {
            values: Vec::with_capacity(values),
            overflow_buf: Vec::with_capacity(overflow),
            btree_stack: Vec::new(),
        }
    }

    fn take_stack(&mut self) -> Vec<PageId> {
        std::mem::take(&mut self.btree_stack)
    }

    fn return_stack(&mut self, stack: Vec<PageId>) {
        self.btree_stack = stack;
    }

    fn prepare_row(&mut self) {
        self.values.clear();
        self.overflow_buf.clear();
    }

    fn split_mut(&mut self) -> (&mut Vec<ValueRefRaw>, &mut Vec<u8>) {
        (&mut self.values, &mut self.overflow_buf)
    }

    fn values_slice(&self) -> &[ValueRefRaw] {
        self.values.as_slice()
    }
}

pub fn read_table(pager: &Pager, page_id: PageId) -> Result<Vec<TableRow>> {
    let mut rows = Vec::new();
    let mut scratch = RowScratch::with_capacity(8, 0);
    scan_table_ref_with_scratch(pager, page_id, &mut scratch, |rowid, row| {
        let mut owned = Vec::with_capacity(row.len());
        for value in row.iter() {
            owned.push(Value::try_from(value)?);
        }
        rows.push(TableRow { rowid, values: owned });
        Ok(())
    })?;
    Ok(rows)
}

pub fn read_table_ref(pager: &Pager, page_id: PageId) -> Result<Vec<TableRow>> {
    read_table(pager, page_id)
}

pub fn scan_table_ref_with_scratch<F>(
    pager: &Pager,
    page_id: PageId,
    scratch: &mut RowScratch,
    mut f: F,
) -> Result<()>
where
    F: for<'row> FnMut(i64, RowView<'row>) -> Result<()>,
{
    scan_table_page_ref(pager, page_id, scratch, &mut f)
}

pub fn scan_table_ref<F>(pager: &Pager, page_id: PageId, f: F) -> Result<()>
where
    F: for<'row> FnMut(i64, RowView<'row>) -> Result<()>,
{
    let mut scratch = RowScratch::with_capacity(8, 0);
    scan_table_ref_with_scratch(pager, page_id, &mut scratch, f)
}

pub fn scan_table_ref_until<F, T>(pager: &Pager, page_id: PageId, mut f: F) -> Result<Option<T>>
where
    F: for<'row> FnMut(i64, RowView<'row>) -> Result<Option<T>>,
{
    let mut scratch = RowScratch::with_capacity(8, 0);
    scan_table_page_ref_until(pager, page_id, &mut scratch, &mut f)
}

pub fn scan_table_cells_with_scratch<F>(pager: &Pager, page_id: PageId, mut f: F) -> Result<()>
where
    F: for<'row> FnMut(i64, RecordPayload<'row>) -> Result<()>,
{
    scan_table_page_cells(pager, page_id, &mut f)
}

pub fn lookup_rowid_payload<'row>(
    pager: &'row Pager,
    page_id: PageId,
    target_rowid: i64,
) -> Result<Option<RecordPayload<'row>>> {
    let mut page_id = page_id;
    let max_pages = pager.page_count().max(1);
    let mut seen_pages = 0u32;

    loop {
        seen_pages += 1;
        if seen_pages > max_pages {
            return Err(Error::Corrupted("btree page cycle detected"));
        }

        let page = pager.page(page_id)?;
        let header = parse_header(&page)?;
        let cell_ptrs = cell_ptrs(&page, &header)?;

        match header.kind {
            BTreeKind::TableLeaf => {
                let cell_count = header.cell_count as usize;
                let mut lo = 0usize;
                let mut hi = cell_count;

                while lo < hi {
                    let mid = (lo + hi) / 2;
                    let offset = u16::from_be_bytes([cell_ptrs[mid * 2], cell_ptrs[mid * 2 + 1]]);
                    let rowid = read_table_leaf_rowid(&page, offset)?;
                    if rowid < target_rowid {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                if lo < cell_count {
                    let offset = u16::from_be_bytes([cell_ptrs[lo * 2], cell_ptrs[lo * 2 + 1]]);
                    let rowid = read_table_leaf_rowid(&page, offset)?;
                    if rowid == target_rowid {
                        let (_rowid, payload) =
                            read_table_cell_payload_from_bytes(pager, page_id, offset)?;
                        return Ok(Some(payload));
                    }
                }

                return Ok(None);
            }
            BTreeKind::TableInterior => {
                let cell_count = header.cell_count as usize;
                let mut lo = 0usize;
                let mut hi = cell_count;

                while lo < hi {
                    let mid = (lo + hi) / 2;
                    let offset = u16::from_be_bytes([cell_ptrs[mid * 2], cell_ptrs[mid * 2 + 1]]);
                    let (_child, key) = read_table_interior_cell(&page, offset)?;
                    if target_rowid <= key {
                        hi = mid;
                    } else {
                        lo = mid + 1;
                    }
                }

                if lo < cell_count {
                    let offset = u16::from_be_bytes([cell_ptrs[lo * 2], cell_ptrs[lo * 2 + 1]]);
                    let (child, _key) = read_table_interior_cell(&page, offset)?;
                    page_id = child;
                } else {
                    let right_most = header
                        .right_most_child
                        .ok_or(Error::Corrupted("missing right-most child pointer"))?;
                    page_id = PageId::try_new(right_most)
                        .ok_or(Error::Corrupted("child page id is zero"))?;
                }
            }
        }
    }
}

pub(crate) fn scan_table_cells_with_scratch_and_stack_until<F, T>(
    pager: &Pager,
    page_id: PageId,
    stack: &mut Vec<PageId>,
    mut f: F,
) -> Result<Option<T>>
where
    F: for<'row> FnMut(i64, RecordPayload<'row>) -> Result<Option<T>>,
{
    stack.clear();
    stack.push(page_id);
    scan_table_page_cells_until_with_stack(pager, stack, &mut f)
}

fn scan_table_page_ref<'pager, F>(
    pager: &'pager Pager,
    page_id: PageId,
    scratch: &mut RowScratch,
    f: &mut F,
) -> Result<()>
where
    F: for<'row> FnMut(i64, RowView<'row>) -> Result<()>,
{
    let page = pager.page(page_id)?;
    let header = parse_header(&page)?;
    let root_cell_ptrs = cell_ptrs(&page, &header)?;

    if matches!(header.kind, BTreeKind::TableLeaf) {
        for idx in 0..header.cell_count as usize {
            let offset = u16::from_be_bytes([root_cell_ptrs[idx * 2], root_cell_ptrs[idx * 2 + 1]]);
            let rowid = read_table_cell_into(pager, &page, offset, scratch)?;
            let row = RowView { values: scratch.values_slice() };
            f(rowid, row)?;
        }
        return Ok(());
    }

    let mut stack = scratch.take_stack();
    stack.clear();

    let max_pages = pager.page_count().max(1);
    let mut seen_pages = 1u32;

    if let Some(right_most) = header.right_most_child {
        let right_most =
            PageId::try_new(right_most).ok_or(Error::Corrupted("child page id is zero"))?;
        stack.push(right_most);
    }

    let page_len = page.usable_bytes().len();
    for idx in (0..header.cell_count as usize).rev() {
        let offset = u16::from_be_bytes([root_cell_ptrs[idx * 2], root_cell_ptrs[idx * 2 + 1]]);
        if offset as usize >= page_len {
            return Err(Error::Corrupted("cell offset out of bounds"));
        }

        let mut decoder = Decoder::new(page.usable_bytes()).split_at(offset as usize);
        let child = read_u32_checked(&mut decoder, "cell child pointer truncated")?;
        let child = PageId::try_new(child).ok_or(Error::Corrupted("child page id is zero"))?;

        // Skip the key separating subtrees; it is not needed for scanning.
        let _ = read_varint_checked(&mut decoder, "cell key truncated")?;
        stack.push(child);
    }

    let result = (|| {
        while let Some(page_id) = stack.pop() {
            seen_pages += 1;
            if seen_pages > max_pages {
                return Err(Error::Corrupted("btree page cycle detected"));
            }

            let page = pager.page(page_id)?;
            let header = parse_header(&page)?;
            let cell_ptrs = cell_ptrs(&page, &header)?;

            match header.kind {
                BTreeKind::TableLeaf => {
                    for idx in 0..header.cell_count as usize {
                        let offset =
                            u16::from_be_bytes([cell_ptrs[idx * 2], cell_ptrs[idx * 2 + 1]]);
                        let rowid = read_table_cell_into(pager, &page, offset, scratch)?;
                        let row = RowView { values: scratch.values_slice() };
                        f(rowid, row)?;
                    }
                }
                BTreeKind::TableInterior => {
                    if let Some(right_most) = header.right_most_child {
                        let right_most = PageId::try_new(right_most)
                            .ok_or(Error::Corrupted("child page id is zero"))?;
                        stack.push(right_most);
                    }

                    let page_len = page.usable_bytes().len();
                    for idx in (0..header.cell_count as usize).rev() {
                        let offset =
                            u16::from_be_bytes([cell_ptrs[idx * 2], cell_ptrs[idx * 2 + 1]]);
                        if offset as usize >= page_len {
                            return Err(Error::Corrupted("cell offset out of bounds"));
                        }

                        let mut decoder =
                            Decoder::new(page.usable_bytes()).split_at(offset as usize);
                        let child = read_u32_checked(&mut decoder, "cell child pointer truncated")?;
                        let child = PageId::try_new(child)
                            .ok_or(Error::Corrupted("child page id is zero"))?;

                        // Skip the key separating subtrees; it is not needed for scanning.
                        let _ = read_varint_checked(&mut decoder, "cell key truncated")?;
                        stack.push(child);
                    }
                }
            }
        }
        Ok(())
    })();

    scratch.return_stack(stack);
    result
}

fn scan_table_page_cells<'pager, F>(pager: &'pager Pager, page_id: PageId, f: &mut F) -> Result<()>
where
    F: for<'row> FnMut(i64, RecordPayload<'row>) -> Result<()>,
{
    let mut stack = vec![page_id];
    scan_table_page_cells_with_stack(pager, &mut stack, f)
}

fn scan_table_page_cells_with_stack<'pager, F>(
    pager: &'pager Pager,
    stack: &mut Vec<PageId>,
    f: &mut F,
) -> Result<()>
where
    F: for<'row> FnMut(i64, RecordPayload<'row>) -> Result<()>,
{
    let max_pages = pager.page_count().max(1);
    let mut seen_pages = 0u32;

    while let Some(page_id) = stack.pop() {
        seen_pages += 1;
        if seen_pages > max_pages {
            return Err(Error::Corrupted("btree page cycle detected"));
        }

        let page = pager.page(page_id)?;
        let header = parse_header(&page)?;
        let cell_ptrs = cell_ptrs(&page, &header)?;

        match header.kind {
            BTreeKind::TableLeaf => {
                for idx in 0..header.cell_count as usize {
                    let offset = u16::from_be_bytes([cell_ptrs[idx * 2], cell_ptrs[idx * 2 + 1]]);
                    let (rowid, payload) = read_table_cell_payload(pager, &page, offset)?;
                    f(rowid, payload)?;
                }
            }
            BTreeKind::TableInterior => {
                if let Some(right_most) = header.right_most_child {
                    let right_most = PageId::try_new(right_most)
                        .ok_or(Error::Corrupted("child page id is zero"))?;
                    stack.push(right_most);
                }

                let page_len = page.usable_bytes().len();
                for idx in (0..header.cell_count as usize).rev() {
                    let offset = u16::from_be_bytes([cell_ptrs[idx * 2], cell_ptrs[idx * 2 + 1]]);
                    if offset as usize >= page_len {
                        return Err(Error::Corrupted("cell offset out of bounds"));
                    }

                    let mut decoder = Decoder::new(page.usable_bytes()).split_at(offset as usize);
                    let child = read_u32_checked(&mut decoder, "cell child pointer truncated")?;
                    let child =
                        PageId::try_new(child).ok_or(Error::Corrupted("child page id is zero"))?;

                    let _ = read_varint_checked(&mut decoder, "cell key truncated")?;
                    stack.push(child);
                }
            }
        }
    }

    Ok(())
}

fn scan_table_page_ref_until<'pager, F, T>(
    pager: &'pager Pager,
    page_id: PageId,
    scratch: &mut RowScratch,
    f: &mut F,
) -> Result<Option<T>>
where
    F: for<'row> FnMut(i64, RowView<'row>) -> Result<Option<T>>,
{
    let page = pager.page(page_id)?;
    let header = parse_header(&page)?;
    let root_cell_ptrs = cell_ptrs(&page, &header)?;

    if matches!(header.kind, BTreeKind::TableLeaf) {
        for idx in 0..header.cell_count as usize {
            let offset = u16::from_be_bytes([root_cell_ptrs[idx * 2], root_cell_ptrs[idx * 2 + 1]]);
            let rowid = read_table_cell_into(pager, &page, offset, scratch)?;
            let row = RowView { values: scratch.values_slice() };
            if let Some(value) = f(rowid, row)? {
                return Ok(Some(value));
            }
        }
        return Ok(None);
    }

    let mut stack = scratch.take_stack();
    stack.clear();
    stack.push(page_id);
    let max_pages = pager.page_count().max(1);
    let mut seen_pages = 0u32;

    let result = (|| {
        while let Some(page_id) = stack.pop() {
            seen_pages += 1;
            if seen_pages > max_pages {
                return Err(Error::Corrupted("btree page cycle detected"));
            }

            let page = pager.page(page_id)?;
            let header = parse_header(&page)?;
            let cell_ptrs = cell_ptrs(&page, &header)?;

            match header.kind {
                BTreeKind::TableLeaf => {
                    for idx in 0..header.cell_count as usize {
                        let offset =
                            u16::from_be_bytes([cell_ptrs[idx * 2], cell_ptrs[idx * 2 + 1]]);
                        let rowid = read_table_cell_into(pager, &page, offset, scratch)?;
                        let row = RowView { values: scratch.values_slice() };
                        if let Some(value) = f(rowid, row)? {
                            return Ok(Some(value));
                        }
                    }
                }
                BTreeKind::TableInterior => {
                    if let Some(right_most) = header.right_most_child {
                        let right_most = PageId::try_new(right_most)
                            .ok_or(Error::Corrupted("child page id is zero"))?;
                        stack.push(right_most);
                    }

                    let page_len = page.usable_bytes().len();
                    for idx in (0..header.cell_count as usize).rev() {
                        let offset =
                            u16::from_be_bytes([cell_ptrs[idx * 2], cell_ptrs[idx * 2 + 1]]);
                        if offset as usize >= page_len {
                            return Err(Error::Corrupted("cell offset out of bounds"));
                        }

                        let mut decoder =
                            Decoder::new(page.usable_bytes()).split_at(offset as usize);
                        let child = read_u32_checked(&mut decoder, "cell child pointer truncated")?;
                        let child = PageId::try_new(child)
                            .ok_or(Error::Corrupted("child page id is zero"))?;

                        // Skip the key separating subtrees; it is not needed for scanning.
                        let _ = read_varint_checked(&mut decoder, "cell key truncated")?;
                        stack.push(child);
                    }
                }
            }
        }

        Ok(None)
    })();

    scratch.return_stack(stack);
    result
}

fn scan_table_page_cells_until_with_stack<'pager, F, T>(
    pager: &'pager Pager,
    stack: &mut Vec<PageId>,
    f: &mut F,
) -> Result<Option<T>>
where
    F: for<'row> FnMut(i64, RecordPayload<'row>) -> Result<Option<T>>,
{
    let max_pages = pager.page_count().max(1);
    let mut seen_pages = 0u32;

    while let Some(page_id) = stack.pop() {
        seen_pages += 1;
        if seen_pages > max_pages {
            return Err(Error::Corrupted("btree page cycle detected"));
        }

        let page = pager.page(page_id)?;
        let header = parse_header(&page)?;
        let cell_ptrs = cell_ptrs(&page, &header)?;

        match header.kind {
            BTreeKind::TableLeaf => {
                for idx in 0..header.cell_count as usize {
                    let offset = u16::from_be_bytes([cell_ptrs[idx * 2], cell_ptrs[idx * 2 + 1]]);
                    let (rowid, payload) = read_table_cell_payload(pager, &page, offset)?;
                    if let Some(value) = f(rowid, payload)? {
                        return Ok(Some(value));
                    }
                }
            }
            BTreeKind::TableInterior => {
                if let Some(right_most) = header.right_most_child {
                    let right_most = PageId::try_new(right_most)
                        .ok_or(Error::Corrupted("child page id is zero"))?;
                    stack.push(right_most);
                }

                let page_len = page.usable_bytes().len();
                for idx in (0..header.cell_count as usize).rev() {
                    let offset = u16::from_be_bytes([cell_ptrs[idx * 2], cell_ptrs[idx * 2 + 1]]);
                    if offset as usize >= page_len {
                        return Err(Error::Corrupted("cell offset out of bounds"));
                    }

                    let mut decoder = Decoder::new(page.usable_bytes()).split_at(offset as usize);
                    let child = read_u32_checked(&mut decoder, "cell child pointer truncated")?;
                    let child =
                        PageId::try_new(child).ok_or(Error::Corrupted("child page id is zero"))?;

                    let _ = read_varint_checked(&mut decoder, "cell key truncated")?;
                    stack.push(child);
                }
            }
        }
    }

    Ok(None)
}

fn read_table_cell_into(
    pager: &Pager,
    page: &PageRef<'_>,
    offset: u16,
    scratch: &mut RowScratch,
) -> Result<i64> {
    scratch.prepare_row();
    let (values, spill) = scratch.split_mut();
    let (rowid, payload) = read_table_cell_payload(pager, page, offset)?;
    decode_record_into(payload, values, spill)?;
    Ok(rowid)
}

fn read_table_cell_payload<'row>(
    pager: &'row Pager,
    page: &'row PageRef<'_>,
    offset: u16,
) -> Result<(i64, RecordPayload<'row>)> {
    let usable = page.usable_bytes();
    if offset as usize >= usable.len() {
        return Err(Error::Corrupted("cell offset out of bounds"));
    }

    let mut pos = offset as usize;
    let remaining = usable.len().saturating_sub(pos);
    let (payload_length, rowid) = if remaining >= 18 {
        // SAFETY: remaining >= 18 guarantees two varints (max 9 bytes each) are
        // in-bounds.
        let payload_length = unsafe { read_varint_unchecked_at(usable, &mut pos) };
        let rowid = unsafe { read_varint_unchecked_at(usable, &mut pos) } as i64;
        (payload_length, rowid)
    } else {
        let payload_length = read_varint_at(usable, &mut pos, "cell payload length truncated")?;
        let rowid = read_varint_at(usable, &mut pos, "cell rowid truncated")? as i64;
        (payload_length, rowid)
    };
    let payload_length =
        usize::try_from(payload_length).map_err(|_| Error::Corrupted("payload is too large"))?;
    if payload_length > MAX_PAYLOAD_BYTES {
        return Err(Error::PayloadTooLarge(payload_length));
    }

    let start = pos;
    let usable_size = page.usable_size();
    let x = usable_size.checked_sub(35).ok_or(Error::Corrupted("usable size underflow"))?;
    if payload_length <= x {
        let end = start + payload_length;
        if end > usable.len() {
            return Err(Error::Corrupted("payload extends past page boundary"));
        }
        return Ok((rowid, RecordPayload::Inline(&usable[start..end])));
    }

    let local_len = local_payload_len(usable_size, payload_length)?;
    let end_local = start + local_len;
    if end_local > usable.len() {
        return Err(Error::Corrupted("payload extends past page boundary"));
    }

    let overflow_end = end_local + 4;
    if overflow_end > usable.len() {
        return Err(Error::Corrupted("overflow pointer out of bounds"));
    }
    let overflow_page = u32::from_be_bytes(usable[end_local..overflow_end].try_into().unwrap());
    if overflow_page == 0 {
        return Err(Error::OverflowChainTruncated);
    }
    let payload =
        OverflowPayload::new(pager, payload_length, &usable[start..end_local], overflow_page);
    Ok((rowid, RecordPayload::Overflow(payload)))
}

fn read_table_cell_payload_from_bytes<'row>(
    pager: &'row Pager,
    page_id: PageId,
    offset: u16,
) -> Result<(i64, RecordPayload<'row>)> {
    let page_bytes = pager.page_bytes(page_id)?;
    let usable_end = pager.header().usable_size.min(page_bytes.len());
    let usable = &page_bytes[..usable_end];

    if offset as usize >= usable.len() {
        return Err(Error::Corrupted("cell offset out of bounds"));
    }

    let mut pos = offset as usize;
    let payload_length = read_varint_at(usable, &mut pos, "cell payload length truncated")?;
    let rowid = read_varint_at(usable, &mut pos, "cell rowid truncated")? as i64;
    let payload_length =
        usize::try_from(payload_length).map_err(|_| Error::Corrupted("payload is too large"))?;
    if payload_length > MAX_PAYLOAD_BYTES {
        return Err(Error::PayloadTooLarge(payload_length));
    }

    let start = pos;
    let usable_size = pager.header().usable_size;
    let x = usable_size.checked_sub(35).ok_or(Error::Corrupted("usable size underflow"))?;
    if payload_length <= x {
        let end = start + payload_length;
        if end > usable.len() {
            return Err(Error::Corrupted("payload extends past page boundary"));
        }
        return Ok((rowid, RecordPayload::Inline(&usable[start..end])));
    }

    let local_len = local_payload_len(usable_size, payload_length)?;
    let end_local = start + local_len;
    if end_local > usable.len() {
        return Err(Error::Corrupted("payload extends past page boundary"));
    }

    let overflow_end = end_local + 4;
    if overflow_end > usable.len() {
        return Err(Error::Corrupted("overflow pointer out of bounds"));
    }
    let overflow_page = u32::from_be_bytes(usable[end_local..overflow_end].try_into().unwrap());
    if overflow_page == 0 {
        return Err(Error::OverflowChainTruncated);
    }
    let payload =
        OverflowPayload::new(pager, payload_length, &usable[start..end_local], overflow_page);
    Ok((rowid, RecordPayload::Overflow(payload)))
}

fn read_table_leaf_rowid(page: &PageRef<'_>, offset: u16) -> Result<i64> {
    let usable = page.usable_bytes();
    if offset as usize >= usable.len() {
        return Err(Error::Corrupted("cell offset out of bounds"));
    }

    let mut pos = offset as usize;
    let payload_length = read_varint_at(usable, &mut pos, "cell payload length truncated")?;
    let payload_length =
        usize::try_from(payload_length).map_err(|_| Error::Corrupted("payload is too large"))?;
    if payload_length > MAX_PAYLOAD_BYTES {
        return Err(Error::PayloadTooLarge(payload_length));
    }
    let rowid = read_varint_at(usable, &mut pos, "cell rowid truncated")? as i64;
    Ok(rowid)
}

fn read_table_interior_cell(page: &PageRef<'_>, offset: u16) -> Result<(PageId, i64)> {
    let usable = page.usable_bytes();
    let start = offset as usize;
    if start + 4 > usable.len() {
        return Err(Error::Corrupted("cell offset out of bounds"));
    }

    let child = u32::from_be_bytes([
        usable[start],
        usable[start + 1],
        usable[start + 2],
        usable[start + 3],
    ]);
    let child = PageId::try_new(child).ok_or(Error::Corrupted("child page id is zero"))?;
    let mut pos = start + 4;
    let key = read_varint_at(usable, &mut pos, "cell key truncated")? as i64;
    Ok((child, key))
}

#[allow(dead_code)]
pub(crate) fn assemble_overflow_payload(
    pager: &Pager,
    payload_len: usize,
    local_len: usize,
    mut overflow_page: u32,
    local_bytes: &[u8],
    overflow_buf: &mut Vec<u8>,
) -> Result<()> {
    overflow_buf.clear();
    overflow_buf.reserve(payload_len);
    overflow_buf.extend_from_slice(local_bytes);

    let mut remaining =
        payload_len.checked_sub(local_len).ok_or(Error::Corrupted("payload length underflow"))?;

    let mut visited = 0u32;
    let max_pages = pager.page_count().max(1);

    while remaining > 0 {
        if visited >= max_pages {
            return Err(Error::OverflowLoopDetected);
        }
        visited += 1;

        let page_id = PageId::try_new(overflow_page).ok_or(Error::OverflowChainTruncated)?;
        let page_bytes = pager.page_bytes(page_id)?;
        if page_bytes.len() < 4 {
            return Err(Error::Corrupted("overflow page too small"));
        }

        let next_page = u32::from_be_bytes(page_bytes[0..4].try_into().unwrap());
        let usable = pager.header().usable_size.min(page_bytes.len());
        if usable < 4 {
            return Err(Error::Corrupted("overflow page usable size too small"));
        }
        let content = &page_bytes[4..usable];
        let take = remaining.min(content.len());
        overflow_buf.extend_from_slice(&content[..take]);
        remaining -= take;

        if remaining == 0 {
            break;
        }
        if next_page == 0 {
            return Err(Error::OverflowChainTruncated);
        }
        overflow_page = next_page;
    }

    if overflow_buf.len() != payload_len {
        return Err(Error::OverflowChainTruncated);
    }

    Ok(())
}

#[derive(Debug)]
enum ValueBytes {
    Inline(RawBytes),
    Spill { start: usize, len: usize },
}

#[derive(Debug)]
enum PendingKind {
    Text,
    Blob,
}

#[derive(Debug)]
struct PendingBytes {
    out_index: usize,
    start: usize,
    len: usize,
    kind: PendingKind,
}

struct OverflowCursor<'row> {
    payload: &'row OverflowPayload<'row>,
    pos: usize,
    segment_start: usize,
    segment: &'row [u8],
    next_overflow: u32,
    visited: u32,
    max_pages: u32,
}

impl<'row> OverflowCursor<'row> {
    fn new(payload: &'row OverflowPayload<'row>, offset: usize) -> Result<Self> {
        if offset > payload.total_len {
            return Err(Error::Corrupted("record payload shorter than declared"));
        }

        let mut cursor = Self {
            payload,
            pos: 0,
            segment_start: 0,
            segment: payload.local,
            next_overflow: payload.first_overflow,
            visited: 0,
            max_pages: payload.pager.page_count().max(1),
        };
        cursor.skip(offset)?;
        Ok(cursor)
    }

    fn position(&self) -> usize {
        self.pos
    }

    fn segment_end(&self) -> usize {
        self.segment_start + self.segment.len()
    }

    fn ensure_segment(&mut self, msg: &'static str) -> Result<()> {
        if self.pos < self.segment_end() {
            return Ok(());
        }
        if self.pos >= self.payload.total_len {
            return Err(Error::Corrupted(msg));
        }
        self.advance_segment()
    }

    fn advance_segment(&mut self) -> Result<()> {
        if self.next_overflow == 0 {
            return Err(Error::OverflowChainTruncated);
        }
        if self.visited >= self.max_pages {
            return Err(Error::OverflowLoopDetected);
        }
        self.visited += 1;

        let page_id = PageId::try_new(self.next_overflow).ok_or(Error::OverflowChainTruncated)?;
        let page_bytes = self.payload.pager.page_bytes(page_id)?;
        if page_bytes.len() < 4 {
            return Err(Error::Corrupted("overflow page too small"));
        }

        let next_page = u32::from_be_bytes(page_bytes[0..4].try_into().unwrap());
        let usable = self.payload.pager.header().usable_size.min(page_bytes.len());
        if usable < 4 {
            return Err(Error::Corrupted("overflow page usable size too small"));
        }
        let content = &page_bytes[4..usable];

        self.segment_start = self.pos;
        self.segment = content;
        self.next_overflow = next_page;
        Ok(())
    }

    fn read_byte(&mut self, msg: &'static str) -> Result<u8> {
        self.ensure_segment(msg)?;
        if self.pos >= self.segment_end() {
            return Err(Error::Corrupted(msg));
        }
        let offset = self.pos - self.segment_start;
        let byte = unsafe { *self.segment.get_unchecked(offset) };
        self.pos += 1;
        Ok(byte)
    }

    fn read_varint(&mut self, msg: &'static str) -> Result<u64> {
        let first = self.read_byte(msg)?;
        if first & 0x80 == 0 {
            return Ok(u64::from(first));
        }

        let mut result = u64::from(first & 0x7F);
        for _ in 0..7 {
            let byte = self.read_byte(msg)?;
            result = (result << 7) | u64::from(byte & 0x7F);
            if byte & 0x80 == 0 {
                return Ok(result);
            }
        }

        let byte = self.read_byte(msg)?;
        Ok((result << 8) | u64::from(byte))
    }

    fn read_bytes_into(&mut self, out: &mut [u8], msg: &'static str) -> Result<()> {
        for byte in out.iter_mut() {
            *byte = self.read_byte(msg)?;
        }
        Ok(())
    }

    fn read_signed_be(&mut self, len: usize) -> Result<i64> {
        let mut buf = [0u8; 8];
        let start = 8 - len;
        self.read_bytes_into(&mut buf[start..], "record payload shorter than declared")?;
        let value = u64::from_be_bytes(buf);
        let shift = (8 - len) * 8;
        Ok(((value << shift) as i64) >> shift)
    }

    fn read_u64_be(&mut self) -> Result<u64> {
        let mut buf = [0u8; 8];
        self.read_bytes_into(&mut buf, "record payload shorter than declared")?;
        Ok(u64::from_be_bytes(buf))
    }

    fn read_value_bytes(&mut self, len: usize, spill: &mut Vec<u8>) -> Result<ValueBytes> {
        if len == 0 {
            return Ok(ValueBytes::Inline(RawBytes::from_slice(&[])));
        }

        self.ensure_segment("record payload shorter than declared")?;
        let available = self.segment_end() - self.pos;
        if len <= available {
            let offset = self.pos - self.segment_start;
            let slice = &self.segment[offset..offset + len];
            self.pos += len;
            return Ok(ValueBytes::Inline(RawBytes::from_slice(slice)));
        }

        let start = spill.len();
        let mut remaining = len;
        while remaining > 0 {
            self.ensure_segment("record payload shorter than declared")?;
            let available = self.segment_end() - self.pos;
            if available == 0 {
                continue;
            }
            let take = remaining.min(available);
            let offset = self.pos - self.segment_start;
            spill.extend_from_slice(&self.segment[offset..offset + take]);
            self.pos += take;
            remaining -= take;
        }

        Ok(ValueBytes::Spill { start, len })
    }

    fn skip(&mut self, mut len: usize) -> Result<()> {
        while len > 0 {
            self.ensure_segment("record payload shorter than declared")?;
            let available = self.segment_end() - self.pos;
            if available == 0 {
                continue;
            }
            let take = len.min(available);
            self.pos += take;
            len -= take;
        }
        Ok(())
    }
}

fn decode_record_project_into_overflow(
    payload: &OverflowPayload<'_>,
    needed_cols: Option<&[u16]>,
    out: &mut Vec<ValueRefRaw>,
    spill: &mut Vec<u8>,
) -> Result<usize> {
    out.clear();
    spill.clear();

    let mut serial_cursor = OverflowCursor::new(payload, 0)?;
    let header_len = serial_cursor.read_varint("record header truncated")? as usize;
    let header_pos = serial_cursor.position();
    if header_len < header_pos || header_len > payload.total_len {
        return Err(Error::Corrupted("invalid record header length"));
    }

    let mut value_cursor = OverflowCursor::new(payload, header_len)?;
    let mut pending: Vec<PendingBytes> = Vec::new();

    if let Some(needed_cols) = needed_cols {
        let mut needed_iter = needed_cols.iter().copied();
        let mut next_needed = needed_iter.next();
        let mut column_count = 0usize;
        while serial_cursor.position() < header_len {
            let serial = serial_cursor.read_varint("record header truncated")?;
            if serial_cursor.position() > header_len {
                return Err(Error::Corrupted("record header truncated"));
            }
            let col_idx = column_count as u16;
            if Some(col_idx) == next_needed {
                let decoded = decode_value_ref_at_cursor(serial, &mut value_cursor, spill)?;
                match decoded {
                    DecodedValue::Ready(raw) => out.push(raw),
                    DecodedValue::Spill { start, len, kind } => {
                        let out_index = out.len();
                        out.push(ValueRefRaw::Null);
                        pending.push(PendingBytes { out_index, start, len, kind });
                    }
                }
                next_needed = needed_iter.next();
            } else {
                skip_value_at_cursor(serial, &mut value_cursor)?;
            }
            column_count += 1;
        }

        for pending in pending {
            let slice = &spill[pending.start..pending.start + pending.len];
            let raw = RawBytes::from_slice(slice);
            out[pending.out_index] = match pending.kind {
                PendingKind::Text => ValueRefRaw::TextBytes(raw),
                PendingKind::Blob => ValueRefRaw::Blob(raw),
            };
        }

        Ok(column_count)
    } else {
        let mut column_count = 0usize;
        while serial_cursor.position() < header_len {
            let serial = serial_cursor.read_varint("record header truncated")?;
            if serial_cursor.position() > header_len {
                return Err(Error::Corrupted("record header truncated"));
            }
            let decoded = decode_value_ref_at_cursor(serial, &mut value_cursor, spill)?;
            match decoded {
                DecodedValue::Ready(raw) => out.push(raw),
                DecodedValue::Spill { start, len, kind } => {
                    let out_index = out.len();
                    out.push(ValueRefRaw::Null);
                    pending.push(PendingBytes { out_index, start, len, kind });
                }
            }
            column_count += 1;
        }

        for pending in pending {
            let slice = &spill[pending.start..pending.start + pending.len];
            let raw = RawBytes::from_slice(slice);
            out[pending.out_index] = match pending.kind {
                PendingKind::Text => ValueRefRaw::TextBytes(raw),
                PendingKind::Blob => ValueRefRaw::Blob(raw),
            };
        }

        Ok(column_count)
    }
}

pub(crate) fn decode_record_project_into(
    payload: RecordPayload<'_>,
    needed_cols: Option<&[u16]>,
    out: &mut Vec<ValueRefRaw>,
    spill: &mut Vec<u8>,
) -> Result<usize> {
    match payload {
        RecordPayload::Inline(bytes) => {
            decode_record_project_into_bytes(bytes, needed_cols, out)
        }
        RecordPayload::Overflow(payload) => {
            decode_record_project_into_overflow(&payload, needed_cols, out, spill)
        }
    }
}

fn decode_record_project_into_bytes(
    payload: &[u8],
    needed_cols: Option<&[u16]>,
    out: &mut Vec<ValueRefRaw>,
) -> Result<usize> {
    out.clear();

    let mut header_pos = 0usize;
    let first = *payload.first().ok_or(Error::Corrupted("record header truncated"))?;
    let header_len = if first < 0x80 {
        header_pos = 1;
        first as usize
    } else {
        read_varint_at(payload, &mut header_pos, "record header truncated")? as usize
    };
    if header_len < header_pos || header_len > payload.len() {
        return Err(Error::Corrupted("invalid record header length"));
    }

    let serial_bytes = &payload[header_pos..header_len];
    let mut serial_pos = 0usize;
    let mut value_pos = header_len;

    if let Some(needed_cols) = needed_cols {
        let mut needed_iter = needed_cols.iter().copied();
        let mut next_needed = needed_iter.next();
        let mut column_count = 0usize;
        while serial_pos < serial_bytes.len() {
            let b = unsafe { *serial_bytes.get_unchecked(serial_pos) };
            let serial = if b < 0x80 {
                serial_pos += 1;
                b as u64
            } else {
                read_varint_at(serial_bytes, &mut serial_pos, "record header truncated")?
            };
            let col_idx = column_count as u16;
            if Some(col_idx) == next_needed {
                out.push(decode_value_ref_at(serial, payload, &mut value_pos)?);
                next_needed = needed_iter.next();
            } else {
                skip_value_at(serial, payload, &mut value_pos)?;
            }
            column_count += 1;
        }
        Ok(column_count)
    } else {
        let mut column_count = 0usize;
        while serial_pos < serial_bytes.len() {
            let b = unsafe { *serial_bytes.get_unchecked(serial_pos) };
            let serial = if b < 0x80 {
                serial_pos += 1;
                b as u64
            } else {
                read_varint_at(serial_bytes, &mut serial_pos, "record header truncated")?
            };
            out.push(decode_value_ref_at(serial, payload, &mut value_pos)?);
            column_count += 1;
        }
        Ok(column_count)
    }
}

pub(crate) fn decode_record_column(
    payload: RecordPayload<'_>,
    col: u16,
    spill: &mut Vec<u8>,
) -> Result<Option<ValueRefRaw>> {
    match payload {
        RecordPayload::Inline(bytes) => decode_record_column_bytes(bytes, col),
        RecordPayload::Overflow(payload) => decode_record_column_overflow(&payload, col, spill),
    }
}

fn decode_record_column_bytes(payload: &[u8], col: u16) -> Result<Option<ValueRefRaw>> {
    let first = *payload.first().ok_or(Error::Corrupted("record header truncated"))?;
    let mut header_pos = 0usize;
    let header_len = if first < 0x80 {
        header_pos = 1;
        first as usize
    } else {
        read_varint_at(payload, &mut header_pos, "record header truncated")? as usize
    };
    if header_len < header_pos || header_len > payload.len() {
        return Err(Error::Corrupted("invalid record header length"));
    }

    let serial_bytes = &payload[header_pos..header_len];
    let mut serial_pos = 0usize;
    let mut value_pos = header_len;

    let target = col as usize;
    let mut idx = 0usize;

    while serial_pos < serial_bytes.len() {
        let b = unsafe { *serial_bytes.get_unchecked(serial_pos) };
        let serial = if b < 0x80 {
            serial_pos += 1;
            b as u64
        } else {
            read_varint_at(serial_bytes, &mut serial_pos, "record header truncated")?
        };

        if idx == target {
            let v = decode_value_ref_at(serial, payload, &mut value_pos)?;
            return Ok(Some(v));
        } else {
            skip_value_at(serial, payload, &mut value_pos)?;
        }
        idx += 1;
    }
    Ok(None)
}

fn decode_record_into(
    payload: RecordPayload<'_>,
    out: &mut Vec<ValueRefRaw>,
    spill: &mut Vec<u8>,
) -> Result<()> {
    match payload {
        RecordPayload::Inline(bytes) => decode_record_into_bytes(bytes, out),
        RecordPayload::Overflow(payload) => decode_record_into_overflow(&payload, out, spill),
    }
}

fn decode_record_into_bytes(payload: &[u8], out: &mut Vec<ValueRefRaw>) -> Result<()> {
    out.clear();

    let mut header_pos = 0usize;
    let first = *payload.first().ok_or(Error::Corrupted("record header truncated"))?;
    let header_len = if first < 0x80 {
        header_pos = 1;
        first as usize
    } else {
        read_varint_at(payload, &mut header_pos, "record header truncated")? as usize
    };
    if header_len < header_pos || header_len > payload.len() {
        return Err(Error::Corrupted("invalid record header length"));
    }

    let serial_bytes = &payload[header_pos..header_len];
    let mut serial_pos = 0usize;
    let mut value_pos = header_len;
    while serial_pos < serial_bytes.len() {
        let b = unsafe { *serial_bytes.get_unchecked(serial_pos) };
        let serial = if b < 0x80 {
            serial_pos += 1;
            b as u64
        } else {
            read_varint_at(serial_bytes, &mut serial_pos, "record header truncated")?
        };
        out.push(decode_value_ref_at(serial, payload, &mut value_pos)?);
    }

    Ok(())
}

fn decode_record_into_overflow(
    payload: &OverflowPayload<'_>,
    out: &mut Vec<ValueRefRaw>,
    spill: &mut Vec<u8>,
) -> Result<()> {
    out.clear();
    spill.clear();

    let mut serial_cursor = OverflowCursor::new(payload, 0)?;
    let header_len = serial_cursor.read_varint("record header truncated")? as usize;
    let header_pos = serial_cursor.position();
    if header_len < header_pos || header_len > payload.total_len {
        return Err(Error::Corrupted("invalid record header length"));
    }

    let mut value_cursor = OverflowCursor::new(payload, header_len)?;
    let mut pending: Vec<PendingBytes> = Vec::new();

    while serial_cursor.position() < header_len {
        let serial = serial_cursor.read_varint("record header truncated")?;
        if serial_cursor.position() > header_len {
            return Err(Error::Corrupted("record header truncated"));
        }
        let decoded = decode_value_ref_at_cursor(serial, &mut value_cursor, spill)?;
        match decoded {
            DecodedValue::Ready(raw) => out.push(raw),
            DecodedValue::Spill { start, len, kind } => {
                let out_index = out.len();
                out.push(ValueRefRaw::Null);
                pending.push(PendingBytes { out_index, start, len, kind });
            }
        }
    }

    for pending in pending {
        let slice = &spill[pending.start..pending.start + pending.len];
        let raw = RawBytes::from_slice(slice);
        out[pending.out_index] = match pending.kind {
            PendingKind::Text => ValueRefRaw::TextBytes(raw),
            PendingKind::Blob => ValueRefRaw::Blob(raw),
        };
    }

    Ok(())
}

fn decode_record_column_overflow(
    payload: &OverflowPayload<'_>,
    col: u16,
    spill: &mut Vec<u8>,
) -> Result<Option<ValueRefRaw>> {
    spill.clear();

    let mut serial_cursor = OverflowCursor::new(payload, 0)?;
    let header_len = serial_cursor.read_varint("record header truncated")? as usize;
    let header_pos = serial_cursor.position();
    if header_len < header_pos || header_len > payload.total_len {
        return Err(Error::Corrupted("invalid record header length"));
    }

    let mut value_cursor = OverflowCursor::new(payload, header_len)?;
    let target = col as usize;
    let mut idx = 0usize;

    while serial_cursor.position() < header_len {
        let serial = serial_cursor.read_varint("record header truncated")?;
        if serial_cursor.position() > header_len {
            return Err(Error::Corrupted("record header truncated"));
        }

        if idx == target {
            let decoded = decode_value_ref_at_cursor(serial, &mut value_cursor, spill)?;
            let raw = match decoded {
                DecodedValue::Ready(raw) => raw,
                DecodedValue::Spill { start, len, kind } => {
                    let slice = &spill[start..start + len];
                    let raw = RawBytes::from_slice(slice);
                    match kind {
                        PendingKind::Text => ValueRefRaw::TextBytes(raw),
                        PendingKind::Blob => ValueRefRaw::Blob(raw),
                    }
                }
            };
            return Ok(Some(raw));
        } else {
            skip_value_at_cursor(serial, &mut value_cursor)?;
        }
        idx += 1;
    }

    Ok(None)
}

fn skip_value_at(serial_type: u64, bytes: &[u8], pos: &mut usize) -> Result<()> {
    let len = serial_type_len(serial_type)?;
    let end = *pos + len;
    if end > bytes.len() {
        return Err(Error::Corrupted("record payload shorter than declared"));
    }
    *pos = end;
    Ok(())
}

fn serial_type_len(serial_type: u64) -> Result<usize> {
    match serial_type {
        0 | 8 | 9 => Ok(0),
        1 => Ok(1),
        2 => Ok(2),
        3 => Ok(3),
        4 => Ok(4),
        5 => Ok(6),
        6 | 7 => Ok(8),
        serial if serial >= 12 && serial % 2 == 0 => Ok(((serial - 12) / 2) as usize),
        serial if serial >= 13 => Ok(((serial - 13) / 2) as usize),
        other => Err(Error::UnsupportedSerialType(other)),
    }
}

fn decode_value_ref_at(serial_type: u64, bytes: &[u8], pos: &mut usize) -> Result<ValueRefRaw> {
    let value = match serial_type {
        0 => ValueRefRaw::Null,
        1 => ValueRefRaw::Integer(read_signed_be_at(bytes, pos, 1)?),
        2 => ValueRefRaw::Integer(read_signed_be_at(bytes, pos, 2)?),
        3 => ValueRefRaw::Integer(read_signed_be_at(bytes, pos, 3)?),
        4 => ValueRefRaw::Integer(read_signed_be_at(bytes, pos, 4)?),
        5 => ValueRefRaw::Integer(read_signed_be_at(bytes, pos, 6)?),
        6 => ValueRefRaw::Integer(read_signed_be_at(bytes, pos, 8)?),
        7 => ValueRefRaw::Real(f64::from_bits(read_u64_be_at(bytes, pos)?)),
        8 => ValueRefRaw::Integer(0),
        9 => ValueRefRaw::Integer(1),
        serial if serial >= 12 && serial % 2 == 0 => {
            let len = ((serial - 12) / 2) as usize;
            ValueRefRaw::Blob(RawBytes::from_slice(read_exact_bytes_at(bytes, pos, len)?))
        }
        serial if serial >= 13 => {
            let len = ((serial - 13) / 2) as usize;
            ValueRefRaw::TextBytes(RawBytes::from_slice(read_exact_bytes_at(bytes, pos, len)?))
        }
        other => return Err(Error::UnsupportedSerialType(other)),
    };

    Ok(value)
}

enum DecodedValue {
    Ready(ValueRefRaw),
    Spill { start: usize, len: usize, kind: PendingKind },
}

fn decode_value_ref_at_cursor(
    serial_type: u64,
    cursor: &mut OverflowCursor<'_>,
    spill: &mut Vec<u8>,
) -> Result<DecodedValue> {
    let value = match serial_type {
        0 => DecodedValue::Ready(ValueRefRaw::Null),
        1 => DecodedValue::Ready(ValueRefRaw::Integer(cursor.read_signed_be(1)?)),
        2 => DecodedValue::Ready(ValueRefRaw::Integer(cursor.read_signed_be(2)?)),
        3 => DecodedValue::Ready(ValueRefRaw::Integer(cursor.read_signed_be(3)?)),
        4 => DecodedValue::Ready(ValueRefRaw::Integer(cursor.read_signed_be(4)?)),
        5 => DecodedValue::Ready(ValueRefRaw::Integer(cursor.read_signed_be(6)?)),
        6 => DecodedValue::Ready(ValueRefRaw::Integer(cursor.read_signed_be(8)?)),
        7 => DecodedValue::Ready(ValueRefRaw::Real(f64::from_bits(cursor.read_u64_be()?))),
        8 => DecodedValue::Ready(ValueRefRaw::Integer(0)),
        9 => DecodedValue::Ready(ValueRefRaw::Integer(1)),
        serial if serial >= 12 && serial % 2 == 0 => {
            let len = ((serial - 12) / 2) as usize;
            match cursor.read_value_bytes(len, spill)? {
                ValueBytes::Inline(raw) => DecodedValue::Ready(ValueRefRaw::Blob(raw)),
                ValueBytes::Spill { start, len } => {
                    DecodedValue::Spill { start, len, kind: PendingKind::Blob }
                }
            }
        }
        serial if serial >= 13 => {
            let len = ((serial - 13) / 2) as usize;
            match cursor.read_value_bytes(len, spill)? {
                ValueBytes::Inline(raw) => DecodedValue::Ready(ValueRefRaw::TextBytes(raw)),
                ValueBytes::Spill { start, len } => {
                    DecodedValue::Spill { start, len, kind: PendingKind::Text }
                }
            }
        }
        other => return Err(Error::UnsupportedSerialType(other)),
    };

    Ok(value)
}

fn skip_value_at_cursor(serial_type: u64, cursor: &mut OverflowCursor<'_>) -> Result<()> {
    let len = serial_type_len(serial_type)?;
    cursor.skip(len)?;
    Ok(())
}

fn read_signed_be_at(bytes: &[u8], pos: &mut usize, len: usize) -> Result<i64> {
    let p = *pos;
    let end = p + len;
    if end > bytes.len() {
        return Err(Error::Corrupted("record payload shorter than declared"));
    }
    *pos = end;

    unsafe {
        Ok(match len {
            1 => i8::from_be_bytes([*bytes.get_unchecked(p)]) as i64,
            2 => {
                let v = ptr::read_unaligned(bytes.as_ptr().add(p) as *const u16);
                i16::from_be_bytes(v.to_be_bytes()) as i64
            }
            3 => {
                let b0 = *bytes.get_unchecked(p) as u32;
                let b1 = *bytes.get_unchecked(p + 1) as u32;
                let b2 = *bytes.get_unchecked(p + 2) as u32;
                let u = (b0 << 16) | (b1 << 8) | b2;
                let i = ((u << 8) as i32) >> 8;
                i as i64
            }
            4 => {
                let v = ptr::read_unaligned(bytes.as_ptr().add(p) as *const u32);
                i32::from_be_bytes(v.to_be_bytes()) as i64
            }
            6 => {
                let b0 = *bytes.get_unchecked(p) as u64;
                let b1 = *bytes.get_unchecked(p + 1) as u64;
                let b2 = *bytes.get_unchecked(p + 2) as u64;
                let b3 = *bytes.get_unchecked(p + 3) as u64;
                let b4 = *bytes.get_unchecked(p + 4) as u64;
                let b5 = *bytes.get_unchecked(p + 5) as u64;
                let u = (b0 << 40) | (b1 << 32) | (b2 << 24) | (b3 << 16) | (b4 << 8) | b5;
                let shift = 16;
                ((u << shift) as i64) >> shift
            }
            8 => {
                let v = ptr::read_unaligned(bytes.as_ptr().add(p) as *const u64);
                u64::from_be(v) as i64
            }
            _ => std::hint::unreachable_unchecked(),
        })
    }
}

fn read_u64_be_at(bytes: &[u8], pos: &mut usize) -> Result<u64> {
    let p = *pos;
    let end = p + 8;
    if end > bytes.len() {
        return Err(Error::Corrupted("record payload shorter than declared"));
    }
    *pos = end;
    unsafe {
        let v = ptr::read_unaligned(bytes.as_ptr().add(p) as *const u64);
        Ok(u64::from_be(v))
    }
}

fn read_u32_checked(decoder: &mut Decoder<'_>, msg: &'static str) -> Result<u32> {
    decoder.try_read_u32().ok_or(Error::Corrupted(msg))
}

fn read_varint_checked(decoder: &mut Decoder<'_>, msg: &'static str) -> Result<u64> {
    decoder.try_read_varint().ok_or(Error::Corrupted(msg))
}

pub(crate) fn read_varint_at(bytes: &[u8], pos: &mut usize, msg: &'static str) -> Result<u64> {
    if *pos >= bytes.len() {
        return Err(Error::Corrupted(msg));
    }

    let first = bytes[*pos];
    *pos += 1;
    if first & 0x80 == 0 {
        return Ok(u64::from(first));
    }

    let mut result = u64::from(first & 0x7F);
    for _ in 0..7 {
        if *pos >= bytes.len() {
            return Err(Error::Corrupted(msg));
        }
        let byte = bytes[*pos];
        *pos += 1;
        result = (result << 7) | u64::from(byte & 0x7F);
        if byte & 0x80 == 0 {
            return Ok(result);
        }
    }

    if *pos >= bytes.len() {
        return Err(Error::Corrupted(msg));
    }
    let byte = bytes[*pos];
    *pos += 1;
    Ok((result << 8) | u64::from(byte))
}

unsafe fn read_varint_unchecked_at(bytes: &[u8], pos: &mut usize) -> u64 {
    let mut idx = *pos;
    // SAFETY: caller guarantees enough bytes for full varint.
    let first = unsafe { *bytes.get_unchecked(idx) };
    idx += 1;
    if first & 0x80 == 0 {
        *pos = idx;
        return u64::from(first);
    }

    let mut result = u64::from(first & 0x7F);
    for _ in 0..7 {
        let byte = unsafe { *bytes.get_unchecked(idx) };
        idx += 1;
        result = (result << 7) | u64::from(byte & 0x7F);
        if byte & 0x80 == 0 {
            *pos = idx;
            return result;
        }
    }

    let byte = unsafe { *bytes.get_unchecked(idx) };
    idx += 1;
    *pos = idx;
    (result << 8) | u64::from(byte)
}

fn read_exact_bytes_at<'row>(bytes: &'row [u8], pos: &mut usize, len: usize) -> Result<&'row [u8]> {
    let start = *pos;
    let end = start + len;
    if end > bytes.len() {
        return Err(Error::Corrupted("record payload shorter than declared"));
    }
    *pos = end;
    Ok(unsafe { bytes.get_unchecked(start..end) })
}

#[derive(Clone, Copy, Debug)]
enum BTreeKind {
    TableLeaf,
    TableInterior,
}

struct BTreeHeader {
    kind: BTreeKind,
    cell_count: u16,
    cell_ptrs_start: usize,
    right_most_child: Option<u32>,
}

fn parse_header(page: &PageRef<'_>) -> Result<BTreeHeader> {
    let offset = page.offset();
    if offset >= page.usable_size() {
        return Err(Error::Corrupted("page header offset out of bounds"));
    }

    let bytes = page.usable_bytes();
    if offset + 8 > bytes.len() {
        return Err(Error::Corrupted("page header truncated"));
    }

    let page_type = bytes[offset];
    let _first_freeblock = u16::from_be_bytes([bytes[offset + 1], bytes[offset + 2]]);
    let cell_count = u16::from_be_bytes([bytes[offset + 3], bytes[offset + 4]]);
    let _start_of_cell_content = u16::from_be_bytes([bytes[offset + 5], bytes[offset + 6]]);
    let _fragmented_free_bytes = bytes[offset + 7];

    let kind = match page_type {
        0x0D => BTreeKind::TableLeaf,
        0x05 => BTreeKind::TableInterior,
        _ => return Err(Error::UnsupportedPageType(page_type)),
    };

    let right_most_child = match kind {
        BTreeKind::TableInterior => {
            if offset + 12 > bytes.len() {
                return Err(Error::Corrupted("page header truncated"));
            }
            Some(u32::from_be_bytes([
                bytes[offset + 8],
                bytes[offset + 9],
                bytes[offset + 10],
                bytes[offset + 11],
            ]))
        }
        BTreeKind::TableLeaf => None,
    };

    let header_size = match kind {
        BTreeKind::TableLeaf => 8,
        BTreeKind::TableInterior => 12,
    };
    let cell_ptrs_start =
        offset.checked_add(header_size).ok_or(Error::Corrupted("cell pointer array overflow"))?;

    Ok(BTreeHeader { kind, cell_count, cell_ptrs_start, right_most_child })
}

fn cell_ptrs<'a>(page: &'a PageRef<'_>, header: &BTreeHeader) -> Result<&'a [u8]> {
    let cell_ptrs_len = header.cell_count as usize * 2;
    let cell_ptrs_end = header
        .cell_ptrs_start
        .checked_add(cell_ptrs_len)
        .ok_or(Error::Corrupted("cell pointer array overflow"))?;
    let bytes = page.usable_bytes();
    if cell_ptrs_end > bytes.len() {
        return Err(Error::Corrupted("cell pointer array out of bounds"));
    }
    Ok(&bytes[header.cell_ptrs_start..cell_ptrs_end])
}

pub(crate) fn local_payload_len(usable_size: usize, payload_len: usize) -> Result<usize> {
    if payload_len == 0 {
        return Ok(0);
    }

    let u = usable_size;
    let x = u.checked_sub(35).ok_or(Error::Corrupted("usable size underflow"))?;
    let m_base = u.checked_sub(12).ok_or(Error::Corrupted("usable size underflow"))?;
    let m = (m_base * 32 / 255).saturating_sub(23);

    if payload_len <= x {
        return Ok(payload_len);
    }

    let k = m + ((payload_len - m) % (u - 4));
    if k <= x { Ok(k) } else { Ok(m) }
}
