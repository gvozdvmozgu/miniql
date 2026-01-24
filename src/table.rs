use std::{fmt, str};

use crate::decoder::Decoder;
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
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Pager(err) => Some(err),
            Self::Utf8(err) => Some(err),
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
}

impl RowScratch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(values: usize, overflow: usize) -> Self {
        Self { values: Vec::with_capacity(values), overflow_buf: Vec::with_capacity(overflow) }
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

pub fn scan_table_cells_with_scratch<F>(
    pager: &Pager,
    page_id: PageId,
    overflow_buf: &mut Vec<u8>,
    mut f: F,
) -> Result<()>
where
    F: for<'row> FnMut(i64, &'row [u8]) -> Result<()>,
{
    scan_table_page_cells(pager, page_id, overflow_buf, &mut f)
}

pub(crate) fn scan_table_cells_with_scratch_until<F, T>(
    pager: &Pager,
    page_id: PageId,
    overflow_buf: &mut Vec<u8>,
    mut f: F,
) -> Result<Option<T>>
where
    F: for<'row> FnMut(i64, &'row [u8]) -> Result<Option<T>>,
{
    scan_table_page_cells_until(pager, page_id, overflow_buf, &mut f)
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
    let mut stack = vec![page_id];
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
                    let offset = u16::from_be_bytes([cell_ptrs[idx * 2], cell_ptrs[idx * 2 + 1]]);
                    if offset as usize >= page_len {
                        return Err(Error::Corrupted("cell offset out of bounds"));
                    }

                    let mut decoder = Decoder::new(page.usable_bytes()).split_at(offset as usize);
                    let child = read_u32_checked(&mut decoder, "cell child pointer truncated")?;
                    let child =
                        PageId::try_new(child).ok_or(Error::Corrupted("child page id is zero"))?;

                    // Skip the key separating subtrees; it is not needed for scanning.
                    let _ = read_varint_checked(&mut decoder, "cell key truncated")?;
                    stack.push(child);
                }
            }
        }
    }

    Ok(())
}

fn scan_table_page_cells<'pager, F>(
    pager: &'pager Pager,
    page_id: PageId,
    overflow_buf: &mut Vec<u8>,
    f: &mut F,
) -> Result<()>
where
    F: for<'row> FnMut(i64, &'row [u8]) -> Result<()>,
{
    let mut stack = vec![page_id];
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
                    let (rowid, payload) =
                        read_table_cell_payload(pager, &page, offset, overflow_buf)?;
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
    let mut stack = vec![page_id];
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
                    let offset = u16::from_be_bytes([cell_ptrs[idx * 2], cell_ptrs[idx * 2 + 1]]);
                    if offset as usize >= page_len {
                        return Err(Error::Corrupted("cell offset out of bounds"));
                    }

                    let mut decoder = Decoder::new(page.usable_bytes()).split_at(offset as usize);
                    let child = read_u32_checked(&mut decoder, "cell child pointer truncated")?;
                    let child =
                        PageId::try_new(child).ok_or(Error::Corrupted("child page id is zero"))?;

                    // Skip the key separating subtrees; it is not needed for scanning.
                    let _ = read_varint_checked(&mut decoder, "cell key truncated")?;
                    stack.push(child);
                }
            }
        }
    }

    Ok(None)
}

fn scan_table_page_cells_until<'pager, F, T>(
    pager: &'pager Pager,
    page_id: PageId,
    overflow_buf: &mut Vec<u8>,
    f: &mut F,
) -> Result<Option<T>>
where
    F: for<'row> FnMut(i64, &'row [u8]) -> Result<Option<T>>,
{
    let mut stack = vec![page_id];
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
                    let (rowid, payload) =
                        read_table_cell_payload(pager, &page, offset, overflow_buf)?;
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
    let (values, overflow_buf) = scratch.split_mut();
    let (rowid, payload) = read_table_cell_payload(pager, page, offset, overflow_buf)?;
    decode_record_into(payload, values)?;
    Ok(rowid)
}

fn read_table_cell_payload<'row>(
    pager: &Pager,
    page: &'row PageRef<'_>,
    offset: u16,
    overflow_buf: &'row mut Vec<u8>,
) -> Result<(i64, &'row [u8])> {
    let usable = page.usable_bytes();
    if offset as usize >= usable.len() {
        return Err(Error::Corrupted("cell offset out of bounds"));
    }

    let mut decoder = Decoder::new(usable).split_at(offset as usize);

    let before = decoder.remaining();
    let payload_length = read_varint_checked(&mut decoder, "cell payload length truncated")?;
    let payload_length_len = before - decoder.remaining();
    let payload_length =
        usize::try_from(payload_length).map_err(|_| Error::Corrupted("payload is too large"))?;
    if payload_length > MAX_PAYLOAD_BYTES {
        return Err(Error::PayloadTooLarge(payload_length));
    }

    let before = decoder.remaining();
    let rowid = read_varint_checked(&mut decoder, "cell rowid truncated")? as i64;
    let rowid_len = before - decoder.remaining();

    let start = offset as usize + payload_length_len + rowid_len;
    let local_len = local_payload_len(page.usable_size(), payload_length)?;
    let end_local =
        start.checked_add(local_len).ok_or(Error::Corrupted("payload length overflow"))?;
    if end_local > usable.len() {
        return Err(Error::Corrupted("payload extends past page boundary"));
    }

    overflow_buf.clear();

    if payload_length <= local_len {
        let payload = &usable[start..start + payload_length];
        Ok((rowid, payload))
    } else {
        let overflow_end =
            end_local.checked_add(4).ok_or(Error::Corrupted("overflow pointer overflow"))?;
        if overflow_end > usable.len() {
            return Err(Error::Corrupted("overflow pointer out of bounds"));
        }
        let overflow_page = u32::from_be_bytes(usable[end_local..overflow_end].try_into().unwrap());
        if overflow_page == 0 {
            return Err(Error::OverflowChainTruncated);
        }
        assemble_overflow_payload(
            pager,
            payload_length,
            local_len,
            overflow_page,
            &usable[start..end_local],
            overflow_buf,
        )?;
        let payload = overflow_buf.as_slice();
        Ok((rowid, payload))
    }
}

fn assemble_overflow_payload(
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

pub(crate) fn decode_record_project_into(
    payload: &[u8],
    needed_cols: Option<&[u16]>,
    out: &mut Vec<ValueRefRaw>,
) -> Result<usize> {
    out.clear();

    let mut header_decoder = Decoder::new(payload);
    let before = header_decoder.remaining();
    let header_len = read_varint_checked(&mut header_decoder, "record header truncated")? as usize;
    let header_len_len = before - header_decoder.remaining();

    if header_len < header_len_len || header_len > payload.len() {
        return Err(Error::Corrupted("invalid record header length"));
    }

    let mut serial_decoder = Decoder::new(&payload[header_len_len..header_len]);
    let mut value_decoder = Decoder::new(&payload[header_len..]);

    if let Some(needed_cols) = needed_cols {
        let mut needed_iter = needed_cols.iter().copied();
        let mut next_needed = needed_iter.next();
        let mut column_count = 0usize;
        while serial_decoder.remaining() > 0 {
            let serial = read_varint_checked(&mut serial_decoder, "record header truncated")?;
            let col_idx = column_count as u16;
            if Some(col_idx) == next_needed {
                out.push(decode_value_ref(serial, &mut value_decoder)?);
                next_needed = needed_iter.next();
            } else {
                skip_value(serial, &mut value_decoder)?;
            }
            column_count += 1;
        }
        Ok(column_count)
    } else {
        let mut column_count = 0usize;
        while serial_decoder.remaining() > 0 {
            let serial = read_varint_checked(&mut serial_decoder, "record header truncated")?;
            out.push(decode_value_ref(serial, &mut value_decoder)?);
            column_count += 1;
        }
        Ok(column_count)
    }
}

fn decode_record_into(payload: &[u8], out: &mut Vec<ValueRefRaw>) -> Result<()> {
    out.clear();

    let mut header_decoder = Decoder::new(payload);
    let before = header_decoder.remaining();
    let header_len = read_varint_checked(&mut header_decoder, "record header truncated")? as usize;
    let header_len_len = before - header_decoder.remaining();

    if header_len < header_len_len || header_len > payload.len() {
        return Err(Error::Corrupted("invalid record header length"));
    }

    let mut serial_decoder = Decoder::new(&payload[header_len_len..header_len]);
    let mut value_decoder = Decoder::new(&payload[header_len..]);
    while serial_decoder.remaining() > 0 {
        let serial = read_varint_checked(&mut serial_decoder, "record header truncated")?;
        out.push(decode_value_ref(serial, &mut value_decoder)?);
    }

    Ok(())
}

fn skip_value<'row>(serial_type: u64, decoder: &mut Decoder<'row>) -> Result<()> {
    let len = serial_type_len(serial_type)?;
    if len > 0 {
        let _ = read_exact_bytes(decoder, len)?;
    }
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

fn decode_value_ref<'row>(serial_type: u64, decoder: &mut Decoder<'row>) -> Result<ValueRefRaw> {
    let value = match serial_type {
        0 => ValueRefRaw::Null,
        1 => ValueRefRaw::Integer(read_signed_be(decoder, 1)?),
        2 => ValueRefRaw::Integer(read_signed_be(decoder, 2)?),
        3 => ValueRefRaw::Integer(read_signed_be(decoder, 3)?),
        4 => ValueRefRaw::Integer(read_signed_be(decoder, 4)?),
        5 => ValueRefRaw::Integer(read_signed_be(decoder, 6)?),
        6 => ValueRefRaw::Integer(read_signed_be(decoder, 8)?),
        7 => ValueRefRaw::Real(f64::from_bits(read_u64_be(decoder)?)),
        8 => ValueRefRaw::Integer(0),
        9 => ValueRefRaw::Integer(1),
        serial if serial >= 12 && serial % 2 == 0 => {
            let len = ((serial - 12) / 2) as usize;
            ValueRefRaw::Blob(RawBytes::from_slice(read_exact_bytes(decoder, len)?))
        }
        serial if serial >= 13 => {
            let len = ((serial - 13) / 2) as usize;
            ValueRefRaw::TextBytes(RawBytes::from_slice(read_exact_bytes(decoder, len)?))
        }
        other => return Err(Error::UnsupportedSerialType(other)),
    };

    Ok(value)
}

fn read_signed_be(decoder: &mut Decoder<'_>, bytes: usize) -> Result<i64> {
    debug_assert!(bytes <= 8);

    let mut buf = [0u8; 8];
    let offset = 8 - bytes;
    buf[offset..].copy_from_slice(read_exact_bytes(decoder, bytes)?);

    let value = u64::from_be_bytes(buf);
    let shift = (8 - bytes) * 8;
    Ok(((value << shift) as i64) >> shift)
}

fn read_u64_be(decoder: &mut Decoder<'_>) -> Result<u64> {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(read_exact_bytes(decoder, 8)?);
    Ok(u64::from_be_bytes(buf))
}

fn read_u8_checked(decoder: &mut Decoder<'_>, msg: &'static str) -> Result<u8> {
    decoder.try_read_u8().ok_or(Error::Corrupted(msg))
}

fn read_u16_checked(decoder: &mut Decoder<'_>, msg: &'static str) -> Result<u16> {
    decoder.try_read_u16().ok_or(Error::Corrupted(msg))
}

fn read_u32_checked(decoder: &mut Decoder<'_>, msg: &'static str) -> Result<u32> {
    decoder.try_read_u32().ok_or(Error::Corrupted(msg))
}

fn read_varint_checked(decoder: &mut Decoder<'_>, msg: &'static str) -> Result<u64> {
    decoder.try_read_varint().ok_or(Error::Corrupted(msg))
}

fn read_exact_bytes<'row>(decoder: &mut Decoder<'row>, len: usize) -> Result<&'row [u8]> {
    decoder.try_read_bytes(len).ok_or(Error::Corrupted("record payload shorter than declared"))
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
    if page.offset() >= page.usable_size() {
        return Err(Error::Corrupted("page header offset out of bounds"));
    }
    let mut decoder = Decoder::new(page.usable_bytes()).split_at(page.offset());
    let page_type = read_u8_checked(&mut decoder, "page header truncated")?;
    let _first_freeblock = read_u16_checked(&mut decoder, "page header truncated")?;
    let cell_count = read_u16_checked(&mut decoder, "page header truncated")?;
    let _start_of_cell_content = read_u16_checked(&mut decoder, "page header truncated")?;
    let _fragmented_free_bytes = read_u8_checked(&mut decoder, "page header truncated")?;

    let kind = match page_type {
        0x0D => BTreeKind::TableLeaf,
        0x05 => BTreeKind::TableInterior,
        _ => return Err(Error::UnsupportedPageType(page_type)),
    };

    let right_most_child = match kind {
        BTreeKind::TableInterior => Some(read_u32_checked(&mut decoder, "page header truncated")?),
        BTreeKind::TableLeaf => None,
    };

    let header_size = match kind {
        BTreeKind::TableLeaf => 8,
        BTreeKind::TableInterior => 12,
    };
    let cell_ptrs_start = page
        .offset()
        .checked_add(header_size)
        .ok_or(Error::Corrupted("cell pointer array overflow"))?;

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

fn local_payload_len(usable_size: usize, payload_len: usize) -> Result<usize> {
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
