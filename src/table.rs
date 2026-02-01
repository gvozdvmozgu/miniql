use std::marker::PhantomData;
use std::{fmt, mem, ptr, str};

use crate::decoder::Decoder;
use crate::join::JoinError;
use crate::pager::{PageId, PageRef, Pager};

/// Result type for table operations.
pub type Result<T> = std::result::Result<T, Error>;

const MAX_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Corruption {
    LimitReached,
    AggregateGroupIndexOutOfBounds,
    AggregateGroupKeyIndexOutOfBounds,
    AggregateStateIndexOutOfBounds,
    BtreePageCycleDetected,
    CellKeyTruncated,
    CellChildPointerTruncated,
    CellOffsetOutOfBounds,
    CellPointerArrayOutOfBounds,
    CellPointerArrayOverflow,
    CellPayloadLengthTruncated,
    CellRowIdTruncated,
    ChildPageIdZero,
    CoveringIndexRequiresColumnProjection,
    CoveringIndexColumnMapMismatch,
    IndexCursorNotPositioned,
    IndexCursorPastEnd,
    IndexRecordHasNoColumns,
    InteriorIndexPageHasNoCells,
    InvalidRecordHeaderLength,
    MappedColumnIndexOutOfBounds,
    MissingChildPointer,
    MissingRightMostChildPointer,
    OverflowPageTooSmall,
    OverflowPageUsableSizeTooSmall,
    OverflowPointerOutOfBounds,
    PageHeaderOffsetOutOfBounds,
    PageHeaderTruncated,
    PayloadExtendsPastPageBoundary,
    PayloadIsTooLarge,
    PayloadLengthUnderflow,
    PredicateColumnNotDecoded,
    RecordHeaderTruncated,
    RecordPayloadShorterThanDeclared,
    RowColumnCountMismatch,
    ScanOverflowPayloadUnsupported,
    UsableSizeUnderflow,
}

impl Corruption {
    fn message(self) -> &'static str {
        match self {
            Self::LimitReached => "__limit_reached__",
            Self::AggregateGroupIndexOutOfBounds => "aggregate group index out of bounds",
            Self::AggregateGroupKeyIndexOutOfBounds => "aggregate group key index out of bounds",
            Self::AggregateStateIndexOutOfBounds => "aggregate state index out of bounds",
            Self::BtreePageCycleDetected => "btree page cycle detected",
            Self::CellKeyTruncated => "cell key truncated",
            Self::CellChildPointerTruncated => "cell child pointer truncated",
            Self::CellOffsetOutOfBounds => "cell offset out of bounds",
            Self::CellPointerArrayOutOfBounds => "cell pointer array out of bounds",
            Self::CellPointerArrayOverflow => "cell pointer array overflow",
            Self::CellPayloadLengthTruncated => "cell payload length truncated",
            Self::CellRowIdTruncated => "cell rowid truncated",
            Self::ChildPageIdZero => "child page id is zero",
            Self::CoveringIndexRequiresColumnProjection => {
                "covering index requires column projection"
            }
            Self::CoveringIndexColumnMapMismatch => {
                "covering index column map does not match needed columns"
            }
            Self::IndexCursorNotPositioned => "index cursor not positioned",
            Self::IndexCursorPastEnd => "index cursor past end",
            Self::IndexRecordHasNoColumns => "index record has no columns",
            Self::InteriorIndexPageHasNoCells => "interior index page has no cells",
            Self::InvalidRecordHeaderLength => "invalid record header length",
            Self::MappedColumnIndexOutOfBounds => "mapped column index out of bounds",
            Self::MissingChildPointer => "missing child pointer",
            Self::MissingRightMostChildPointer => "missing right-most child pointer",
            Self::OverflowPageTooSmall => "overflow page too small",
            Self::OverflowPageUsableSizeTooSmall => "overflow page usable size too small",
            Self::OverflowPointerOutOfBounds => "overflow pointer out of bounds",
            Self::PageHeaderOffsetOutOfBounds => "page header offset out of bounds",
            Self::PageHeaderTruncated => "page header truncated",
            Self::PayloadExtendsPastPageBoundary => "payload extends past page boundary",
            Self::PayloadIsTooLarge => "payload is too large",
            Self::PayloadLengthUnderflow => "payload length underflow",
            Self::PredicateColumnNotDecoded => "predicate column not decoded",
            Self::RecordHeaderTruncated => "record header truncated",
            Self::RecordPayloadShorterThanDeclared => "record payload shorter than declared",
            Self::RowColumnCountMismatch => "row column count does not match table schema",
            Self::ScanOverflowPayloadUnsupported => "scan does not support overflow payloads",
            Self::UsableSizeUnderflow => "usable size underflow",
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryError {
    AggregateProjectionEmpty,
    NonAggregateRequiresGroupBy,
    NonAggregateMustAppearInGroupBy,
    AggregateRequiresGroupByOrFunctions,
    AggregateExprOnlyColOrLit,
}

impl QueryError {
    fn message(self) -> &'static str {
        match self {
            Self::AggregateProjectionEmpty => "aggregate projection cannot be empty",
            Self::NonAggregateRequiresGroupBy => "non-aggregate expression requires GROUP BY",
            Self::NonAggregateMustAppearInGroupBy => {
                "non-aggregate expression must appear in GROUP BY"
            }
            Self::AggregateRequiresGroupByOrFunctions => {
                "aggregate query requires GROUP BY or aggregate functions"
            }
            Self::AggregateExprOnlyColOrLit => {
                "aggregate expressions support only column and literal operands"
            }
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScratchKind {
    Bytes,
    Serials,
    Offsets,
    Values,
}

impl ScratchKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Bytes => "bytes",
            Self::Serials => "serials",
            Self::Offsets => "offsets",
            Self::Values => "values",
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueKind {
    Null,
    Integer,
    Real,
    Text,
    Blob,
    Bytes,
    Missing,
}

impl ValueKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Null => "NULL",
            Self::Integer => "INTEGER",
            Self::Real => "REAL",
            Self::Text => "TEXT",
            Self::Blob => "BLOB",
            Self::Bytes => "BYTES",
            Self::Missing => "<missing>",
        }
    }
}

/// Table decoding and validation errors.
#[non_exhaustive]
#[derive(Debug)]
pub enum Error {
    Pager(crate::pager::Error),
    UnsupportedPageType(u8),
    UnsupportedSerialType(u64),
    UnsupportedSerialTypeAt { col: usize, serial: u64 },
    Corrupted(Corruption),
    Query(QueryError),
    InvalidColumnIndex { col: u16, column_count: usize },
    ScratchTooSmall { kind: ScratchKind, needed: usize, capacity: usize },
    TypeMismatch { col: usize, expected: ValueKind, got: ValueKind },
    TypeMismatchSerial { col: usize, expected: ValueKind, got_serial: u64 },
    SchemaMismatch { expected: usize, got: usize },
    Utf8(str::Utf8Error),
    InvalidUtf8 { col: usize, err: str::Utf8Error },
    TableNotFound,
    PayloadTooLarge(usize),
    OverflowChainTruncated,
    OverflowLoopDetected,
    Join(JoinError),
    RowDecode { rowid: i64, page_id: PageId, source: Box<Error> },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pager(err) => write!(f, "{err}"),
            Self::UnsupportedPageType(kind) => write!(f, "Unsupported page type: 0x{kind:02X}"),
            Self::UnsupportedSerialType(serial) => {
                write!(f, "Unsupported record serial type: {serial}")
            }
            Self::UnsupportedSerialTypeAt { col, serial } => {
                write!(f, "Unsupported record serial type at column {col}: {serial}")
            }
            Self::Corrupted(kind) => write!(f, "Corrupted table page: {}", kind.message()),
            Self::Query(kind) => write!(f, "Query error: {}", kind.message()),
            Self::InvalidColumnIndex { col, column_count } => {
                write!(f, "Invalid column index {col} (column count {column_count})")
            }
            Self::ScratchTooSmall { kind, needed, capacity } => {
                write!(
                    f,
                    "Scratch buffer too small ({}): need {needed}, capacity {capacity}",
                    kind.as_str()
                )
            }
            Self::TypeMismatch { col, expected, got } => {
                write!(
                    f,
                    "Type mismatch at column {col}: expected {}, got {}",
                    expected.as_str(),
                    got.as_str()
                )
            }
            Self::TypeMismatchSerial { col, expected, got_serial } => {
                if let Some(kind) = serial_type_kind(*got_serial) {
                    write!(
                        f,
                        "Type mismatch at column {col}: expected {}, got {} (serial {got_serial})",
                        expected.as_str(),
                        kind.as_str(),
                    )
                } else {
                    write!(
                        f,
                        "Type mismatch at column {col}: expected {}, got serial {got_serial}",
                        expected.as_str()
                    )
                }
            }
            Self::SchemaMismatch { expected, got } => {
                write!(f, "Schema mismatch: expected {expected} columns, got {got}")
            }
            Self::Utf8(err) => write!(f, "{err}"),
            Self::InvalidUtf8 { col, err } => {
                write!(f, "Invalid UTF-8 at column {col}: {err}")
            }
            Self::TableNotFound => f.write_str("Table not found in sqlite_schema"),
            Self::PayloadTooLarge(size) => write!(f, "Payload too large: {size} bytes"),
            Self::OverflowChainTruncated => f.write_str("Overflow chain is truncated"),
            Self::OverflowLoopDetected => f.write_str("Overflow chain contains a loop"),
            Self::Join(err) => write!(f, "{err}"),
            Self::RowDecode { rowid, page_id, source } => {
                write!(
                    f,
                    "Row decode error (rowid {rowid}, page {}): {source}",
                    page_id.into_inner()
                )
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Pager(err) => Some(err),
            Self::Utf8(err) => Some(err),
            Self::InvalidUtf8 { err, .. } => Some(err),
            Self::Join(err) => Some(err),
            Self::RowDecode { source, .. } => Some(source),
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

fn display_blob(bytes: &[u8], f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("x'")?;
    for byte in bytes {
        write!(f, "{byte:02x}")?;
    }
    f.write_str("'")
}

/// Borrowed value reference into a row payload.
#[derive(Debug, Clone, Copy)]
pub enum ValueRef<'row> {
    Null,
    Integer(i64),
    Real(f64),
    Text(&'row [u8]),
    Blob(&'row [u8]),
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RawBytes {
    ptr: *const u8,
    len: usize,
}

impl RawBytes {
    #[inline]
    pub(crate) fn from_slice(bytes: &[u8]) -> Self {
        Self { ptr: bytes.as_ptr(), len: bytes.len() }
    }

    #[inline]
    unsafe fn as_slice<'row>(self) -> &'row [u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum BytesSpan {
    Mmap(RawBytes),
    Scratch(RawBytes),
}

impl BytesSpan {
    #[inline]
    fn mmap(bytes: &[u8]) -> Self {
        Self::Mmap(RawBytes::from_slice(bytes))
    }

    #[inline]
    fn scratch(bytes: &[u8]) -> Self {
        Self::Scratch(RawBytes::from_slice(bytes))
    }

    #[inline]
    unsafe fn as_slice<'row>(self) -> &'row [u8] {
        match self {
            Self::Mmap(raw) | Self::Scratch(raw) => unsafe { raw.as_slice() },
        }
    }

    #[inline]
    unsafe fn as_slice_with_scratch(self, scratch_bytes: &[u8]) -> &[u8] {
        match self {
            Self::Mmap(raw) => unsafe { raw.as_slice() },
            Self::Scratch(raw) => {
                let base = scratch_bytes.as_ptr() as usize;
                let ptr = raw.ptr as usize;
                let end = ptr.saturating_add(raw.len);
                let limit = base.saturating_add(scratch_bytes.len());
                debug_assert!(ptr >= base && end <= limit);
                unsafe { raw.as_slice() }
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ValueSlot {
    Null,
    Integer(i64),
    Real(f64),
    Text(BytesSpan),
    Blob(BytesSpan),
}

/// Reference to a record payload.
#[derive(Clone, Copy)]
pub enum PayloadRef<'row> {
    Inline(&'row [u8]),
    Overflow(OverflowPayload<'row>),
}

/// Reference to a table cell (rowid + payload).
#[derive(Clone, Copy)]
pub struct CellRef<'row> {
    rowid: i64,
    payload: PayloadRef<'row>,
    page_id: PageId,
    cell_offset: u16,
}

impl<'row> CellRef<'row> {
    #[inline]
    /// Return the rowid for this cell.
    pub fn rowid(self) -> i64 {
        self.rowid
    }

    #[inline]
    /// Return the payload reference for this cell.
    pub fn payload(self) -> PayloadRef<'row> {
        self.payload
    }

    #[inline]
    /// Return the page id containing this cell.
    pub fn page_id(self) -> PageId {
        self.page_id
    }

    #[inline]
    /// Return the cell offset within the page.
    pub fn cell_offset(self) -> u16 {
        self.cell_offset
    }
}

/// Overflow payload descriptor for large rows.
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

    /// Returns the local (inline) portion of the overflow payload.
    #[inline]
    pub fn local(&self) -> &'row [u8] {
        self.local
    }
}

impl<'row> PayloadRef<'row> {
    /// Materialize the payload into a contiguous buffer.
    pub fn to_vec(&self) -> Result<Vec<u8>> {
        match self {
            PayloadRef::Inline(bytes) => Ok(bytes.to_vec()),
            PayloadRef::Overflow(payload) => {
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

    /// Return the total payload length in bytes.
    pub fn len(&self) -> usize {
        match self {
            PayloadRef::Inline(bytes) => bytes.len(),
            PayloadRef::Overflow(payload) => payload.total_len,
        }
    }

    /// Returns true when the payload is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl ValueSlot {
    #[inline]
    pub(crate) unsafe fn as_value_ref<'row>(self) -> ValueRef<'row> {
        match self {
            Self::Null => ValueRef::Null,
            Self::Integer(value) => ValueRef::Integer(value),
            Self::Real(value) => ValueRef::Real(value),
            Self::Text(bytes) => ValueRef::Text(unsafe { bytes.as_slice() }),
            Self::Blob(bytes) => ValueRef::Blob(unsafe { bytes.as_slice() }),
        }
    }

    #[inline]
    pub(crate) unsafe fn as_value_ref_with_scratch<'row>(
        self,
        scratch_bytes: &'row [u8],
    ) -> ValueRef<'row> {
        match self {
            Self::Null => ValueRef::Null,
            Self::Integer(value) => ValueRef::Integer(value),
            Self::Real(value) => ValueRef::Real(value),
            Self::Text(bytes) => {
                ValueRef::Text(unsafe { bytes.as_slice_with_scratch(scratch_bytes) })
            }
            Self::Blob(bytes) => {
                ValueRef::Blob(unsafe { bytes.as_slice_with_scratch(scratch_bytes) })
            }
        }
    }
}

impl<'row> ValueRef<'row> {
    /// Return the value as UTF-8 text.
    pub fn as_text(&self) -> Option<&'row str> {
        match self {
            Self::Text(bytes) => str::from_utf8(bytes).ok(),
            _ => None,
        }
    }

    /// Return the raw text bytes.
    pub fn text_bytes(&self) -> Option<&'row [u8]> {
        match self {
            Self::Text(bytes) => Some(*bytes),
            _ => None,
        }
    }

    /// Return the value as an integer.
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
            Self::Text(bytes) => match str::from_utf8(bytes) {
                Ok(value) => f.write_str(value),
                Err(_) => f.write_str("<invalid utf8>"),
            },
            Self::Blob(bytes) => display_blob(bytes, f),
        }
    }
}

/// Row view that decodes values on demand.
///
/// This is more efficient than decoding entire rows when you only need to
/// access a subset of columns or just want to count rows without reading data.
/// For repeated column access, use `RowView::cached` with `RowCache` to avoid
/// re-scanning the header per column.
#[derive(Clone, Copy)]
pub struct RowView<'row> {
    payload: &'row [u8],
    header_len: usize,
    serial_bytes: &'row [u8],
}

impl<'row> RowView<'row> {
    /// Create a row view from inline payload bytes.
    #[inline]
    pub fn from_inline(payload: &'row [u8]) -> Result<Self> {
        let mut header_pos = 0usize;
        let first = *payload.first().ok_or(Error::Corrupted(Corruption::RecordHeaderTruncated))?;
        let header_len = if first < 0x80 {
            header_pos = 1;
            first as usize
        } else {
            read_varint_at(payload, &mut header_pos, Corruption::RecordHeaderTruncated)? as usize
        };
        if header_len < header_pos || header_len > payload.len() {
            return Err(Error::Corrupted(Corruption::InvalidRecordHeaderLength));
        }

        let serial_bytes = &payload[header_pos..header_len];
        Ok(Self { payload, header_len, serial_bytes })
    }

    /// Prepare a cached view for repeated column access without re-scanning
    /// headers.
    #[inline]
    pub fn cached<'cache>(
        &self,
        cache: &'cache mut RowCache,
    ) -> Result<CachedRowView<'row, 'cache>> {
        cache.ensure(self)?;
        Ok(CachedRowView { row: *self, cache })
    }

    /// Number of columns in the row.
    #[inline]
    pub fn column_count(&self) -> usize {
        // Count varints in serial_bytes
        let mut count = 0usize;
        let mut pos = 0usize;
        while pos < self.serial_bytes.len() {
            let b = unsafe { *self.serial_bytes.get_unchecked(pos) };
            if b < 0x80 {
                pos += 1;
            } else {
                // Skip multi-byte varint
                while pos < self.serial_bytes.len() {
                    let byte = unsafe { *self.serial_bytes.get_unchecked(pos) };
                    pos += 1;
                    if byte & 0x80 == 0 {
                        break;
                    }
                }
            }
            count += 1;
        }
        count
    }

    /// Decode and return a single column value by index.
    #[inline]
    pub fn get(&self, col_idx: usize) -> Result<Option<ValueRef<'row>>> {
        // Parse serial types up to col_idx
        let mut serial_pos = 0usize;
        let mut value_pos = self.header_len;
        let mut current_col = 0usize;

        while serial_pos < self.serial_bytes.len() {
            let b = unsafe { *self.serial_bytes.get_unchecked(serial_pos) };
            let serial = if b < 0x80 {
                serial_pos += 1;
                b as u64
            } else {
                read_varint_at(
                    self.serial_bytes,
                    &mut serial_pos,
                    Corruption::RecordHeaderTruncated,
                )?
            };

            if current_col == col_idx {
                // Decode this column
                return Ok(Some(decode_value_ref_inline(serial, self.payload, value_pos)?));
            }

            // Skip this column's value
            value_pos += serial_type_len_fast(serial);
            current_col += 1;
        }

        Ok(None)
    }

    /// Get an i64 value or return a type mismatch error.
    #[inline]
    pub fn get_i64(&self, col_idx: usize) -> Result<i64> {
        match self.get(col_idx)? {
            Some(ValueRef::Integer(v)) => Ok(v),
            Some(other) => Err(Error::TypeMismatch {
                col: col_idx,
                expected: ValueKind::Integer,
                got: value_ref_kind(other),
            }),
            None => Err(Error::InvalidColumnIndex {
                col: col_idx as u16,
                column_count: self.column_count(),
            }),
        }
    }

    /// Get a text value or return a type mismatch error.
    #[inline]
    pub fn get_text(&self, col_idx: usize) -> Result<&'row str> {
        match self.get(col_idx)? {
            Some(ValueRef::Text(bytes)) => Ok(str::from_utf8(bytes)?),
            Some(other) => Err(Error::TypeMismatch {
                col: col_idx,
                expected: ValueKind::Text,
                got: value_ref_kind(other),
            }),
            None => Err(Error::InvalidColumnIndex {
                col: col_idx as u16,
                column_count: self.column_count(),
            }),
        }
    }

    /// Get raw bytes (text or blob) or return a type mismatch error.
    #[inline]
    pub fn get_bytes(&self, col_idx: usize) -> Result<&'row [u8]> {
        match self.get(col_idx)? {
            Some(ValueRef::Text(bytes)) | Some(ValueRef::Blob(bytes)) => Ok(bytes),
            Some(other) => Err(Error::TypeMismatch {
                col: col_idx,
                expected: ValueKind::Bytes,
                got: value_ref_kind(other),
            }),
            None => Err(Error::InvalidColumnIndex {
                col: col_idx as u16,
                column_count: self.column_count(),
            }),
        }
    }
}

/// Cached header offsets/types for a row view.
#[derive(Debug, Default)]
pub struct RowCache {
    serials: Vec<u64>,
    offsets: Vec<u32>,
    payload_ptr: *const u8,
    payload_len: usize,
}

impl RowCache {
    /// Create an empty row cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a row cache with column capacity.
    pub fn with_capacity(cols: usize) -> Self {
        Self {
            serials: Vec::with_capacity(cols),
            offsets: Vec::with_capacity(cols),
            payload_ptr: ptr::null(),
            payload_len: 0,
        }
    }

    #[inline]
    fn matches(&self, row: &RowView<'_>) -> bool {
        self.payload_ptr == row.payload.as_ptr() && self.payload_len == row.payload.len()
    }

    #[inline]
    fn ensure(&mut self, row: &RowView<'_>) -> Result<()> {
        if self.matches(row) {
            return Ok(());
        }
        self.fill(row)
    }

    fn fill(&mut self, row: &RowView<'_>) -> Result<()> {
        self.serials.clear();
        self.offsets.clear();

        let mut serial_pos = 0usize;
        let mut value_pos = row.header_len;
        while serial_pos < row.serial_bytes.len() {
            let b = unsafe { *row.serial_bytes.get_unchecked(serial_pos) };
            let serial = if b < 0x80 {
                serial_pos += 1;
                b as u64
            } else {
                read_varint_at(
                    row.serial_bytes,
                    &mut serial_pos,
                    Corruption::RecordHeaderTruncated,
                )?
            };

            self.serials.push(serial);
            self.offsets.push(value_pos as u32);
            value_pos += serial_type_len_fast(serial);
        }

        self.payload_ptr = row.payload.as_ptr();
        self.payload_len = row.payload.len();
        Ok(())
    }
}

/// Row view backed by a cached header for repeated column access.
pub struct CachedRowView<'row, 'cache> {
    row: RowView<'row>,
    cache: &'cache RowCache,
}

impl<'row, 'cache> CachedRowView<'row, 'cache> {
    /// Number of columns in the row.
    #[inline]
    pub fn column_count(&self) -> usize {
        self.cache.serials.len()
    }

    /// Decode and return a single column value by index using cached offsets.
    #[inline]
    pub fn get(&self, col_idx: usize) -> Result<Option<ValueRef<'row>>> {
        let serial = match self.cache.serials.get(col_idx) {
            Some(serial) => *serial,
            None => return Ok(None),
        };
        let offset = self.cache.offsets[col_idx] as usize;
        Ok(Some(decode_value_ref_inline(serial, self.row.payload, offset)?))
    }

    /// Get an i64 value or return a type mismatch error.
    #[inline]
    pub fn get_i64(&self, col_idx: usize) -> Result<i64> {
        match self.get(col_idx)? {
            Some(ValueRef::Integer(v)) => Ok(v),
            Some(other) => Err(Error::TypeMismatch {
                col: col_idx,
                expected: ValueKind::Integer,
                got: value_ref_kind(other),
            }),
            None => Err(Error::InvalidColumnIndex {
                col: col_idx as u16,
                column_count: self.column_count(),
            }),
        }
    }

    /// Get a text value or return a type mismatch error.
    #[inline]
    pub fn get_text(&self, col_idx: usize) -> Result<&'row str> {
        match self.get(col_idx)? {
            Some(ValueRef::Text(bytes)) => Ok(str::from_utf8(bytes)?),
            Some(other) => Err(Error::TypeMismatch {
                col: col_idx,
                expected: ValueKind::Text,
                got: value_ref_kind(other),
            }),
            None => Err(Error::InvalidColumnIndex {
                col: col_idx as u16,
                column_count: self.column_count(),
            }),
        }
    }

    /// Get raw bytes (text or blob) or return a type mismatch error.
    #[inline]
    pub fn get_bytes(&self, col_idx: usize) -> Result<&'row [u8]> {
        match self.get(col_idx)? {
            Some(ValueRef::Text(bytes)) | Some(ValueRef::Blob(bytes)) => Ok(bytes),
            Some(other) => Err(Error::TypeMismatch {
                col: col_idx,
                expected: ValueKind::Bytes,
                got: value_ref_kind(other),
            }),
            None => Err(Error::InvalidColumnIndex {
                col: col_idx as u16,
                column_count: self.column_count(),
            }),
        }
    }
}

#[inline]
fn value_ref_kind(value: ValueRef<'_>) -> ValueKind {
    match value {
        ValueRef::Null => ValueKind::Null,
        ValueRef::Integer(_) => ValueKind::Integer,
        ValueRef::Real(_) => ValueKind::Real,
        ValueRef::Text(_) => ValueKind::Text,
        ValueRef::Blob(_) => ValueKind::Blob,
    }
}

#[inline]
fn serial_type_kind(serial: u64) -> Option<ValueKind> {
    match serial {
        0 => Some(ValueKind::Null),
        1 | 2 | 3 | 4 | 5 | 6 | 8 | 9 => Some(ValueKind::Integer),
        7 => Some(ValueKind::Real),
        10 | 11 => None,
        serial if serial >= 12 => {
            if serial & 1 == 0 {
                Some(ValueKind::Blob)
            } else {
                Some(ValueKind::Text)
            }
        }
        _ => None,
    }
}

/// Decode a value directly from inline payload (no intermediate ValueSlot).
#[inline]
fn decode_value_ref_inline(
    serial_type: u64,
    payload: &[u8],
    mut pos: usize,
) -> Result<ValueRef<'_>> {
    Ok(match serial_type {
        0 => ValueRef::Null,
        8 => ValueRef::Integer(0),
        9 => ValueRef::Integer(1),
        1 => ValueRef::Integer(read_signed_be_at(payload, &mut pos, 1)?),
        2 => ValueRef::Integer(read_signed_be_at(payload, &mut pos, 2)?),
        3 => ValueRef::Integer(read_signed_be_at(payload, &mut pos, 3)?),
        4 => ValueRef::Integer(read_signed_be_at(payload, &mut pos, 4)?),
        5 => ValueRef::Integer(read_signed_be_at(payload, &mut pos, 6)?),
        6 => ValueRef::Integer(read_signed_be_at(payload, &mut pos, 8)?),
        7 => ValueRef::Real(f64::from_bits(read_u64_be_at(payload, &mut pos)?)),
        10 | 11 => return Err(Error::UnsupportedSerialType(serial_type)),
        serial => {
            let len = ((serial - 12) / 2) as usize;
            let slice = read_exact_bytes_at(payload, &mut pos, len)?;
            if serial & 1 == 0 { ValueRef::Blob(slice) } else { ValueRef::Text(slice) }
        }
    })
}

/// Marker type for NULL fields in typed decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Null;

/// Borrow policy for typed decoding of TEXT/BLOB fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorrowPolicy {
    /// Borrow from the record payload when possible.
    PreferBorrow,
    /// Always copy TEXT/BLOB fields into scratch storage.
    AlwaysCopy,
}

/// Column count strictness for typed decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnMode {
    /// Require an exact column count match.
    Strict,
    /// Allow extra columns beyond the decoded schema.
    AllowExtraColumns,
}

/// Options for typed table scans.
#[derive(Debug, Clone, Copy)]
pub struct TypedScanOptions {
    pub borrow_policy: BorrowPolicy,
    pub column_mode: ColumnMode,
}

impl Default for TypedScanOptions {
    fn default() -> Self {
        Self { borrow_policy: BorrowPolicy::PreferBorrow, column_mode: ColumnMode::Strict }
    }
}

impl TypedScanOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn borrow_policy(mut self, policy: BorrowPolicy) -> Self {
        self.borrow_policy = policy;
        self
    }

    pub fn column_mode(mut self, mode: ColumnMode) -> Self {
        self.column_mode = mode;
        self
    }
}

/// Source bytes for a decoded field.
#[derive(Debug, Clone, Copy)]
pub struct FieldSource<'row> {
    col: usize,
    span: BytesSpan,
    scratch: RawBytes,
    _marker: PhantomData<&'row [u8]>,
}

impl<'row> FieldSource<'row> {
    #[inline]
    fn new(col: usize, span: BytesSpan, scratch: RawBytes) -> Self {
        Self { col, span, scratch, _marker: PhantomData }
    }

    /// Column index (0-based).
    #[inline]
    pub fn column(&self) -> usize {
        self.col
    }

    /// Return the raw field bytes.
    #[inline]
    pub fn bytes(&self) -> &'row [u8] {
        let scratch = unsafe { self.scratch.as_slice() };
        unsafe { self.span.as_slice_with_scratch(scratch) }
    }

    #[inline]
    pub fn type_mismatch(&self, expected: ValueKind, serial: u64) -> Error {
        Error::TypeMismatchSerial { col: self.col, expected, got_serial: serial }
    }

    #[inline]
    pub fn invalid_utf8(&self, err: str::Utf8Error) -> Error {
        Error::InvalidUtf8 { col: self.col, err }
    }

    #[inline]
    pub fn unsupported_serial(&self, serial: u64) -> Error {
        Error::UnsupportedSerialTypeAt { col: self.col, serial }
    }
}

/// Decode a single field from a record.
pub trait DecodeField<'row>: Sized {
    fn decode(serial_type: u64, source: FieldSource<'row>) -> Result<Self>;
}

/// Decode a typed row from a record.
pub trait DecodeRecord {
    type Row<'row>;
    const COLS: usize;
    fn decode<'row>(decoder: &mut RecordDecoder<'row>) -> Result<Self::Row<'row>>;
}

#[derive(Default)]
struct RecordScratch {
    serials: Vec<u64>,
    bytes: Vec<u8>,
}

enum RecordCursor<'row> {
    Inline { payload: &'row [u8], pos: usize },
    Overflow { cursor: OverflowCursor<'row> },
}

enum SerialMode<'row> {
    Collected,
    InlineStream { serial_bytes: &'row [u8], serial_pos: usize },
}

/// Typed record decoder that validates serial types and decodes directly into
/// Rust types.
pub struct RecordDecoder<'row> {
    serials: Vec<u64>,
    scratch: Vec<u8>,
    cursor: RecordCursor<'row>,
    serial_mode: SerialMode<'row>,
    idx: usize,
    column_count: Option<usize>,
    borrow_policy: BorrowPolicy,
    expected_cols: usize,
}

impl<'row> RecordDecoder<'row> {
    fn new_with_scratch(
        payload: PayloadRef<'row>,
        mut scratch: RecordScratch,
        options: TypedScanOptions,
        expected_cols: usize,
    ) -> Result<Self> {
        const INLINE_FAST_COLS: usize = 8;

        scratch.serials.clear();
        scratch.bytes.clear();

        let (total_value_len, text_blob_len, cursor, serial_mode, column_count) = match payload {
            PayloadRef::Inline(bytes)
                if options.borrow_policy == BorrowPolicy::PreferBorrow
                    && expected_cols <= INLINE_FAST_COLS =>
            {
                let (header_len, serial_bytes) = parse_record_header_inline(bytes)?;
                let cursor = RecordCursor::Inline { payload: bytes, pos: header_len };
                (
                    0usize,
                    0usize,
                    cursor,
                    SerialMode::InlineStream { serial_bytes, serial_pos: 0 },
                    None,
                )
            }
            PayloadRef::Inline(bytes) => {
                let (header_len, total_value_len, text_blob_len) =
                    parse_record_header_bytes(bytes, &mut scratch.serials)?;
                let cursor = RecordCursor::Inline { payload: bytes, pos: header_len };
                (
                    total_value_len,
                    text_blob_len,
                    cursor,
                    SerialMode::Collected,
                    Some(scratch.serials.len()),
                )
            }
            PayloadRef::Overflow(payload) => {
                let (cursor, total_value_len, text_blob_len) =
                    parse_record_header_overflow(payload, &mut scratch.serials)?;
                let cursor = RecordCursor::Overflow { cursor };
                (
                    total_value_len,
                    text_blob_len,
                    cursor,
                    SerialMode::Collected,
                    Some(scratch.serials.len()),
                )
            }
        };

        let scratch_needed = match payload {
            PayloadRef::Inline(_) => match options.borrow_policy {
                BorrowPolicy::PreferBorrow => 0,
                BorrowPolicy::AlwaysCopy => text_blob_len,
            },
            PayloadRef::Overflow(_) => total_value_len,
        };
        if scratch.bytes.capacity() < scratch_needed {
            scratch.bytes.reserve(scratch_needed - scratch.bytes.capacity());
        }

        Ok(Self {
            serials: scratch.serials,
            scratch: scratch.bytes,
            cursor,
            serial_mode,
            idx: 0,
            column_count,
            borrow_policy: options.borrow_policy,
            expected_cols,
        })
    }

    fn into_scratch(self) -> RecordScratch {
        RecordScratch { serials: self.serials, bytes: self.scratch }
    }

    /// Number of columns in the record.
    #[inline]
    pub fn column_count(&self) -> usize {
        if let Some(count) = self.column_count {
            return count;
        }
        match &self.serial_mode {
            SerialMode::Collected => self.serials.len(),
            SerialMode::InlineStream { serial_bytes, .. } => {
                count_serials(serial_bytes).unwrap_or(0)
            }
        }
    }

    /// Number of remaining columns.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.column_count().saturating_sub(self.idx)
    }

    /// Read and decode the next column.
    pub fn read<T: DecodeField<'row>>(&mut self) -> Result<T> {
        let (col, serial) = self.next_serial()?;

        let len = serial_type_len_checked(serial, col)?;
        let span = match &mut self.cursor {
            RecordCursor::Inline { payload, pos } => {
                let slice = read_exact_bytes_at(payload, pos, len)?;
                BytesSpan::mmap(slice)
            }
            RecordCursor::Overflow { cursor } => cursor.take_span(len, &mut self.scratch)?,
        };

        let span = if self.borrow_policy == BorrowPolicy::AlwaysCopy && serial >= 12 {
            self.copy_span_to_scratch(span, len)?
        } else {
            span
        };

        let source = FieldSource::new(col, span, RawBytes::from_slice(&self.scratch));
        T::decode(serial, source)
    }

    /// Read and decode the next column as an i64 without allocating
    /// `FieldSource`.
    pub fn read_i64(&mut self) -> Result<i64> {
        let (col, serial) = self.next_serial()?;
        match serial {
            1..=6 => {
                let len = match serial {
                    1 => 1,
                    2 => 2,
                    3 => 3,
                    4 => 4,
                    5 => 6,
                    _ => 8,
                };
                match &mut self.cursor {
                    RecordCursor::Inline { payload, pos } => read_signed_be_at(payload, pos, len),
                    RecordCursor::Overflow { cursor } => cursor.read_signed_be(len),
                }
            }
            8 => Ok(0),
            9 => Ok(1),
            10 | 11 => Err(Error::UnsupportedSerialTypeAt { col, serial }),
            _ => Err(Error::TypeMismatchSerial {
                col,
                expected: ValueKind::Integer,
                got_serial: serial,
            }),
        }
    }

    /// Read and decode the next column as UTF-8 text without allocating
    /// `FieldSource`.
    pub fn read_text(&mut self) -> Result<&'row str> {
        let (col, serial) = self.next_serial()?;
        match serial {
            10 | 11 => Err(Error::UnsupportedSerialTypeAt { col, serial }),
            serial if serial >= 13 && serial & 1 == 1 => {
                let len = ((serial - 13) / 2) as usize;
                let span = self.read_span(len)?;
                let span = if self.borrow_policy == BorrowPolicy::AlwaysCopy {
                    self.copy_span_to_scratch(span, len)?
                } else {
                    span
                };
                let scratch = RawBytes::from_slice(&self.scratch);
                let scratch = unsafe { scratch.as_slice() };
                let bytes = unsafe { span.as_slice_with_scratch(scratch) };
                str::from_utf8(bytes).map_err(|err| Error::InvalidUtf8 { col, err })
            }
            _ => Err(Error::TypeMismatchSerial {
                col,
                expected: ValueKind::Text,
                got_serial: serial,
            }),
        }
    }

    /// Skip the next column without decoding.
    pub fn skip(&mut self) -> Result<()> {
        let (col, serial) = self.next_serial()?;

        let len = serial_type_len_checked(serial, col)?;
        match &mut self.cursor {
            RecordCursor::Inline { payload, pos } => {
                let end = pos.saturating_add(len);
                if end > payload.len() {
                    return Err(Error::Corrupted(Corruption::RecordPayloadShorterThanDeclared));
                }
                *pos = end;
            }
            RecordCursor::Overflow { cursor } => {
                cursor.skip(len)?;
            }
        }
        Ok(())
    }

    fn skip_remaining(&mut self) -> Result<()> {
        while self.has_remaining_serials() {
            self.skip()?;
        }
        Ok(())
    }

    fn ensure_column_count(&mut self, expected: usize, mode: ColumnMode) -> Result<()> {
        let Some(count) = self.column_count else {
            return Ok(());
        };
        match mode {
            ColumnMode::Strict => {
                if count != expected {
                    return Err(Error::SchemaMismatch { expected, got: count });
                }
            }
            ColumnMode::AllowExtraColumns => {
                if count < expected {
                    return Err(Error::SchemaMismatch { expected, got: count });
                }
            }
        }
        Ok(())
    }

    fn finish(&mut self, expected: usize, mode: ColumnMode) -> Result<()> {
        if self.idx != expected {
            return Err(Error::SchemaMismatch { expected, got: self.idx });
        }
        if matches!(mode, ColumnMode::AllowExtraColumns) {
            self.skip_remaining()?;
        } else if let SerialMode::InlineStream { serial_bytes, serial_pos } = &self.serial_mode
            && *serial_pos < serial_bytes.len()
        {
            let got = count_serials(serial_bytes).unwrap_or(self.idx + 1);
            return Err(Error::SchemaMismatch { expected, got });
        }
        Ok(())
    }

    fn copy_span_to_scratch(&mut self, span: BytesSpan, len: usize) -> Result<BytesSpan> {
        if matches!(span, BytesSpan::Scratch(_)) {
            return Ok(span);
        }

        ensure_bytes_capacity(&self.scratch, len)?;
        let start = self.scratch.len();
        let bytes = unsafe { span.as_slice() };
        self.scratch.extend_from_slice(bytes);
        let slice = &self.scratch[start..start + len];
        Ok(BytesSpan::scratch(slice))
    }

    fn read_span(&mut self, len: usize) -> Result<BytesSpan> {
        match &mut self.cursor {
            RecordCursor::Inline { payload, pos } => {
                let slice = read_exact_bytes_at(payload, pos, len)?;
                Ok(BytesSpan::mmap(slice))
            }
            RecordCursor::Overflow { cursor } => cursor.take_span(len, &mut self.scratch),
        }
    }

    fn has_remaining_serials(&self) -> bool {
        match &self.serial_mode {
            SerialMode::Collected => self.idx < self.serials.len(),
            SerialMode::InlineStream { serial_bytes, serial_pos } => {
                *serial_pos < serial_bytes.len()
            }
        }
    }

    fn next_serial(&mut self) -> Result<(usize, u64)> {
        let col = self.idx;
        match &mut self.serial_mode {
            SerialMode::Collected => {
                let Some(serial) = self.serials.get(col) else {
                    return Err(Error::SchemaMismatch { expected: self.expected_cols, got: col });
                };
                self.idx += 1;
                Ok((col, *serial))
            }
            SerialMode::InlineStream { serial_bytes, serial_pos } => {
                if *serial_pos >= serial_bytes.len() {
                    return Err(Error::SchemaMismatch { expected: self.expected_cols, got: col });
                }
                let pos = *serial_pos;
                let b0 = unsafe { *serial_bytes.get_unchecked(pos) };
                let serial = if b0 < 0x80 {
                    *serial_pos = pos + 1;
                    u64::from(b0)
                } else {
                    read_varint_at(serial_bytes, serial_pos, Corruption::RecordHeaderTruncated)?
                };
                self.idx += 1;
                Ok((col, serial))
            }
        }
    }
}

impl<'row> DecodeField<'row> for Null {
    fn decode(serial_type: u64, source: FieldSource<'row>) -> Result<Self> {
        if serial_type == 0 {
            Ok(Null)
        } else {
            Err(source.type_mismatch(ValueKind::Null, serial_type))
        }
    }
}

impl<'row> DecodeField<'row> for i64 {
    fn decode(serial_type: u64, source: FieldSource<'row>) -> Result<Self> {
        match serial_type {
            1..=6 => {
                let bytes = source.bytes();
                let mut pos = 0usize;
                let len = bytes.len();
                read_signed_be_at(bytes, &mut pos, len)
            }
            8 => Ok(0),
            9 => Ok(1),
            _ => Err(source.type_mismatch(ValueKind::Integer, serial_type)),
        }
    }
}

impl<'row> DecodeField<'row> for f64 {
    fn decode(serial_type: u64, source: FieldSource<'row>) -> Result<Self> {
        if serial_type != 7 {
            return Err(source.type_mismatch(ValueKind::Real, serial_type));
        }
        let bytes = source.bytes();
        let mut pos = 0usize;
        let raw = read_u64_be_at(bytes, &mut pos)?;
        Ok(f64::from_bits(raw))
    }
}

impl<'row> DecodeField<'row> for &'row str {
    fn decode(serial_type: u64, source: FieldSource<'row>) -> Result<Self> {
        if serial_type < 13 || serial_type & 1 == 0 {
            return Err(source.type_mismatch(ValueKind::Text, serial_type));
        }
        let bytes = source.bytes();
        str::from_utf8(bytes).map_err(|err| source.invalid_utf8(err))
    }
}

impl<'row> DecodeField<'row> for &'row [u8] {
    fn decode(serial_type: u64, source: FieldSource<'row>) -> Result<Self> {
        if serial_type < 12 || serial_type & 1 == 1 {
            return Err(source.type_mismatch(ValueKind::Blob, serial_type));
        }
        Ok(source.bytes())
    }
}

impl<'row> DecodeField<'row> for String {
    fn decode(serial_type: u64, source: FieldSource<'row>) -> Result<Self> {
        if serial_type < 13 || serial_type & 1 == 0 {
            return Err(source.type_mismatch(ValueKind::Text, serial_type));
        }
        let bytes = source.bytes();
        let text = str::from_utf8(bytes).map_err(|err| source.invalid_utf8(err))?;
        Ok(text.to_owned())
    }
}

impl<'row> DecodeField<'row> for Vec<u8> {
    fn decode(serial_type: u64, source: FieldSource<'row>) -> Result<Self> {
        if serial_type < 12 || serial_type & 1 == 1 {
            return Err(source.type_mismatch(ValueKind::Blob, serial_type));
        }
        Ok(source.bytes().to_vec())
    }
}

impl<'row, T> DecodeField<'row> for Option<T>
where
    T: DecodeField<'row>,
{
    fn decode(serial_type: u64, source: FieldSource<'row>) -> Result<Self> {
        if serial_type == 0 { Ok(None) } else { T::decode(serial_type, source).map(Some) }
    }
}

/// Scan a table and decode rows into a typed schema.
pub fn scan_table_typed<D, F>(pager: &Pager, page_id: PageId, f: F) -> Result<()>
where
    D: DecodeRecord,
    F: for<'row> FnMut(i64, D::Row<'row>) -> Result<()>,
{
    scan_table_typed_with_options(pager, page_id, TypedScanOptions::default(), f)
}

/// Scan a table with typed decoding using the inline-only row view path.
///
/// This avoids payload dispatch and rowid varint parsing but does not support
/// overflow payloads.
pub fn scan_table_typed_inline<D, F>(pager: &Pager, page_id: PageId, f: F) -> Result<()>
where
    D: DecodeRecord,
    F: for<'row> FnMut(i64, D::Row<'row>) -> Result<()>,
{
    scan_table_typed_inline_with_options(pager, page_id, TypedScanOptions::default(), f)
}

/// Scan a table and decode rows into a typed schema with custom options.
pub fn scan_table_typed_with_options<D, F>(
    pager: &Pager,
    page_id: PageId,
    options: TypedScanOptions,
    mut f: F,
) -> Result<()>
where
    D: DecodeRecord,
    F: for<'row> FnMut(i64, D::Row<'row>) -> Result<()>,
{
    let mut scratch = RecordScratch::default();
    let mut stack = Vec::with_capacity(64);
    stack.push(page_id);
    scan_table_payloads_with_stack(pager, &mut stack, &mut |rowid, page_id, payload| {
        let mut decoder = match RecordDecoder::new_with_scratch(
            payload,
            mem::take(&mut scratch),
            options,
            D::COLS,
        ) {
            Ok(decoder) => decoder,
            Err(err) => return Err(Error::RowDecode { rowid, page_id, source: Box::new(err) }),
        };

        if let Err(err) = decoder.ensure_column_count(D::COLS, options.column_mode) {
            return Err(Error::RowDecode { rowid, page_id, source: Box::new(err) });
        }

        let row = match D::decode(&mut decoder) {
            Ok(row) => row,
            Err(err) => return Err(Error::RowDecode { rowid, page_id, source: Box::new(err) }),
        };

        if let Err(err) = decoder.finish(D::COLS, options.column_mode) {
            return Err(Error::RowDecode { rowid, page_id, source: Box::new(err) });
        }

        scratch = decoder.into_scratch();
        f(rowid, row)
    })
}

/// Scan a table with typed decoding and custom options using the inline-only
/// row view path.
///
/// This avoids payload dispatch and rowid varint parsing but does not support
/// overflow payloads.
pub fn scan_table_typed_inline_with_options<D, F>(
    pager: &Pager,
    page_id: PageId,
    options: TypedScanOptions,
    mut f: F,
) -> Result<()>
where
    D: DecodeRecord,
    F: for<'row> FnMut(i64, D::Row<'row>) -> Result<()>,
{
    let mut scratch = RecordScratch::default();
    scan_table(pager, page_id, |rowid, row| {
        let payload = PayloadRef::Inline(row.payload);
        let mut decoder = match RecordDecoder::new_with_scratch(
            payload,
            mem::take(&mut scratch),
            options,
            D::COLS,
        ) {
            Ok(decoder) => decoder,
            Err(err) => return Err(Error::RowDecode { rowid, page_id, source: Box::new(err) }),
        };

        if let Err(err) = decoder.ensure_column_count(D::COLS, options.column_mode) {
            return Err(Error::RowDecode { rowid, page_id, source: Box::new(err) });
        }

        let row = match D::decode(&mut decoder) {
            Ok(row) => row,
            Err(err) => return Err(Error::RowDecode { rowid, page_id, source: Box::new(err) }),
        };

        if let Err(err) = decoder.finish(D::COLS, options.column_mode) {
            return Err(Error::RowDecode { rowid, page_id, source: Box::new(err) });
        }

        scratch = decoder.into_scratch();
        f(rowid, row)
    })
}

#[inline]
fn serial_type_len_checked(serial: u64, col: usize) -> Result<usize> {
    match serial {
        10 | 11 => Err(Error::UnsupportedSerialTypeAt { col, serial }),
        _ => Ok(serial_type_len_fast(serial)),
    }
}

fn parse_record_header_inline(payload: &[u8]) -> Result<(usize, &[u8])> {
    let mut header_pos = 0usize;
    let first = *payload.first().ok_or(Error::Corrupted(Corruption::RecordHeaderTruncated))?;
    let header_len = if first < 0x80 {
        header_pos = 1;
        first as usize
    } else {
        read_varint_at(payload, &mut header_pos, Corruption::RecordHeaderTruncated)? as usize
    };
    if header_len < header_pos || header_len > payload.len() {
        return Err(Error::Corrupted(Corruption::InvalidRecordHeaderLength));
    }
    Ok((header_len, &payload[header_pos..header_len]))
}

fn count_serials(serial_bytes: &[u8]) -> Result<usize> {
    let mut pos = 0usize;
    let mut count = 0usize;
    while pos < serial_bytes.len() {
        let b = unsafe { *serial_bytes.get_unchecked(pos) };
        if b < 0x80 {
            pos += 1;
        } else {
            read_varint_at(serial_bytes, &mut pos, Corruption::RecordHeaderTruncated)?;
        }
        count += 1;
    }
    Ok(count)
}

fn parse_record_header_bytes(
    payload: &[u8],
    serials: &mut Vec<u64>,
) -> Result<(usize, usize, usize)> {
    let mut header_pos = 0usize;
    let first = *payload.first().ok_or(Error::Corrupted(Corruption::RecordHeaderTruncated))?;
    let header_len = if first < 0x80 {
        header_pos = 1;
        first as usize
    } else {
        read_varint_at(payload, &mut header_pos, Corruption::RecordHeaderTruncated)? as usize
    };
    if header_len < header_pos || header_len > payload.len() {
        return Err(Error::Corrupted(Corruption::InvalidRecordHeaderLength));
    }

    serials.clear();
    let serial_bytes = &payload[header_pos..header_len];
    let mut serial_pos = 0usize;
    let mut total_value_len = 0usize;
    let mut text_blob_len = 0usize;
    while serial_pos < serial_bytes.len() {
        let b = unsafe { *serial_bytes.get_unchecked(serial_pos) };
        let serial = if b < 0x80 {
            serial_pos += 1;
            b as u64
        } else {
            read_varint_at(serial_bytes, &mut serial_pos, Corruption::RecordHeaderTruncated)?
        };
        if serial_pos > serial_bytes.len() {
            return Err(Error::Corrupted(Corruption::RecordHeaderTruncated));
        }
        let col = serials.len();
        if serial == 10 || serial == 11 {
            return Err(Error::UnsupportedSerialTypeAt { col, serial });
        }
        serials.push(serial);
        let len = serial_type_len_fast(serial);
        total_value_len = total_value_len.saturating_add(len);
        if serial >= 12 {
            text_blob_len = text_blob_len.saturating_add(len);
        }
    }

    Ok((header_len, total_value_len, text_blob_len))
}

fn parse_record_header_overflow<'row>(
    payload: OverflowPayload<'row>,
    serials: &mut Vec<u64>,
) -> Result<(OverflowCursor<'row>, usize, usize)> {
    let mut cursor = OverflowCursor::new(payload, 0)?;
    let header_len = cursor.read_varint(Corruption::RecordHeaderTruncated)? as usize;
    let header_pos = cursor.position();
    if header_len < header_pos || header_len > payload.total_len {
        return Err(Error::Corrupted(Corruption::InvalidRecordHeaderLength));
    }

    serials.clear();
    let mut total_value_len = 0usize;
    let mut text_blob_len = 0usize;
    while cursor.position() < header_len {
        let serial = cursor.read_varint(Corruption::RecordHeaderTruncated)?;
        if cursor.position() > header_len {
            return Err(Error::Corrupted(Corruption::RecordHeaderTruncated));
        }
        let col = serials.len();
        if serial == 10 || serial == 11 {
            return Err(Error::UnsupportedSerialTypeAt { col, serial });
        }
        serials.push(serial);
        let len = serial_type_len_fast(serial);
        total_value_len = total_value_len.saturating_add(len);
        if serial >= 12 {
            text_blob_len = text_blob_len.saturating_add(len);
        }
    }

    Ok((cursor, total_value_len, text_blob_len))
}

/// Scan a table with on-demand row decoding.
///
/// This is more efficient than fully decoding rows when you only need to count
/// rows or access a small subset of columns, as it defers decoding until values
/// are actually accessed.
///
/// Note: Only works with inline payloads (non-overflow rows). For tables with
/// large TEXT/BLOB columns that overflow, use `scan_table_cells_with_scratch`
/// and decode columns from the payload.
pub fn scan_table<F>(pager: &Pager, page_id: PageId, mut f: F) -> Result<()>
where
    F: for<'row> FnMut(i64, RowView<'row>) -> Result<()>,
{
    let mut stack = Vec::with_capacity(64);
    stack.push(page_id);
    scan_table_with_stack(pager, &mut stack, &mut f)
}

pub(crate) fn scan_table_with_stack<F>(
    pager: &Pager,
    stack: &mut Vec<PageId>,
    f: &mut F,
) -> Result<()>
where
    F: for<'row> FnMut(i64, RowView<'row>) -> Result<()>,
{
    let max_pages = pager.page_count().max(1);
    let mut seen_pages = 0u32;

    while let Some(page_id) = stack.pop() {
        seen_pages += 1;
        if seen_pages > max_pages {
            return Err(Error::Corrupted(Corruption::BtreePageCycleDetected));
        }

        let page = pager.page(page_id)?;
        let header = parse_header(&page)?;
        let cell_ptrs = cell_ptrs(&page, &header)?;

        match header.kind {
            BTreeKind::TableLeaf => {
                for idx in 0..header.cell_count as usize {
                    let offset = cell_ptr_at(cell_ptrs, idx)?;
                    let (rowid, payload) = read_scan_cell(pager, &page, offset)?;
                    match payload {
                        PayloadRef::Inline(bytes) => {
                            let row = RowView::from_inline(bytes)?;
                            f(rowid, row)?;
                        }
                        PayloadRef::Overflow(_) => {
                            return Err(Error::Corrupted(
                                Corruption::ScanOverflowPayloadUnsupported,
                            ));
                        }
                    }
                }
            }
            BTreeKind::TableInterior => {
                if let Some(right_most) = header.right_most_child {
                    let right_most = PageId::try_new(right_most)
                        .ok_or(Error::Corrupted(Corruption::ChildPageIdZero))?;
                    stack.push(right_most);
                }

                let page_len = page.usable_bytes().len();
                for idx in (0..header.cell_count as usize).rev() {
                    let offset = cell_ptr_at(cell_ptrs, idx)?;
                    if offset as usize >= page_len {
                        return Err(Error::Corrupted(Corruption::CellOffsetOutOfBounds));
                    }

                    let mut decoder = Decoder::new(page.usable_bytes()).split_at(offset as usize);
                    let child =
                        read_u32_checked(&mut decoder, Corruption::CellChildPointerTruncated)?;
                    let child = PageId::try_new(child)
                        .ok_or(Error::Corrupted(Corruption::ChildPageIdZero))?;

                    let _ = read_varint_checked(&mut decoder, Corruption::CellKeyTruncated)?;
                    stack.push(child);
                }
            }
        }
    }

    Ok(())
}

/// Read a cell's rowid and payload without full parsing.
#[inline]
fn read_scan_cell<'row>(
    pager: &'row Pager,
    page: &'row PageRef<'_>,
    offset: u16,
) -> Result<(i64, PayloadRef<'row>)> {
    let usable = page.usable_bytes();
    if offset as usize >= usable.len() {
        return Err(Error::Corrupted(Corruption::CellOffsetOutOfBounds));
    }

    let mut pos = offset as usize;
    let remaining = usable.len().saturating_sub(pos);
    let (payload_length, rowid) = if remaining >= 18 {
        let payload_length = unsafe { read_varint_unchecked_at(usable, &mut pos) };
        let rowid = unsafe { read_varint_unchecked_at(usable, &mut pos) } as i64;
        (payload_length, rowid)
    } else {
        let payload_length =
            read_varint_at(usable, &mut pos, Corruption::CellPayloadLengthTruncated)?;
        let rowid = read_varint_at(usable, &mut pos, Corruption::CellRowIdTruncated)? as i64;
        (payload_length, rowid)
    };
    let payload_length = usize::try_from(payload_length)
        .map_err(|_| Error::Corrupted(Corruption::PayloadIsTooLarge))?;
    if payload_length > MAX_PAYLOAD_BYTES {
        return Err(Error::PayloadTooLarge(payload_length));
    }

    let start = pos;
    let usable_size = page.usable_size();
    let x = usable_size.checked_sub(35).ok_or(Error::Corrupted(Corruption::UsableSizeUnderflow))?;
    if payload_length <= x {
        let end = start + payload_length;
        if end > usable.len() {
            return Err(Error::Corrupted(Corruption::PayloadExtendsPastPageBoundary));
        }
        return Ok((rowid, PayloadRef::Inline(&usable[start..end])));
    }

    // Overflow payload
    let local_len = local_payload_len(usable_size, payload_length)?;
    let end_local = start + local_len;
    if end_local > usable.len() {
        return Err(Error::Corrupted(Corruption::PayloadExtendsPastPageBoundary));
    }

    let overflow_end = end_local + 4;
    if overflow_end > usable.len() {
        return Err(Error::Corrupted(Corruption::OverflowPointerOutOfBounds));
    }
    let overflow_page = u32::from_be_bytes(usable[end_local..overflow_end].try_into().unwrap());
    if overflow_page == 0 {
        return Err(Error::OverflowChainTruncated);
    }
    let payload =
        OverflowPayload::new(pager, payload_length, &usable[start..end_local], overflow_page);
    Ok((rowid, PayloadRef::Overflow(payload)))
}

/// Scan raw table cells with a caller-provided scratch buffer.
pub fn scan_table_cells_with_scratch<F>(pager: &Pager, page_id: PageId, mut f: F) -> Result<()>
where
    F: for<'row> FnMut(CellRef<'row>) -> Result<()>,
{
    scan_table_page_cells(pager, page_id, &mut f)
}

/// Lookup a cell by rowid within a table b-tree.
pub fn lookup_rowid_cell<'row>(
    pager: &'row Pager,
    page_id: PageId,
    target_rowid: i64,
) -> Result<Option<CellRef<'row>>> {
    let mut page_id = page_id;
    let max_pages = pager.page_count().max(1);
    let mut seen_pages = 0u32;

    loop {
        seen_pages += 1;
        if seen_pages > max_pages {
            return Err(Error::Corrupted(Corruption::BtreePageCycleDetected));
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
                    let offset = cell_ptr_at(cell_ptrs, mid)?;
                    let rowid = read_table_leaf_rowid(&page, offset)?;
                    if rowid < target_rowid {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                if lo < cell_count {
                    let offset = cell_ptr_at(cell_ptrs, lo)?;
                    let rowid = read_table_leaf_rowid(&page, offset)?;
                    if rowid == target_rowid {
                        let cell = read_table_cell_ref_from_bytes(pager, page_id, offset)?;
                        return Ok(Some(cell));
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
                    let offset = cell_ptr_at(cell_ptrs, mid)?;
                    let (_child, key) = read_table_interior_cell(&page, offset)?;
                    if target_rowid <= key {
                        hi = mid;
                    } else {
                        lo = mid + 1;
                    }
                }

                if lo < cell_count {
                    let offset = cell_ptr_at(cell_ptrs, lo)?;
                    let (child, _key) = read_table_interior_cell(&page, offset)?;
                    page_id = child;
                } else {
                    let right_most = header
                        .right_most_child
                        .ok_or(Error::Corrupted(Corruption::MissingRightMostChildPointer))?;
                    page_id = PageId::try_new(right_most)
                        .ok_or(Error::Corrupted(Corruption::ChildPageIdZero))?;
                }
            }
        }
    }
}

pub(crate) fn scan_table_cells_with_scratch_and_stack<F>(
    pager: &Pager,
    page_id: PageId,
    stack: &mut Vec<PageId>,
    mut f: F,
) -> Result<()>
where
    F: for<'row> FnMut(CellRef<'row>) -> Result<()>,
{
    stack.clear();
    stack.push(page_id);
    scan_table_page_cells_with_stack(pager, stack, &mut f)
}

pub(crate) fn scan_table_cells_with_scratch_and_stack_until<F, T>(
    pager: &Pager,
    page_id: PageId,
    stack: &mut Vec<PageId>,
    mut f: F,
) -> Result<Option<T>>
where
    F: for<'row> FnMut(CellRef<'row>) -> Result<Option<T>>,
{
    stack.clear();
    stack.push(page_id);
    scan_table_page_cells_until_with_stack(pager, stack, &mut f)
}

fn scan_table_payloads_with_stack<'pager, F>(
    pager: &'pager Pager,
    stack: &mut Vec<PageId>,
    f: &mut F,
) -> Result<()>
where
    F: for<'row> FnMut(i64, PageId, PayloadRef<'row>) -> Result<()>,
{
    let max_pages = pager.page_count().max(1);
    let mut seen_pages = 0u32;

    while let Some(page_id) = stack.pop() {
        seen_pages += 1;
        if seen_pages > max_pages {
            return Err(Error::Corrupted(Corruption::BtreePageCycleDetected));
        }

        let page = pager.page(page_id)?;
        let header = parse_header(&page)?;
        let cell_ptrs = cell_ptrs(&page, &header)?;

        match header.kind {
            BTreeKind::TableLeaf => {
                for idx in 0..header.cell_count as usize {
                    let offset = cell_ptr_at(cell_ptrs, idx)?;
                    let (rowid, payload) = read_scan_cell(pager, &page, offset)?;
                    f(rowid, page_id, payload)?;
                }
            }
            BTreeKind::TableInterior => {
                if let Some(right_most) = header.right_most_child {
                    let right_most = PageId::try_new(right_most)
                        .ok_or(Error::Corrupted(Corruption::ChildPageIdZero))?;
                    stack.push(right_most);
                }

                let page_len = page.usable_bytes().len();
                for idx in (0..header.cell_count as usize).rev() {
                    let offset = cell_ptr_at(cell_ptrs, idx)?;
                    if offset as usize >= page_len {
                        return Err(Error::Corrupted(Corruption::CellOffsetOutOfBounds));
                    }

                    let mut decoder = Decoder::new(page.usable_bytes()).split_at(offset as usize);
                    let child =
                        read_u32_checked(&mut decoder, Corruption::CellChildPointerTruncated)?;
                    let child = PageId::try_new(child)
                        .ok_or(Error::Corrupted(Corruption::ChildPageIdZero))?;

                    let _ = read_varint_checked(&mut decoder, Corruption::CellKeyTruncated)?;
                    stack.push(child);
                }
            }
        }
    }

    Ok(())
}

fn scan_table_page_cells<'pager, F>(pager: &'pager Pager, page_id: PageId, f: &mut F) -> Result<()>
where
    F: for<'row> FnMut(CellRef<'row>) -> Result<()>,
{
    let mut stack = Vec::with_capacity(64);
    stack.push(page_id);
    scan_table_page_cells_with_stack(pager, &mut stack, f)
}

fn scan_table_page_cells_with_stack<'pager, F>(
    pager: &'pager Pager,
    stack: &mut Vec<PageId>,
    f: &mut F,
) -> Result<()>
where
    F: for<'row> FnMut(CellRef<'row>) -> Result<()>,
{
    let max_pages = pager.page_count().max(1);
    let mut seen_pages = 0u32;

    while let Some(page_id) = stack.pop() {
        seen_pages += 1;
        if seen_pages > max_pages {
            return Err(Error::Corrupted(Corruption::BtreePageCycleDetected));
        }

        let page = pager.page(page_id)?;
        let header = parse_header(&page)?;
        let cell_ptrs = cell_ptrs(&page, &header)?;

        match header.kind {
            BTreeKind::TableLeaf => {
                for idx in 0..header.cell_count as usize {
                    let offset = cell_ptr_at(cell_ptrs, idx)?;
                    let cell = read_table_cell_ref(pager, page_id, &page, offset)?;
                    f(cell)?;
                }
            }
            BTreeKind::TableInterior => {
                if let Some(right_most) = header.right_most_child {
                    let right_most = PageId::try_new(right_most)
                        .ok_or(Error::Corrupted(Corruption::ChildPageIdZero))?;
                    stack.push(right_most);
                }

                let page_len = page.usable_bytes().len();
                for idx in (0..header.cell_count as usize).rev() {
                    let offset = cell_ptr_at(cell_ptrs, idx)?;
                    if offset as usize >= page_len {
                        return Err(Error::Corrupted(Corruption::CellOffsetOutOfBounds));
                    }

                    let mut decoder = Decoder::new(page.usable_bytes()).split_at(offset as usize);
                    let child =
                        read_u32_checked(&mut decoder, Corruption::CellChildPointerTruncated)?;
                    let child = PageId::try_new(child)
                        .ok_or(Error::Corrupted(Corruption::ChildPageIdZero))?;

                    let _ = read_varint_checked(&mut decoder, Corruption::CellKeyTruncated)?;
                    stack.push(child);
                }
            }
        }
    }

    Ok(())
}

fn scan_table_page_cells_until_with_stack<'pager, F, T>(
    pager: &'pager Pager,
    stack: &mut Vec<PageId>,
    f: &mut F,
) -> Result<Option<T>>
where
    F: for<'row> FnMut(CellRef<'row>) -> Result<Option<T>>,
{
    let max_pages = pager.page_count().max(1);
    let mut seen_pages = 0u32;

    while let Some(page_id) = stack.pop() {
        seen_pages += 1;
        if seen_pages > max_pages {
            return Err(Error::Corrupted(Corruption::BtreePageCycleDetected));
        }

        let page = pager.page(page_id)?;
        let header = parse_header(&page)?;
        let cell_ptrs = cell_ptrs(&page, &header)?;

        match header.kind {
            BTreeKind::TableLeaf => {
                for idx in 0..header.cell_count as usize {
                    let offset = cell_ptr_at(cell_ptrs, idx)?;
                    let cell = read_table_cell_ref(pager, page_id, &page, offset)?;
                    if let Some(value) = f(cell)? {
                        return Ok(Some(value));
                    }
                }
            }
            BTreeKind::TableInterior => {
                if let Some(right_most) = header.right_most_child {
                    let right_most = PageId::try_new(right_most)
                        .ok_or(Error::Corrupted(Corruption::ChildPageIdZero))?;
                    stack.push(right_most);
                }

                let page_len = page.usable_bytes().len();
                for idx in (0..header.cell_count as usize).rev() {
                    let offset = cell_ptr_at(cell_ptrs, idx)?;
                    if offset as usize >= page_len {
                        return Err(Error::Corrupted(Corruption::CellOffsetOutOfBounds));
                    }

                    let mut decoder = Decoder::new(page.usable_bytes()).split_at(offset as usize);
                    let child =
                        read_u32_checked(&mut decoder, Corruption::CellChildPointerTruncated)?;
                    let child = PageId::try_new(child)
                        .ok_or(Error::Corrupted(Corruption::ChildPageIdZero))?;

                    let _ = read_varint_checked(&mut decoder, Corruption::CellKeyTruncated)?;
                    stack.push(child);
                }
            }
        }
    }

    Ok(None)
}

#[inline]
fn read_table_cell_ref<'row>(
    pager: &'row Pager,
    page_id: PageId,
    page: &'row PageRef<'_>,
    offset: u16,
) -> Result<CellRef<'row>> {
    let usable = page.usable_bytes();
    if offset as usize >= usable.len() {
        return Err(Error::Corrupted(Corruption::CellOffsetOutOfBounds));
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
        let payload_length =
            read_varint_at(usable, &mut pos, Corruption::CellPayloadLengthTruncated)?;
        let rowid = read_varint_at(usable, &mut pos, Corruption::CellRowIdTruncated)? as i64;
        (payload_length, rowid)
    };
    let payload_length = usize::try_from(payload_length)
        .map_err(|_| Error::Corrupted(Corruption::PayloadIsTooLarge))?;
    if payload_length > MAX_PAYLOAD_BYTES {
        return Err(Error::PayloadTooLarge(payload_length));
    }

    let start = pos;
    let usable_size = page.usable_size();
    let x = usable_size.checked_sub(35).ok_or(Error::Corrupted(Corruption::UsableSizeUnderflow))?;
    if payload_length <= x {
        let end = start + payload_length;
        if end > usable.len() {
            return Err(Error::Corrupted(Corruption::PayloadExtendsPastPageBoundary));
        }
        return Ok(CellRef {
            rowid,
            payload: PayloadRef::Inline(&usable[start..end]),
            page_id,
            cell_offset: offset,
        });
    }

    let local_len = local_payload_len(usable_size, payload_length)?;
    let end_local = start + local_len;
    if end_local > usable.len() {
        return Err(Error::Corrupted(Corruption::PayloadExtendsPastPageBoundary));
    }

    let overflow_end = end_local + 4;
    if overflow_end > usable.len() {
        return Err(Error::Corrupted(Corruption::OverflowPointerOutOfBounds));
    }
    let overflow_page = u32::from_be_bytes(usable[end_local..overflow_end].try_into().unwrap());
    if overflow_page == 0 {
        return Err(Error::OverflowChainTruncated);
    }
    let payload =
        OverflowPayload::new(pager, payload_length, &usable[start..end_local], overflow_page);
    Ok(CellRef { rowid, payload: PayloadRef::Overflow(payload), page_id, cell_offset: offset })
}

pub(crate) fn read_table_cell_ref_from_bytes<'row>(
    pager: &'row Pager,
    page_id: PageId,
    offset: u16,
) -> Result<CellRef<'row>> {
    let page_bytes = pager.page_bytes(page_id)?;
    let usable_end = pager.header().usable_size.min(page_bytes.len());
    let usable = &page_bytes[..usable_end];

    if offset as usize >= usable.len() {
        return Err(Error::Corrupted(Corruption::CellOffsetOutOfBounds));
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
        let payload_length =
            read_varint_at(usable, &mut pos, Corruption::CellPayloadLengthTruncated)?;
        let rowid = read_varint_at(usable, &mut pos, Corruption::CellRowIdTruncated)? as i64;
        (payload_length, rowid)
    };
    let payload_length = usize::try_from(payload_length)
        .map_err(|_| Error::Corrupted(Corruption::PayloadIsTooLarge))?;
    if payload_length > MAX_PAYLOAD_BYTES {
        return Err(Error::PayloadTooLarge(payload_length));
    }

    let start = pos;
    let usable_size = pager.header().usable_size;
    let x = usable_size.checked_sub(35).ok_or(Error::Corrupted(Corruption::UsableSizeUnderflow))?;
    if payload_length <= x {
        let end = start + payload_length;
        if end > usable.len() {
            return Err(Error::Corrupted(Corruption::PayloadExtendsPastPageBoundary));
        }
        return Ok(CellRef {
            rowid,
            payload: PayloadRef::Inline(&usable[start..end]),
            page_id,
            cell_offset: offset,
        });
    }

    let local_len = local_payload_len(usable_size, payload_length)?;
    let end_local = start + local_len;
    if end_local > usable.len() {
        return Err(Error::Corrupted(Corruption::PayloadExtendsPastPageBoundary));
    }

    let overflow_end = end_local + 4;
    if overflow_end > usable.len() {
        return Err(Error::Corrupted(Corruption::OverflowPointerOutOfBounds));
    }
    let overflow_page = u32::from_be_bytes(usable[end_local..overflow_end].try_into().unwrap());
    if overflow_page == 0 {
        return Err(Error::OverflowChainTruncated);
    }
    let payload =
        OverflowPayload::new(pager, payload_length, &usable[start..end_local], overflow_page);
    Ok(CellRef { rowid, payload: PayloadRef::Overflow(payload), page_id, cell_offset: offset })
}

fn read_table_leaf_rowid(page: &PageRef<'_>, offset: u16) -> Result<i64> {
    let usable = page.usable_bytes();
    if offset as usize >= usable.len() {
        return Err(Error::Corrupted(Corruption::CellOffsetOutOfBounds));
    }

    let mut pos = offset as usize;
    let remaining = usable.len().saturating_sub(pos);
    let payload_length = if remaining >= 18 {
        // SAFETY: remaining >= 18 guarantees two varints (max 9 bytes each) are
        // in-bounds.
        unsafe { read_varint_unchecked_at(usable, &mut pos) }
    } else {
        read_varint_at(usable, &mut pos, Corruption::CellPayloadLengthTruncated)?
    };
    let payload_length = usize::try_from(payload_length)
        .map_err(|_| Error::Corrupted(Corruption::PayloadIsTooLarge))?;
    if payload_length > MAX_PAYLOAD_BYTES {
        return Err(Error::PayloadTooLarge(payload_length));
    }
    let rowid = if remaining >= 18 {
        // SAFETY: remaining >= 18 guarantees two varints (max 9 bytes each) are
        // in-bounds.
        (unsafe { read_varint_unchecked_at(usable, &mut pos) }) as i64
    } else {
        read_varint_at(usable, &mut pos, Corruption::CellRowIdTruncated)? as i64
    };
    Ok(rowid)
}

fn read_table_interior_cell(page: &PageRef<'_>, offset: u16) -> Result<(PageId, i64)> {
    let usable = page.usable_bytes();
    let start = offset as usize;
    if start + 4 > usable.len() {
        return Err(Error::Corrupted(Corruption::CellOffsetOutOfBounds));
    }

    let child = u32::from_be_bytes([
        usable[start],
        usable[start + 1],
        usable[start + 2],
        usable[start + 3],
    ]);
    let child = PageId::try_new(child).ok_or(Error::Corrupted(Corruption::ChildPageIdZero))?;
    let mut pos = start + 4;
    let key = read_varint_at(usable, &mut pos, Corruption::CellKeyTruncated)? as i64;
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

    let mut remaining = payload_len
        .checked_sub(local_len)
        .ok_or(Error::Corrupted(Corruption::PayloadLengthUnderflow))?;

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
            return Err(Error::Corrupted(Corruption::OverflowPageTooSmall));
        }

        let next_page = u32::from_be_bytes(page_bytes[0..4].try_into().unwrap());
        let usable = pager.header().usable_size.min(page_bytes.len());
        if usable < 4 {
            return Err(Error::Corrupted(Corruption::OverflowPageUsableSizeTooSmall));
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

fn push_checked<T>(out: &mut Vec<T>, value: T, kind: ScratchKind) -> Result<()> {
    if out.len() == out.capacity() {
        let needed = out.len().saturating_add(1);
        return Err(Error::ScratchTooSmall { kind, needed, capacity: out.capacity() });
    }
    out.push(value);
    Ok(())
}

fn ensure_bytes_capacity(bytes: &Vec<u8>, additional: usize) -> Result<()> {
    let needed = bytes.len().saturating_add(additional);
    if needed > bytes.capacity() {
        return Err(Error::ScratchTooSmall {
            kind: ScratchKind::Bytes,
            needed,
            capacity: bytes.capacity(),
        });
    }
    Ok(())
}

struct OverflowCursor<'row> {
    payload: OverflowPayload<'row>,
    pos: usize,
    segment_start: usize,
    segment: &'row [u8],
    next_overflow: u32,
    visited: u32,
    max_pages: u32,
}

impl<'row> OverflowCursor<'row> {
    fn new(payload: OverflowPayload<'row>, offset: usize) -> Result<Self> {
        if offset > payload.total_len {
            return Err(Error::Corrupted(Corruption::RecordPayloadShorterThanDeclared));
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

    fn ensure_segment(&mut self, msg: Corruption) -> Result<()> {
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
            return Err(Error::Corrupted(Corruption::OverflowPageTooSmall));
        }

        let next_page = u32::from_be_bytes(page_bytes[0..4].try_into().unwrap());
        let usable = self.payload.pager.header().usable_size.min(page_bytes.len());
        if usable < 4 {
            return Err(Error::Corrupted(Corruption::OverflowPageUsableSizeTooSmall));
        }
        let content = &page_bytes[4..usable];

        self.segment_start = self.pos;
        self.segment = content;
        self.next_overflow = next_page;
        Ok(())
    }

    fn read_byte(&mut self, msg: Corruption) -> Result<u8> {
        self.ensure_segment(msg)?;
        if self.pos >= self.segment_end() {
            return Err(Error::Corrupted(msg));
        }
        let offset = self.pos - self.segment_start;
        let byte = unsafe { *self.segment.get_unchecked(offset) };
        self.pos += 1;
        Ok(byte)
    }

    fn read_varint(&mut self, msg: Corruption) -> Result<u64> {
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

    fn read_bytes_into(&mut self, out: &mut [u8], msg: Corruption) -> Result<()> {
        let mut written = 0usize;
        while written < out.len() {
            self.ensure_segment(msg)?;
            let available = self.segment_end() - self.pos;
            if available == 0 {
                continue;
            }
            let take = (out.len() - written).min(available);
            let offset = self.pos - self.segment_start;
            out[written..written + take].copy_from_slice(&self.segment[offset..offset + take]);
            self.pos += take;
            written += take;
        }
        Ok(())
    }

    fn read_signed_be(&mut self, len: usize) -> Result<i64> {
        let mut buf = [0u8; 8];
        let start = 8 - len;
        self.read_bytes_into(&mut buf[start..], Corruption::RecordPayloadShorterThanDeclared)?;
        let value = u64::from_be_bytes(buf);
        let shift = (8 - len) * 8;
        Ok(((value << shift) as i64) >> shift)
    }

    fn read_u64_be(&mut self) -> Result<u64> {
        let mut buf = [0u8; 8];
        self.read_bytes_into(&mut buf, Corruption::RecordPayloadShorterThanDeclared)?;
        Ok(u64::from_be_bytes(buf))
    }

    fn take_span(&mut self, len: usize, bytes: &mut Vec<u8>) -> Result<BytesSpan> {
        if len == 0 {
            return Ok(BytesSpan::mmap(&[]));
        }

        self.ensure_segment(Corruption::RecordPayloadShorterThanDeclared)?;
        let available = self.segment_end() - self.pos;
        if len <= available {
            let offset = self.pos - self.segment_start;
            let slice = &self.segment[offset..offset + len];
            self.pos += len;
            return Ok(BytesSpan::mmap(slice));
        }

        ensure_bytes_capacity(bytes, len)?;
        let start = bytes.len();
        let mut remaining = len;
        while remaining > 0 {
            self.ensure_segment(Corruption::RecordPayloadShorterThanDeclared)?;
            let available = self.segment_end() - self.pos;
            if available == 0 {
                continue;
            }
            let take = remaining.min(available);
            let offset = self.pos - self.segment_start;
            bytes.extend_from_slice(&self.segment[offset..offset + take]);
            self.pos += take;
            remaining -= take;
        }

        let slice = &bytes[start..start + len];
        Ok(BytesSpan::scratch(slice))
    }

    fn skip(&mut self, mut len: usize) -> Result<()> {
        while len > 0 {
            self.ensure_segment(Corruption::RecordPayloadShorterThanDeclared)?;
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
    out: &mut Vec<ValueSlot>,
    bytes: &mut Vec<u8>,
    serials: &mut Vec<u64>,
    offsets: &mut Vec<u32>,
) -> Result<usize> {
    out.clear();
    bytes.clear();
    serials.clear();
    offsets.clear();

    let mut serial_cursor = OverflowCursor::new(*payload, 0)?;
    let header_len = serial_cursor.read_varint(Corruption::RecordHeaderTruncated)? as usize;
    let header_pos = serial_cursor.position();
    if header_len < header_pos || header_len > payload.total_len {
        return Err(Error::Corrupted(Corruption::InvalidRecordHeaderLength));
    }

    let mut value_pos = header_len;
    while serial_cursor.position() < header_len {
        let serial = serial_cursor.read_varint(Corruption::RecordHeaderTruncated)?;
        if serial_cursor.position() > header_len {
            return Err(Error::Corrupted(Corruption::RecordHeaderTruncated));
        }
        push_checked(serials, serial, ScratchKind::Serials)?;
        push_checked(offsets, value_pos as u32, ScratchKind::Offsets)?;
        value_pos += serial_type_len_fast(serial);
    }

    let mut value_cursor = OverflowCursor::new(*payload, header_len)?;
    if let Some(needed_cols) = needed_cols {
        let mut needed_iter = needed_cols.iter().copied();
        let mut next_needed = needed_iter.next();
        for (idx, serial) in serials.iter().copied().enumerate() {
            let col_idx = idx as u16;
            if Some(col_idx) == next_needed {
                let raw = decode_value_ref_at_cursor(serial, &mut value_cursor, bytes)?;
                push_checked(out, raw, ScratchKind::Values)?;
                next_needed = needed_iter.next();
            } else {
                skip_value_at_cursor(serial, &mut value_cursor)?;
            }
        }
    } else {
        for serial in serials.iter().copied() {
            let raw = decode_value_ref_at_cursor(serial, &mut value_cursor, bytes)?;
            push_checked(out, raw, ScratchKind::Values)?;
        }
    }

    Ok(serials.len())
}

pub(crate) fn decode_record_project_into(
    payload: PayloadRef<'_>,
    needed_cols: Option<&[u16]>,
    out: &mut Vec<ValueSlot>,
    bytes: &mut Vec<u8>,
    serials: &mut Vec<u64>,
    offsets: &mut Vec<u32>,
) -> Result<usize> {
    match payload {
        PayloadRef::Inline(payload_bytes) => {
            decode_record_project_into_bytes(payload_bytes, needed_cols, out, serials, offsets)
        }
        PayloadRef::Overflow(payload) => {
            decode_record_project_into_overflow(&payload, needed_cols, out, bytes, serials, offsets)
        }
    }
}

pub(crate) fn decode_record_project_into_mapped(
    payload: PayloadRef<'_>,
    needed_map: &[(u16, usize)],
    out: &mut Vec<ValueSlot>,
    bytes: &mut Vec<u8>,
    serials: &mut Vec<u64>,
    offsets: &mut Vec<u32>,
) -> Result<usize> {
    match payload {
        PayloadRef::Inline(bytes) => {
            decode_record_project_into_bytes_mapped(bytes, needed_map, out)
        }
        PayloadRef::Overflow(payload) => decode_record_project_into_overflow_mapped(
            &payload, needed_map, out, bytes, serials, offsets,
        ),
    }
}

pub(crate) fn record_column_count(payload: PayloadRef<'_>) -> Result<usize> {
    match payload {
        PayloadRef::Inline(bytes) => record_column_count_bytes(bytes),
        PayloadRef::Overflow(payload) => record_column_count_overflow(&payload),
    }
}

fn record_column_count_bytes(payload: &[u8]) -> Result<usize> {
    let mut header_pos = 0usize;
    let first = *payload.first().ok_or(Error::Corrupted(Corruption::RecordHeaderTruncated))?;
    let header_len = if first < 0x80 {
        header_pos = 1;
        first as usize
    } else {
        read_varint_at(payload, &mut header_pos, Corruption::RecordHeaderTruncated)? as usize
    };
    if header_len < header_pos || header_len > payload.len() {
        return Err(Error::Corrupted(Corruption::InvalidRecordHeaderLength));
    }

    let serial_bytes = &payload[header_pos..header_len];
    let mut serial_pos = 0usize;
    let mut count = 0usize;
    while serial_pos < serial_bytes.len() {
        let b = unsafe { *serial_bytes.get_unchecked(serial_pos) };
        if b < 0x80 {
            serial_pos += 1;
        } else {
            read_varint_at(serial_bytes, &mut serial_pos, Corruption::RecordHeaderTruncated)?;
        }
        count += 1;
    }

    Ok(count)
}

fn record_column_count_overflow(payload: &OverflowPayload<'_>) -> Result<usize> {
    let mut serial_cursor = OverflowCursor::new(*payload, 0)?;
    let header_len = serial_cursor.read_varint(Corruption::RecordHeaderTruncated)? as usize;
    let header_pos = serial_cursor.position();
    if header_len < header_pos || header_len > payload.total_len {
        return Err(Error::Corrupted(Corruption::InvalidRecordHeaderLength));
    }

    let mut count = 0usize;
    while serial_cursor.position() < header_len {
        let _serial = serial_cursor.read_varint(Corruption::RecordHeaderTruncated)?;
        if serial_cursor.position() > header_len {
            return Err(Error::Corrupted(Corruption::RecordHeaderTruncated));
        }
        count += 1;
    }

    Ok(count)
}

#[inline(always)]
fn decode_record_project_into_bytes(
    payload: &[u8],
    needed_cols: Option<&[u16]>,
    out: &mut Vec<ValueSlot>,
    serials: &mut Vec<u64>,
    offsets: &mut Vec<u32>,
) -> Result<usize> {
    out.clear();
    serials.clear();
    offsets.clear();

    let mut header_pos = 0usize;
    let first = *payload.first().ok_or(Error::Corrupted(Corruption::RecordHeaderTruncated))?;
    let header_len = if first < 0x80 {
        header_pos = 1;
        first as usize
    } else {
        read_varint_at(payload, &mut header_pos, Corruption::RecordHeaderTruncated)? as usize
    };
    if header_len < header_pos || header_len > payload.len() {
        return Err(Error::Corrupted(Corruption::InvalidRecordHeaderLength));
    }

    let serial_bytes = &payload[header_pos..header_len];

    if let Some(needed_cols) = needed_cols {
        // PROJECTION PATH: SQLite optimization - parse header only up to max needed
        // column
        let max_needed_col = needed_cols.iter().map(|&c| c as usize).max().unwrap_or(0);
        let needed_cap = max_needed_col.saturating_add(1);
        if serials.capacity() < needed_cap {
            serials.reserve(needed_cap - serials.capacity());
        }
        if offsets.capacity() < needed_cap {
            offsets.reserve(needed_cap - offsets.capacity());
        }

        // Phase 1: Parse serial types lazily (only up to max_needed_col + 1)
        let mut serial_pos = 0usize;
        let mut value_pos = header_len;
        while serial_pos < serial_bytes.len() && serials.len() <= max_needed_col {
            let b = unsafe { *serial_bytes.get_unchecked(serial_pos) };
            let serial = if b < 0x80 {
                serial_pos += 1;
                b as u64
            } else {
                read_varint_at(serial_bytes, &mut serial_pos, Corruption::RecordHeaderTruncated)?
            };
            serials.push(serial);
            offsets.push(value_pos as u32);
            value_pos += serial_type_len_fast(serial);
        }

        // Count remaining columns without storing (needed for column_count validation)
        let mut column_count = serials.len();
        while serial_pos < serial_bytes.len() {
            let b = unsafe { *serial_bytes.get_unchecked(serial_pos) };
            if b < 0x80 {
                serial_pos += 1;
            } else {
                // Skip multi-byte varint
                while serial_pos < serial_bytes.len() {
                    let byte = unsafe { *serial_bytes.get_unchecked(serial_pos) };
                    serial_pos += 1;
                    if byte & 0x80 == 0 {
                        break;
                    }
                }
            }
            column_count += 1;
        }

        // Check capacity once before decoding
        if out.capacity() < needed_cols.len() {
            return Err(Error::ScratchTooSmall {
                kind: ScratchKind::Values,
                needed: needed_cols.len(),
                capacity: out.capacity(),
            });
        }

        // Phase 2: Decode only needed columns
        match needed_cols.len() {
            0 => {}
            1 => {
                // Fast path for single column projection
                let col_idx = needed_cols[0] as usize;
                if col_idx < column_count {
                    let serial = unsafe { *serials.get_unchecked(col_idx) };
                    let mut decode_pos = unsafe { *offsets.get_unchecked(col_idx) } as usize;
                    let value = decode_value_ref_at(serial, payload, &mut decode_pos)?;
                    out.push(value);
                }
            }
            _ => {
                // Multi-column path: use cached offsets (SQLite aOffset[] style)
                // needed_cols is already sorted by build_plan
                for &col in needed_cols {
                    let col_idx = col as usize;
                    if col_idx >= column_count {
                        continue;
                    }
                    let serial = unsafe { *serials.get_unchecked(col_idx) };
                    let mut decode_pos = unsafe { *offsets.get_unchecked(col_idx) } as usize;
                    let value = decode_value_ref_at(serial, payload, &mut decode_pos)?;
                    out.push(value);
                }
            }
        }
        Ok(column_count)
    } else {
        // NO PROJECTION PATH: single-pass parse + decode (no intermediate serials
        // storage) Estimate column count from header size (each serial type is
        // at least 1 byte)
        let estimated_cols = serial_bytes.len();
        if out.capacity() < estimated_cols {
            return Err(Error::ScratchTooSmall {
                kind: ScratchKind::Values,
                needed: estimated_cols,
                capacity: out.capacity(),
            });
        }

        let mut serial_pos = 0usize;
        let mut value_pos = header_len;

        while serial_pos < serial_bytes.len() {
            // Parse serial type inline
            let b = unsafe { *serial_bytes.get_unchecked(serial_pos) };
            let serial = if b < 0x80 {
                serial_pos += 1;
                b as u64
            } else {
                read_varint_at(serial_bytes, &mut serial_pos, Corruption::RecordHeaderTruncated)?
            };

            // Decode value immediately - no capacity check needed
            let value = decode_value_ref_at(serial, payload, &mut value_pos)?;
            out.push(value);
        }

        Ok(out.len())
    }
}

/// Lookup table for serial type lengths (serial types 0-127).
/// This is the same optimization SQLite uses in sqlite3SmallTypeSizes[].
/// For serial_type < 128, we can use a single table lookup instead of branching.
#[rustfmt::skip]
static SERIAL_TYPE_LEN: [u8; 128] = [
    //  0   1   2   3   4   5   6   7   8   9
        0,  1,  2,  3,  4,  6,  8,  8,  0,  0,  // 0-9
        0,  0,  0,  0,  1,  1,  2,  2,  3,  3,  // 10-19
        4,  4,  5,  5,  6,  6,  7,  7,  8,  8,  // 20-29
        9,  9, 10, 10, 11, 11, 12, 12, 13, 13,  // 30-39
       14, 14, 15, 15, 16, 16, 17, 17, 18, 18,  // 40-49
       19, 19, 20, 20, 21, 21, 22, 22, 23, 23,  // 50-59
       24, 24, 25, 25, 26, 26, 27, 27, 28, 28,  // 60-69
       29, 29, 30, 30, 31, 31, 32, 32, 33, 33,  // 70-79
       34, 34, 35, 35, 36, 36, 37, 37, 38, 38,  // 80-89
       39, 39, 40, 40, 41, 41, 42, 42, 43, 43,  // 90-99
       44, 44, 45, 45, 46, 46, 47, 47, 48, 48,  // 100-109
       49, 49, 50, 50, 51, 51, 52, 52, 53, 53,  // 110-119
       54, 54, 55, 55, 56, 56, 57, 57,          // 120-127
];

/// Fast inline version of serial_type_len using lookup table for small types
#[inline(always)]
fn serial_type_len_fast(serial_type: u64) -> usize {
    if serial_type < 128 {
        // Fast path: table lookup (no branches)
        SERIAL_TYPE_LEN[serial_type as usize] as usize
    } else {
        // Slow path: compute for large serial types (rare - very long strings/blobs)
        ((serial_type - 12) / 2) as usize
    }
}

fn decode_record_project_into_bytes_mapped(
    payload: &[u8],
    needed_map: &[(u16, usize)],
    out: &mut Vec<ValueSlot>,
) -> Result<usize> {
    out.clear();
    out.resize(needed_map.len(), ValueSlot::Null);

    let mut header_pos = 0usize;
    let first = *payload.first().ok_or(Error::Corrupted(Corruption::RecordHeaderTruncated))?;
    let header_len = if first < 0x80 {
        header_pos = 1;
        first as usize
    } else {
        read_varint_at(payload, &mut header_pos, Corruption::RecordHeaderTruncated)? as usize
    };
    if header_len < header_pos || header_len > payload.len() {
        return Err(Error::Corrupted(Corruption::InvalidRecordHeaderLength));
    }

    let serial_bytes = &payload[header_pos..header_len];
    let mut serial_pos = 0usize;
    let mut value_pos = header_len;
    let mut needed_iter = needed_map.iter().copied();
    let mut next_needed = needed_iter.next();
    let mut column_count = 0usize;

    while serial_pos < serial_bytes.len() {
        let b = unsafe { *serial_bytes.get_unchecked(serial_pos) };
        let serial = if b < 0x80 {
            serial_pos += 1;
            b as u64
        } else {
            read_varint_at(serial_bytes, &mut serial_pos, Corruption::RecordHeaderTruncated)?
        };
        let col_idx = column_count as u16;
        if let Some((needed_col, out_idx)) = next_needed
            && col_idx == needed_col
        {
            let value = decode_value_ref_at(serial, payload, &mut value_pos)?;
            if out_idx >= out.len() {
                return Err(Error::Corrupted(Corruption::MappedColumnIndexOutOfBounds));
            }
            out[out_idx] = value;
            next_needed = needed_iter.next();
        } else {
            skip_value_at(serial, payload, &mut value_pos)?;
        }
        column_count += 1;
    }

    Ok(column_count)
}

fn decode_record_project_into_overflow_mapped(
    payload: &OverflowPayload<'_>,
    needed_map: &[(u16, usize)],
    out: &mut Vec<ValueSlot>,
    bytes: &mut Vec<u8>,
    serials: &mut Vec<u64>,
    offsets: &mut Vec<u32>,
) -> Result<usize> {
    out.clear();
    out.resize(needed_map.len(), ValueSlot::Null);
    bytes.clear();
    serials.clear();
    offsets.clear();

    let mut serial_cursor = OverflowCursor::new(*payload, 0)?;
    let header_len = serial_cursor.read_varint(Corruption::RecordHeaderTruncated)? as usize;
    let header_pos = serial_cursor.position();
    if header_len < header_pos || header_len > payload.total_len {
        return Err(Error::Corrupted(Corruption::InvalidRecordHeaderLength));
    }

    let mut value_pos = header_len;
    while serial_cursor.position() < header_len {
        let serial = serial_cursor.read_varint(Corruption::RecordHeaderTruncated)?;
        if serial_cursor.position() > header_len {
            return Err(Error::Corrupted(Corruption::RecordHeaderTruncated));
        }
        push_checked(serials, serial, ScratchKind::Serials)?;
        push_checked(offsets, value_pos as u32, ScratchKind::Offsets)?;
        value_pos += serial_type_len_fast(serial);
    }

    let mut value_cursor = OverflowCursor::new(*payload, header_len)?;
    let mut needed_iter = needed_map.iter().copied();
    let mut next_needed = needed_iter.next();

    for (idx, serial) in serials.iter().copied().enumerate() {
        let col_idx = idx as u16;
        if let Some((needed_col, out_idx)) = next_needed
            && col_idx == needed_col
        {
            let raw = decode_value_ref_at_cursor(serial, &mut value_cursor, bytes)?;
            if out_idx >= out.len() {
                return Err(Error::Corrupted(Corruption::MappedColumnIndexOutOfBounds));
            }
            out[out_idx] = raw;
            next_needed = needed_iter.next();
        } else {
            skip_value_at_cursor(serial, &mut value_cursor)?;
        }
    }

    Ok(serials.len())
}

pub(crate) fn decode_record_column(
    payload: PayloadRef<'_>,
    col: u16,
    bytes: &mut Vec<u8>,
) -> Result<Option<ValueSlot>> {
    match payload {
        PayloadRef::Inline(bytes) => decode_record_column_bytes(bytes, col),
        PayloadRef::Overflow(payload) => decode_record_column_overflow(&payload, col, bytes),
    }
}

pub(crate) fn decode_record_first_column(
    payload: PayloadRef<'_>,
    bytes: &mut Vec<u8>,
) -> Result<Option<ValueSlot>> {
    match payload {
        PayloadRef::Inline(bytes) => decode_record_first_column_bytes(bytes),
        PayloadRef::Overflow(payload) => decode_record_first_column_overflow(&payload, bytes),
    }
}

fn decode_record_first_column_bytes(payload: &[u8]) -> Result<Option<ValueSlot>> {
    let first = *payload.first().ok_or(Error::Corrupted(Corruption::RecordHeaderTruncated))?;
    let mut header_pos = 0usize;
    let header_len = if first < 0x80 {
        header_pos = 1;
        first as usize
    } else {
        read_varint_at(payload, &mut header_pos, Corruption::RecordHeaderTruncated)? as usize
    };
    if header_len < header_pos || header_len > payload.len() {
        return Err(Error::Corrupted(Corruption::InvalidRecordHeaderLength));
    }
    if header_pos >= header_len {
        return Ok(None);
    }

    let serial = read_varint_at(payload, &mut header_pos, Corruption::RecordHeaderTruncated)?;
    if header_pos > header_len {
        return Err(Error::Corrupted(Corruption::RecordHeaderTruncated));
    }

    let mut value_pos = header_len;
    let v = decode_value_ref_at(serial, payload, &mut value_pos)?;
    Ok(Some(v))
}

fn decode_record_first_column_overflow(
    payload: &OverflowPayload<'_>,
    bytes: &mut Vec<u8>,
) -> Result<Option<ValueSlot>> {
    bytes.clear();

    let mut serial_cursor = OverflowCursor::new(*payload, 0)?;
    let header_len = serial_cursor.read_varint(Corruption::RecordHeaderTruncated)? as usize;
    let header_pos = serial_cursor.position();
    if header_len < header_pos || header_len > payload.total_len {
        return Err(Error::Corrupted(Corruption::InvalidRecordHeaderLength));
    }
    if header_pos >= header_len {
        return Ok(None);
    }

    let serial = serial_cursor.read_varint(Corruption::RecordHeaderTruncated)?;
    if serial_cursor.position() > header_len {
        return Err(Error::Corrupted(Corruption::RecordHeaderTruncated));
    }

    let mut value_cursor = OverflowCursor::new(*payload, header_len)?;
    let raw = decode_value_ref_at_cursor(serial, &mut value_cursor, bytes)?;
    Ok(Some(raw))
}

fn decode_record_column_bytes(payload: &[u8], col: u16) -> Result<Option<ValueSlot>> {
    let first = *payload.first().ok_or(Error::Corrupted(Corruption::RecordHeaderTruncated))?;
    let mut header_pos = 0usize;
    let header_len = if first < 0x80 {
        header_pos = 1;
        first as usize
    } else {
        read_varint_at(payload, &mut header_pos, Corruption::RecordHeaderTruncated)? as usize
    };
    if header_len < header_pos || header_len > payload.len() {
        return Err(Error::Corrupted(Corruption::InvalidRecordHeaderLength));
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
            read_varint_at(serial_bytes, &mut serial_pos, Corruption::RecordHeaderTruncated)?
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

fn decode_record_column_overflow(
    payload: &OverflowPayload<'_>,
    col: u16,
    bytes: &mut Vec<u8>,
) -> Result<Option<ValueSlot>> {
    bytes.clear();

    let mut serial_cursor = OverflowCursor::new(*payload, 0)?;
    let header_len = serial_cursor.read_varint(Corruption::RecordHeaderTruncated)? as usize;
    let header_pos = serial_cursor.position();
    if header_len < header_pos || header_len > payload.total_len {
        return Err(Error::Corrupted(Corruption::InvalidRecordHeaderLength));
    }

    let mut value_cursor = OverflowCursor::new(*payload, header_len)?;
    let target = col as usize;
    let mut idx = 0usize;

    while serial_cursor.position() < header_len {
        let serial = serial_cursor.read_varint(Corruption::RecordHeaderTruncated)?;
        if serial_cursor.position() > header_len {
            return Err(Error::Corrupted(Corruption::RecordHeaderTruncated));
        }

        if idx == target {
            let raw = decode_value_ref_at_cursor(serial, &mut value_cursor, bytes)?;
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
        return Err(Error::Corrupted(Corruption::RecordPayloadShorterThanDeclared));
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

#[inline]
fn decode_value_ref_at(serial_type: u64, bytes: &[u8], pos: &mut usize) -> Result<ValueSlot> {
    // Optimized match order: most common types first, no redundant checks
    let value = match serial_type {
        0 => ValueSlot::Null,
        8 => ValueSlot::Integer(0),
        9 => ValueSlot::Integer(1),
        1 => ValueSlot::Integer(read_signed_be_at(bytes, pos, 1)?),
        2 => ValueSlot::Integer(read_signed_be_at(bytes, pos, 2)?),
        3 => ValueSlot::Integer(read_signed_be_at(bytes, pos, 3)?),
        4 => ValueSlot::Integer(read_signed_be_at(bytes, pos, 4)?),
        5 => ValueSlot::Integer(read_signed_be_at(bytes, pos, 6)?),
        6 => ValueSlot::Integer(read_signed_be_at(bytes, pos, 8)?),
        7 => ValueSlot::Real(f64::from_bits(read_u64_be_at(bytes, pos)?)),
        10 | 11 => return Err(Error::UnsupportedSerialType(serial_type)),
        // For serial >= 12: even = blob, odd = text. Length = (serial-12)/2 or (serial-13)/2
        serial => {
            let len = ((serial - 12) / 2) as usize;
            let slice = read_exact_bytes_at(bytes, pos, len)?;
            if serial & 1 == 0 {
                ValueSlot::Blob(BytesSpan::mmap(slice))
            } else {
                ValueSlot::Text(BytesSpan::mmap(slice))
            }
        }
    };

    Ok(value)
}

fn decode_value_ref_at_cursor(
    serial_type: u64,
    cursor: &mut OverflowCursor<'_>,
    bytes: &mut Vec<u8>,
) -> Result<ValueSlot> {
    let value = match serial_type {
        0 => ValueSlot::Null,
        1 => ValueSlot::Integer(cursor.read_signed_be(1)?),
        2 => ValueSlot::Integer(cursor.read_signed_be(2)?),
        3 => ValueSlot::Integer(cursor.read_signed_be(3)?),
        4 => ValueSlot::Integer(cursor.read_signed_be(4)?),
        5 => ValueSlot::Integer(cursor.read_signed_be(6)?),
        6 => ValueSlot::Integer(cursor.read_signed_be(8)?),
        7 => ValueSlot::Real(f64::from_bits(cursor.read_u64_be()?)),
        8 => ValueSlot::Integer(0),
        9 => ValueSlot::Integer(1),
        serial if serial >= 12 && serial % 2 == 0 => {
            let len = ((serial - 12) / 2) as usize;
            ValueSlot::Blob(cursor.take_span(len, bytes)?)
        }
        serial if serial >= 13 => {
            let len = ((serial - 13) / 2) as usize;
            ValueSlot::Text(cursor.take_span(len, bytes)?)
        }
        other => return Err(Error::UnsupportedSerialType(other)),
    };

    Ok(value)
}

#[inline]
fn skip_value_at_cursor(serial_type: u64, cursor: &mut OverflowCursor<'_>) -> Result<()> {
    let len = serial_type_len(serial_type)?;
    cursor.skip(len)?;
    Ok(())
}

#[inline]
fn read_signed_be_at(bytes: &[u8], pos: &mut usize, len: usize) -> Result<i64> {
    let p = *pos;
    let end = p + len;
    if end > bytes.len() {
        return Err(Error::Corrupted(Corruption::RecordPayloadShorterThanDeclared));
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

#[inline]
fn read_u64_be_at(bytes: &[u8], pos: &mut usize) -> Result<u64> {
    let p = *pos;
    let end = p + 8;
    if end > bytes.len() {
        return Err(Error::Corrupted(Corruption::RecordPayloadShorterThanDeclared));
    }
    *pos = end;
    unsafe {
        let v = ptr::read_unaligned(bytes.as_ptr().add(p) as *const u64);
        Ok(u64::from_be(v))
    }
}

fn read_u32_checked(decoder: &mut Decoder<'_>, msg: Corruption) -> Result<u32> {
    decoder.try_read_u32().ok_or(Error::Corrupted(msg))
}

fn read_varint_checked(decoder: &mut Decoder<'_>, msg: Corruption) -> Result<u64> {
    decoder.try_read_varint().ok_or(Error::Corrupted(msg))
}

#[inline]
pub(crate) fn read_varint_at(bytes: &[u8], pos: &mut usize, msg: Corruption) -> Result<u64> {
    let mut idx = *pos;
    if idx >= bytes.len() {
        return Err(Error::Corrupted(msg));
    }

    // Unrolled for common cases (1-2 byte varints are most common)
    // SAFETY: we just checked idx < bytes.len()
    let b0 = unsafe { *bytes.get_unchecked(idx) };
    idx += 1;
    if b0 < 0x80 {
        *pos = idx;
        return Ok(u64::from(b0));
    }

    if idx >= bytes.len() {
        return Err(Error::Corrupted(msg));
    }
    let b1 = unsafe { *bytes.get_unchecked(idx) };
    idx += 1;
    if b1 < 0x80 {
        *pos = idx;
        return Ok((u64::from(b0 & 0x7F) << 7) | u64::from(b1));
    }

    // Rare case: 3+ byte varint - fall back to loop
    let mut result = (u64::from(b0 & 0x7F) << 7) | u64::from(b1 & 0x7F);
    for _ in 0..6 {
        if idx >= bytes.len() {
            return Err(Error::Corrupted(msg));
        }
        let byte = unsafe { *bytes.get_unchecked(idx) };
        idx += 1;
        result = (result << 7) | u64::from(byte & 0x7F);
        if byte & 0x80 == 0 {
            *pos = idx;
            return Ok(result);
        }
    }

    if idx >= bytes.len() {
        return Err(Error::Corrupted(msg));
    }
    let byte = unsafe { *bytes.get_unchecked(idx) };
    idx += 1;
    *pos = idx;
    Ok((result << 8) | u64::from(byte))
}

#[inline(always)]
unsafe fn read_varint_unchecked_at(bytes: &[u8], pos: &mut usize) -> u64 {
    let mut idx = *pos;
    // SAFETY: caller guarantees enough bytes for full varint.

    // Unrolled for common cases (1-2 byte varints are most common)
    let b0 = unsafe { *bytes.get_unchecked(idx) };
    idx += 1;
    if b0 < 0x80 {
        *pos = idx;
        return u64::from(b0);
    }

    let b1 = unsafe { *bytes.get_unchecked(idx) };
    idx += 1;
    if b1 < 0x80 {
        *pos = idx;
        return (u64::from(b0 & 0x7F) << 7) | u64::from(b1);
    }

    // Rare case: 3+ byte varint - fall back to loop
    let mut result = (u64::from(b0 & 0x7F) << 7) | u64::from(b1 & 0x7F);
    for _ in 0..6 {
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

#[inline]
fn read_exact_bytes_at<'row>(bytes: &'row [u8], pos: &mut usize, len: usize) -> Result<&'row [u8]> {
    let start = *pos;
    let end = start + len;
    if end > bytes.len() {
        return Err(Error::Corrupted(Corruption::RecordPayloadShorterThanDeclared));
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

#[inline]
fn parse_header(page: &PageRef<'_>) -> Result<BTreeHeader> {
    let offset = page.offset();
    if offset >= page.usable_size() {
        return Err(Error::Corrupted(Corruption::PageHeaderOffsetOutOfBounds));
    }

    let bytes = page.usable_bytes();
    if offset + 8 > bytes.len() {
        return Err(Error::Corrupted(Corruption::PageHeaderTruncated));
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
                return Err(Error::Corrupted(Corruption::PageHeaderTruncated));
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
    let cell_ptrs_start = offset
        .checked_add(header_size)
        .ok_or(Error::Corrupted(Corruption::CellPointerArrayOverflow))?;

    Ok(BTreeHeader { kind, cell_count, cell_ptrs_start, right_most_child })
}

#[inline]
fn cell_ptrs<'a>(page: &'a PageRef<'_>, header: &BTreeHeader) -> Result<&'a [u8]> {
    let cell_ptrs_len = header.cell_count as usize * 2;
    let cell_ptrs_end = header
        .cell_ptrs_start
        .checked_add(cell_ptrs_len)
        .ok_or(Error::Corrupted(Corruption::CellPointerArrayOverflow))?;
    let bytes = page.usable_bytes();
    if cell_ptrs_end > bytes.len() {
        return Err(Error::Corrupted(Corruption::CellPointerArrayOutOfBounds));
    }
    Ok(&bytes[header.cell_ptrs_start..cell_ptrs_end])
}

#[inline(always)]
fn cell_ptr_at(cell_ptrs: &[u8], idx: usize) -> Result<u16> {
    let offset =
        idx.checked_mul(2).ok_or(Error::Corrupted(Corruption::CellPointerArrayOverflow))?;
    if offset + 1 >= cell_ptrs.len() {
        return Err(Error::Corrupted(Corruption::CellPointerArrayOutOfBounds));
    }
    Ok(u16::from_be_bytes([cell_ptrs[offset], cell_ptrs[offset + 1]]))
}

#[inline]
pub(crate) fn local_payload_len(usable_size: usize, payload_len: usize) -> Result<usize> {
    if payload_len == 0 {
        return Ok(0);
    }

    let u = usable_size;
    let x = u.checked_sub(35).ok_or(Error::Corrupted(Corruption::UsableSizeUnderflow))?;
    let m_base = u.checked_sub(12).ok_or(Error::Corrupted(Corruption::UsableSizeUnderflow))?;
    let m = (m_base * 32 / 255).saturating_sub(23);

    if payload_len <= x {
        return Ok(payload_len);
    }

    let k = m + ((payload_len - m) % (u - 4));
    if k <= x { Ok(k) } else { Ok(m) }
}
