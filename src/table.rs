use std::{fmt, str};

use crate::decoder::Decoder;
use crate::pager::{Page, PageId, Pager};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Pager(crate::pager::Error),
    UnsupportedPageType(u8),
    UnsupportedSerialType(u64),
    Corrupted(&'static str),
    Utf8(str::Utf8Error),
    TableNotFound(String),
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
            Self::Utf8(err) => write!(f, "{err}"),
            Self::TableNotFound(name) => {
                write!(f, "Table '{name}' not found in sqlite_schema")
            }
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
pub enum ValueRef<'a> {
    Null,
    Integer(i64),
    Real(f64),
    Text(&'a str),
    Blob(&'a [u8]),
}

impl<'a> ValueRef<'a> {
    pub fn as_text(&self) -> Option<&'a str> {
        match self {
            Self::Text(text) => Some(text),
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
            Self::Text(value) => f.write_str(value),
            Self::Blob(bytes) => Value::display_blob(bytes, f),
        }
    }
}

impl<'a> From<ValueRef<'a>> for Value {
    fn from(value: ValueRef<'a>) -> Self {
        match value {
            ValueRef::Null => Value::Null,
            ValueRef::Integer(value) => Value::Integer(value),
            ValueRef::Real(value) => Value::Real(value),
            ValueRef::Text(text) => Value::Text(text.to_owned()),
            ValueRef::Blob(bytes) => Value::Blob(bytes.to_owned()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TableRow {
    pub rowid: i64,
    pub values: Vec<Value>,
}

#[derive(Debug, Clone)]
pub struct TableRowRef<'a> {
    pub rowid: i64,
    pub values: Vec<ValueRef<'a>>,
}

pub fn read_table(pager: &Pager, page_id: PageId) -> Result<Vec<TableRow>> {
    let borrowed = read_table_ref(pager, page_id)?;

    let mut rows = Vec::with_capacity(borrowed.len());
    for row in borrowed {
        rows.push(TableRow {
            rowid: row.rowid,
            values: row.values.into_iter().map(Value::from).collect(),
        });
    }

    Ok(rows)
}

pub fn read_table_ref<'a>(pager: &'a Pager, page_id: PageId) -> Result<Vec<TableRowRef<'a>>> {
    let mut rows = Vec::new();
    read_table_page_ref(pager, page_id, &mut rows)?;
    Ok(rows)
}

fn read_table_page_ref<'a>(
    pager: &'a Pager,
    page_id: PageId,
    rows: &mut Vec<TableRowRef<'a>>,
) -> Result<()> {
    pager.read_page(page_id)?;
    let page = pager.page(page_id).ok_or(Error::Corrupted("missing page after read"))?;
    let header = parse_header(page)?;

    let cell_ptrs_len = header.cell_count as usize * 2;
    let cell_ptrs_end = header
        .cell_ptrs_start
        .checked_add(cell_ptrs_len)
        .ok_or(Error::Corrupted("cell pointer array overflow"))?;
    let bytes = page.bytes();
    if cell_ptrs_end > bytes.len() {
        return Err(Error::Corrupted("cell pointer array out of bounds"));
    }
    let cell_ptrs = &bytes[header.cell_ptrs_start..cell_ptrs_end];

    match header.kind {
        BTreeKind::TableLeaf => {
            rows.reserve(header.cell_count as usize);
            for idx in 0..header.cell_count as usize {
                let offset = u16::from_be_bytes([cell_ptrs[idx * 2], cell_ptrs[idx * 2 + 1]]);
                rows.push(read_table_cell_ref(page, offset)?);
            }
        }
        BTreeKind::TableInterior => {
            let page_len = bytes.len();
            for idx in 0..header.cell_count as usize {
                let offset = u16::from_be_bytes([cell_ptrs[idx * 2], cell_ptrs[idx * 2 + 1]]);
                if offset as usize >= page_len {
                    return Err(Error::Corrupted("cell offset out of bounds"));
                }

                let mut decoder = page.decoder().split_at(offset as usize);
                let child = decoder.read_u32();
                let child =
                    PageId::try_new(child).ok_or(Error::Corrupted("child page id is zero"))?;

                // Skip the key separating subtrees; it is not needed for scanning.
                let _ = decoder.read_varint();
                read_table_page_ref(pager, child, rows)?;
            }

            if let Some(right_most) = header.right_most_child {
                let right_most =
                    PageId::try_new(right_most).ok_or(Error::Corrupted("child page id is zero"))?;
                read_table_page_ref(pager, right_most, rows)?;
            }
        }
    }

    Ok(())
}

fn read_table_cell_ref<'a>(page: &'a Page, offset: u16) -> Result<TableRowRef<'a>> {
    if offset as usize >= page.bytes().len() {
        return Err(Error::Corrupted("cell offset out of bounds"));
    }

    let mut decoder = page.decoder().split_at(offset as usize);

    let before = decoder.remaining();
    let payload_length = decoder.read_varint();
    let payload_length_len = before - decoder.remaining();
    let payload_length =
        usize::try_from(payload_length).map_err(|_| Error::Corrupted("payload is too large"))?;

    let before = decoder.remaining();
    let rowid = decoder.read_varint() as i64;
    let rowid_len = before - decoder.remaining();

    let start = offset as usize + payload_length_len + rowid_len;
    let end =
        start.checked_add(payload_length).ok_or(Error::Corrupted("payload length overflow"))?;

    let bytes = page.bytes();
    if end > bytes.len() {
        return Err(Error::Corrupted("payload extends past page boundary"));
    }

    let values = decode_record_ref(&bytes[start..end])?;
    Ok(TableRowRef { rowid, values })
}

fn decode_record_ref<'a>(payload: &'a [u8]) -> Result<Vec<ValueRef<'a>>> {
    let mut header_decoder = Decoder::new(payload);
    let before = header_decoder.remaining();
    let header_len = header_decoder.read_varint() as usize;
    let header_len_len = before - header_decoder.remaining();

    if header_len < header_len_len || header_len > payload.len() {
        return Err(Error::Corrupted("invalid record header length"));
    }

    let mut serial_decoder = Decoder::new(&payload[header_len_len..header_len]);
    let mut value_decoder = Decoder::new(&payload[header_len..]);
    let mut values = Vec::with_capacity(8);
    while serial_decoder.remaining() > 0 {
        let serial = serial_decoder.read_varint();
        values.push(decode_value_ref(serial, &mut value_decoder)?);
    }

    Ok(values)
}

fn decode_value_ref<'a>(serial_type: u64, decoder: &mut Decoder<'a>) -> Result<ValueRef<'a>> {
    let value = match serial_type {
        0 => ValueRef::Null,
        1 => ValueRef::Integer(read_signed_be(decoder, 1)?),
        2 => ValueRef::Integer(read_signed_be(decoder, 2)?),
        3 => ValueRef::Integer(read_signed_be(decoder, 3)?),
        4 => ValueRef::Integer(read_signed_be(decoder, 4)?),
        5 => ValueRef::Integer(read_signed_be(decoder, 6)?),
        6 => ValueRef::Integer(read_signed_be(decoder, 8)?),
        7 => ValueRef::Real(f64::from_bits(read_u64_be(decoder)?)),
        8 => ValueRef::Integer(0),
        9 => ValueRef::Integer(1),
        serial if serial >= 12 && serial % 2 == 0 => {
            let len = ((serial - 12) / 2) as usize;
            ValueRef::Blob(read_exact_bytes(decoder, len)?)
        }
        serial if serial >= 13 => {
            let len = ((serial - 13) / 2) as usize;
            let text = str::from_utf8(read_exact_bytes(decoder, len)?)?;
            ValueRef::Text(text)
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

fn read_exact_bytes<'bytes>(decoder: &mut Decoder<'bytes>, len: usize) -> Result<&'bytes [u8]> {
    if decoder.remaining() < len {
        return Err(Error::Corrupted("record payload shorter than declared"));
    }

    Ok(decoder.read_bytes(len))
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

fn parse_header(page: &Page) -> Result<BTreeHeader> {
    let mut decoder = page.decoder_after_header();
    let page_type = decoder.read_u8();
    let _first_freeblock = decoder.read_u16();
    let cell_count = decoder.read_u16();
    let _start_of_cell_content = decoder.read_u16();
    let _fragmented_free_bytes = decoder.read_u8();

    let kind = match page_type {
        0x0D => BTreeKind::TableLeaf,
        0x05 => BTreeKind::TableInterior,
        _ => return Err(Error::UnsupportedPageType(page_type)),
    };

    let right_most_child = match kind {
        BTreeKind::TableInterior => Some(decoder.read_u32()),
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
