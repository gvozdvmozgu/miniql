use crate::pager::PageRef;
use crate::table::{self, Corruption};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BTreeKind {
    TableLeaf,
    TableInterior,
    IndexLeaf,
    IndexInterior,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BTreeHeader {
    pub(crate) kind: BTreeKind,
    pub(crate) cell_count: u16,
    pub(crate) cell_ptrs_start: usize,
    pub(crate) right_most_child: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum BTreePageKind {
    Table,
    Index,
}

#[inline]
pub(crate) fn parse_table_header(page: &PageRef<'_>) -> table::Result<BTreeHeader> {
    parse_header(page, BTreePageKind::Table)
}

#[inline]
pub(crate) fn parse_index_header(page: &PageRef<'_>) -> table::Result<BTreeHeader> {
    parse_header(page, BTreePageKind::Index)
}

#[inline]
pub(crate) fn cell_ptrs<'a>(
    page: &'a PageRef<'_>,
    header: &BTreeHeader,
) -> table::Result<&'a [u8]> {
    let cell_ptrs_len = header.cell_count as usize * 2;
    let cell_ptrs_end = header
        .cell_ptrs_start
        .checked_add(cell_ptrs_len)
        .ok_or(table::Error::Corrupted(Corruption::CellPointerArrayOverflow))?;
    let bytes = page.usable_bytes();
    if cell_ptrs_end > bytes.len() {
        return Err(table::Error::Corrupted(Corruption::CellPointerArrayOutOfBounds));
    }
    Ok(&bytes[header.cell_ptrs_start..cell_ptrs_end])
}

#[inline(always)]
pub(crate) fn cell_ptr_at(cell_ptrs: &[u8], idx: usize) -> table::Result<u16> {
    let offset =
        idx.checked_mul(2).ok_or(table::Error::Corrupted(Corruption::CellPointerArrayOverflow))?;
    if offset + 1 >= cell_ptrs.len() {
        return Err(table::Error::Corrupted(Corruption::CellPointerArrayOutOfBounds));
    }
    Ok(u16::from_be_bytes([cell_ptrs[offset], cell_ptrs[offset + 1]]))
}

#[inline]
fn parse_header(page: &PageRef<'_>, kind: BTreePageKind) -> table::Result<BTreeHeader> {
    let offset = page.offset();
    if offset >= page.usable_size() {
        return Err(table::Error::Corrupted(Corruption::PageHeaderOffsetOutOfBounds));
    }

    let bytes = page.usable_bytes();
    if offset + 8 > bytes.len() {
        return Err(table::Error::Corrupted(Corruption::PageHeaderTruncated));
    }

    let page_type = bytes[offset];
    let _first_freeblock = u16::from_be_bytes([bytes[offset + 1], bytes[offset + 2]]);
    let cell_count = u16::from_be_bytes([bytes[offset + 3], bytes[offset + 4]]);
    let _start_of_cell_content = u16::from_be_bytes([bytes[offset + 5], bytes[offset + 6]]);
    let _fragmented_free_bytes = bytes[offset + 7];

    let kind = match (kind, page_type) {
        (BTreePageKind::Table, 0x0D) => BTreeKind::TableLeaf,
        (BTreePageKind::Table, 0x05) => BTreeKind::TableInterior,
        (BTreePageKind::Index, 0x0A) => BTreeKind::IndexLeaf,
        (BTreePageKind::Index, 0x02) => BTreeKind::IndexInterior,
        (_, other) => return Err(table::Error::UnsupportedPageType(other)),
    };

    let right_most_child = match kind {
        BTreeKind::TableInterior | BTreeKind::IndexInterior => {
            if offset + 12 > bytes.len() {
                return Err(table::Error::Corrupted(Corruption::PageHeaderTruncated));
            }
            Some(u32::from_be_bytes([
                bytes[offset + 8],
                bytes[offset + 9],
                bytes[offset + 10],
                bytes[offset + 11],
            ]))
        }
        BTreeKind::TableLeaf | BTreeKind::IndexLeaf => None,
    };

    let header_size = match kind {
        BTreeKind::TableLeaf | BTreeKind::IndexLeaf => 8,
        BTreeKind::TableInterior | BTreeKind::IndexInterior => 12,
    };
    let cell_ptrs_start = offset
        .checked_add(header_size)
        .ok_or(table::Error::Corrupted(Corruption::CellPointerArrayOverflow))?;

    Ok(BTreeHeader { kind, cell_count, cell_ptrs_start, right_most_child })
}
