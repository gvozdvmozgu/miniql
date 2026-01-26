use std::cmp::Ordering;

use crate::join::Error;
use crate::pager::{PageId, PageRef, Pager};
use crate::table::{self, ValueRef};

/// Result type for index operations.
pub type Result<T> = table::Result<T>;

const MAX_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;

/// Scratch buffers for index cursor operations.
#[derive(Debug)]
pub struct IndexScratch {
    stack: Vec<StackEntry>,
    bytes: Vec<u8>,
    serials: Vec<u64>,
}

impl IndexScratch {
    /// Create an empty scratch buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a scratch buffer with capacity hints.
    pub fn with_capacity(values: usize, overflow: usize) -> Self {
        Self {
            stack: Vec::with_capacity(64),
            bytes: Vec::with_capacity(overflow),
            serials: Vec::with_capacity(values),
        }
    }
}

impl Default for IndexScratch {
    fn default() -> Self {
        Self::with_capacity(0, 0)
    }
}

#[derive(Clone, Copy, Debug)]
struct StackEntry {
    page_id: PageId,
    child_slot: usize,
}

#[derive(Clone, Copy, Debug)]
struct LeafPos {
    page_id: PageId,
    cell_index: usize,
}

#[derive(Clone, Copy)]
struct IndexCellRef<'row> {
    child: Option<PageId>,
    payload: table::PayloadRef<'row>,
}

impl<'row> IndexCellRef<'row> {
    #[inline]
    fn child(self) -> Option<PageId> {
        self.child
    }

    #[inline]
    fn payload(self) -> table::PayloadRef<'row> {
        self.payload
    }
}

/// Cursor over an index b-tree.
pub struct IndexCursor<'db, 'scratch> {
    pager: &'db Pager,
    root: PageId,
    key_col: u16,
    scratch: &'scratch mut IndexScratch,
    leaf: Option<LeafPos>,
}

impl<'db, 'scratch> IndexCursor<'db, 'scratch> {
    /// Create a new cursor for an index root and key column.
    pub fn new(
        pager: &'db Pager,
        root: PageId,
        key_col: u16,
        scratch: &'scratch mut IndexScratch,
    ) -> Self {
        Self { pager, root, key_col, scratch, leaf: None }
    }

    /// Seek to the first entry with key >= `target`.
    pub fn seek_ge(&mut self, target: ValueRef<'_>) -> Result<bool> {
        self.scratch.stack.clear();
        self.leaf = None;

        let mut page_id = self.root;
        let max_pages = self.pager.page_count().max(1);
        let mut seen_pages = 0u32;

        loop {
            seen_pages += 1;
            if seen_pages > max_pages {
                return Err(table::Error::Corrupted("btree page cycle detected"));
            }

            let page = self.pager.page(page_id)?;
            let header = parse_header(&page)?;
            let cell_ptrs = cell_ptrs(&page, &header)?;

            match header.kind {
                IndexKind::Leaf => {
                    let cell_count = header.cell_count as usize;
                    let idx = self.lower_bound_leaf(&page, cell_ptrs, cell_count, target)?;
                    self.leaf = Some(LeafPos { page_id, cell_index: idx });

                    if idx < cell_count {
                        return Ok(true);
                    }

                    return self.advance_from_leaf_end();
                }
                IndexKind::Interior => {
                    let cell_count = header.cell_count as usize;
                    let mut lo = 0usize;
                    let mut hi = cell_count;
                    let mut candidate_child: Option<PageId> = None;

                    while lo < hi {
                        let mid = (lo + hi) / 2;
                        let offset = cell_ptr_at(cell_ptrs, mid)?;
                        let cell = read_index_interior_cell(self.pager, &page, offset)?;
                        let child =
                            cell.child().ok_or(table::Error::Corrupted("missing child pointer"))?;
                        let key = decode_key_from_payload(
                            self.key_col,
                            cell.payload(),
                            &mut self.scratch.bytes,
                        )?;
                        let cmp = compare_total(target, key);

                        if cmp == Ordering::Greater {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                            candidate_child = Some(child);
                        }
                    }

                    if lo < cell_count {
                        let child = candidate_child
                            .ok_or(table::Error::Corrupted("missing child pointer"))?;
                        self.scratch.stack.push(StackEntry { page_id, child_slot: lo });
                        page_id = child;
                    } else {
                        let right_most = header
                            .right_most_child
                            .ok_or(table::Error::Corrupted("missing right-most child pointer"))?;
                        let right_most = PageId::try_new(right_most)
                            .ok_or(table::Error::Corrupted("child page id is zero"))?;
                        self.scratch.stack.push(StackEntry { page_id, child_slot: cell_count });
                        page_id = right_most;
                    }
                }
            }
        }
    }

    /// Advance forward until the current key is >= `target`.
    pub fn advance_to_ge(&mut self, target: ValueRef<'_>) -> Result<bool> {
        if self.leaf.is_none() {
            return self.seek_ge(target);
        }

        loop {
            let cmp = self.compare_current_key(target)?;
            if cmp != Ordering::Greater {
                return Ok(true);
            }
            if !self.next()? {
                return Ok(false);
            }
        }
    }

    #[allow(clippy::should_implement_trait)]
    /// Advance to the next entry.
    pub fn next(&mut self) -> Result<bool> {
        let Some(leaf) = self.leaf else {
            return Ok(false);
        };

        let page = self.pager.page(leaf.page_id)?;
        let header = parse_header(&page)?;
        let cell_count = header.cell_count as usize;

        if leaf.cell_index + 1 < cell_count {
            self.leaf = Some(LeafPos { page_id: leaf.page_id, cell_index: leaf.cell_index + 1 });
            return Ok(true);
        }

        self.leaf = Some(LeafPos { page_id: leaf.page_id, cell_index: cell_count });
        self.advance_from_leaf_end()
    }

    /// Check whether the current key equals `target`.
    pub fn key_eq(&mut self, target: ValueRef<'_>) -> Result<bool> {
        let Some(leaf) = self.leaf else {
            return Ok(false);
        };

        let page = self.pager.page(leaf.page_id)?;
        let header = parse_header(&page)?;
        if leaf.cell_index >= header.cell_count as usize {
            return Ok(false);
        }
        let cell_ptrs = cell_ptrs(&page, &header)?;
        let offset = cell_ptr_at(cell_ptrs, leaf.cell_index)?;
        let cell = read_index_leaf_cell(self.pager, &page, offset)?;
        let key = decode_key_from_payload(self.key_col, cell.payload(), &mut self.scratch.bytes)?;
        if matches!(key, ValueRef::Real(value) if value.is_nan())
            || matches!(target, ValueRef::Real(value) if value.is_nan())
        {
            return Ok(false);
        }
        if matches!(key, ValueRef::Null) || matches!(target, ValueRef::Null) {
            return Ok(false);
        }

        Ok(compare_total(target, key) == Ordering::Equal)
    }

    /// Return the rowid at the current cursor position.
    pub fn current_rowid(&mut self) -> Result<i64> {
        let Some(leaf) = self.leaf else {
            return Err(table::Error::Corrupted("index cursor not positioned"));
        };

        let page = self.pager.page(leaf.page_id)?;
        let header = parse_header(&page)?;
        if leaf.cell_index >= header.cell_count as usize {
            return Err(table::Error::Corrupted("index cursor past end"));
        }
        let cell_ptrs = cell_ptrs(&page, &header)?;
        let offset = cell_ptr_at(cell_ptrs, leaf.cell_index)?;
        let cell = read_index_leaf_cell(self.pager, &page, offset)?;
        let count = table::record_column_count(cell.payload())?;
        let last =
            count.checked_sub(1).ok_or_else(|| table::Error::from(Error::MissingIndexRowId))?;
        let last = u16::try_from(last).map_err(|_| table::Error::from(Error::MissingIndexRowId))?;
        let Some(value) =
            table::decode_record_column(cell.payload(), last, &mut self.scratch.bytes)?
        else {
            return Err(Error::MissingIndexRowId.into());
        };

        match unsafe { value.as_value_ref() } {
            ValueRef::Integer(rowid) => Ok(rowid),
            _ => Err(Error::MissingIndexRowId.into()),
        }
    }

    /// Execute `f` with the payload and rowid at the current cursor position.
    pub fn with_current_payload_and_rowid<T, F>(&mut self, f: F) -> Result<T>
    where
        F: for<'row> FnOnce(table::PayloadRef<'row>, i64) -> Result<T>,
    {
        let Some(leaf) = self.leaf else {
            return Err(table::Error::Corrupted("index cursor not positioned"));
        };

        let page = self.pager.page(leaf.page_id)?;
        let header = parse_header(&page)?;
        if leaf.cell_index >= header.cell_count as usize {
            return Err(table::Error::Corrupted("index cursor past end"));
        }
        let cell_ptrs = cell_ptrs(&page, &header)?;
        let offset = cell_ptr_at(cell_ptrs, leaf.cell_index)?;
        let cell = read_index_leaf_cell(self.pager, &page, offset)?;
        let payload = cell.payload();

        let count = table::record_column_count(payload)?;
        let last =
            count.checked_sub(1).ok_or_else(|| table::Error::from(Error::MissingIndexRowId))?;
        let last = u16::try_from(last).map_err(|_| table::Error::from(Error::MissingIndexRowId))?;
        let Some(value) = table::decode_record_column(payload, last, &mut self.scratch.bytes)?
        else {
            return Err(Error::MissingIndexRowId.into());
        };

        let rowid = match unsafe { value.as_value_ref() } {
            ValueRef::Integer(rowid) => rowid,
            _ => return Err(Error::MissingIndexRowId.into()),
        };

        f(payload, rowid)
    }

    fn lower_bound_leaf(
        &mut self,
        page: &PageRef<'_>,
        cell_ptrs: &[u8],
        cell_count: usize,
        target: ValueRef<'_>,
    ) -> Result<usize> {
        let mut lo = 0usize;
        let mut hi = cell_count;

        while lo < hi {
            let mid = (lo + hi) / 2;
            let offset = cell_ptr_at(cell_ptrs, mid)?;
            let cell = read_index_leaf_cell(self.pager, page, offset)?;
            let key =
                decode_key_from_payload(self.key_col, cell.payload(), &mut self.scratch.bytes)?;
            let cmp = compare_total(target, key);
            if cmp == Ordering::Greater {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        Ok(lo)
    }

    fn compare_current_key(&mut self, target: ValueRef<'_>) -> Result<Ordering> {
        let Some(leaf) = self.leaf else {
            return Err(table::Error::Corrupted("index cursor not positioned"));
        };

        let page = self.pager.page(leaf.page_id)?;
        let header = parse_header(&page)?;
        if leaf.cell_index >= header.cell_count as usize {
            return Err(table::Error::Corrupted("index cursor past end"));
        }
        let cell_ptrs = cell_ptrs(&page, &header)?;
        let offset = cell_ptr_at(cell_ptrs, leaf.cell_index)?;
        let cell = read_index_leaf_cell(self.pager, &page, offset)?;
        let key = decode_key_from_payload(self.key_col, cell.payload(), &mut self.scratch.bytes)?;
        Ok(compare_total(target, key))
    }

    fn advance_from_leaf_end(&mut self) -> Result<bool> {
        while let Some(entry) = self.scratch.stack.pop() {
            let page = self.pager.page(entry.page_id)?;
            let header = parse_header(&page)?;
            let cell_count = header.cell_count as usize;

            if entry.child_slot < cell_count {
                let next_slot = entry.child_slot + 1;
                self.scratch
                    .stack
                    .push(StackEntry { page_id: entry.page_id, child_slot: next_slot });

                let child = child_page_for_slot(self.pager, &page, &header, next_slot)?;
                let leaf = self.descend_leftmost(child)?;
                self.leaf = Some(leaf);
                return Ok(true);
            }
        }

        self.leaf = None;
        Ok(false)
    }

    fn descend_leftmost(&mut self, mut page_id: PageId) -> Result<LeafPos> {
        let max_pages = self.pager.page_count().max(1);
        let mut seen_pages = 0u32;

        loop {
            seen_pages += 1;
            if seen_pages > max_pages {
                return Err(table::Error::Corrupted("btree page cycle detected"));
            }

            let page = self.pager.page(page_id)?;
            let header = parse_header(&page)?;
            if matches!(header.kind, IndexKind::Leaf) {
                return Ok(LeafPos { page_id, cell_index: 0 });
            }

            let cell_ptrs = cell_ptrs(&page, &header)?;
            if header.cell_count == 0 {
                return Err(table::Error::Corrupted("interior index page has no cells"));
            }
            let offset = cell_ptr_at(cell_ptrs, 0)?;
            let cell = read_index_interior_cell(self.pager, &page, offset)?;
            let child = cell.child().ok_or(table::Error::Corrupted("missing child pointer"))?;
            self.scratch.stack.push(StackEntry { page_id, child_slot: 0 });
            page_id = child;
        }
    }
}

pub(crate) fn index_key_len(
    pager: &Pager,
    root: PageId,
    scratch: &mut IndexScratch,
) -> Result<Option<usize>> {
    scratch.stack.clear();
    scratch.bytes.clear();
    scratch.serials.clear();

    let mut page_id = root;
    let max_pages = pager.page_count().max(1);
    let mut seen_pages = 0u32;

    loop {
        seen_pages += 1;
        if seen_pages > max_pages {
            return Err(table::Error::Corrupted("btree page cycle detected"));
        }

        let page = pager.page(page_id)?;
        let header = parse_header(&page)?;

        if header.cell_count == 0 {
            return match header.kind {
                IndexKind::Leaf => Ok(None),
                IndexKind::Interior => {
                    Err(table::Error::Corrupted("interior index page has no cells"))
                }
            };
        }

        match header.kind {
            IndexKind::Leaf => {
                let cell_ptrs = cell_ptrs(&page, &header)?;
                let offset = cell_ptr_at(cell_ptrs, 0)?;
                let cell = read_index_leaf_cell(pager, &page, offset)?;
                let count = table::record_column_count(cell.payload())?;
                if count == 0 {
                    return Err(table::Error::Corrupted("index record has no columns"));
                }
                return Ok(Some(count.saturating_sub(1)));
            }
            IndexKind::Interior => {
                let cell_ptrs = cell_ptrs(&page, &header)?;
                let offset = cell_ptr_at(cell_ptrs, 0)?;
                let cell = read_index_interior_cell(pager, &page, offset)?;
                page_id = cell.child().ok_or(table::Error::Corrupted("missing child pointer"))?;
            }
        }
    }
}

fn compare_total(left: ValueRef<'_>, right: ValueRef<'_>) -> Ordering {
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

fn decode_key_from_payload<'row>(
    key_col: u16,
    payload: table::PayloadRef<'row>,
    bytes: &'row mut Vec<u8>,
) -> Result<ValueRef<'row>> {
    let raw = if key_col == 0 {
        table::decode_record_first_column(payload, bytes)?
    } else {
        table::decode_record_column(payload, key_col, bytes)?
    };
    let Some(raw) = raw else {
        return Err(Error::IndexKeyNotComparable.into());
    };
    Ok(unsafe { raw.as_value_ref() })
}

#[derive(Clone, Copy, Debug)]
enum IndexKind {
    Leaf,
    Interior,
}

struct IndexHeader {
    kind: IndexKind,
    cell_count: u16,
    cell_ptrs_start: usize,
    right_most_child: Option<u32>,
}

fn parse_header(page: &PageRef<'_>) -> Result<IndexHeader> {
    let offset = page.offset();
    if offset >= page.usable_size() {
        return Err(table::Error::Corrupted("page header offset out of bounds"));
    }

    let bytes = page.usable_bytes();
    if offset + 8 > bytes.len() {
        return Err(table::Error::Corrupted("page header truncated"));
    }

    let page_type = bytes[offset];
    let _first_freeblock = u16::from_be_bytes([bytes[offset + 1], bytes[offset + 2]]);
    let cell_count = u16::from_be_bytes([bytes[offset + 3], bytes[offset + 4]]);
    let _start_of_cell_content = u16::from_be_bytes([bytes[offset + 5], bytes[offset + 6]]);
    let _fragmented_free_bytes = bytes[offset + 7];

    let kind = match page_type {
        0x0A => IndexKind::Leaf,
        0x02 => IndexKind::Interior,
        other => return Err(table::Error::UnsupportedPageType(other)),
    };

    let right_most_child = match kind {
        IndexKind::Interior => {
            if offset + 12 > bytes.len() {
                return Err(table::Error::Corrupted("page header truncated"));
            }
            Some(u32::from_be_bytes([
                bytes[offset + 8],
                bytes[offset + 9],
                bytes[offset + 10],
                bytes[offset + 11],
            ]))
        }
        IndexKind::Leaf => None,
    };

    let header_size = match kind {
        IndexKind::Leaf => 8,
        IndexKind::Interior => 12,
    };
    let cell_ptrs_start = offset
        .checked_add(header_size)
        .ok_or(table::Error::Corrupted("cell pointer array overflow"))?;

    Ok(IndexHeader { kind, cell_count, cell_ptrs_start, right_most_child })
}

fn cell_ptrs<'a>(page: &'a PageRef<'_>, header: &IndexHeader) -> Result<&'a [u8]> {
    let cell_ptrs_len = header.cell_count as usize * 2;
    let cell_ptrs_end = header
        .cell_ptrs_start
        .checked_add(cell_ptrs_len)
        .ok_or(table::Error::Corrupted("cell pointer array overflow"))?;
    let bytes = page.usable_bytes();
    if cell_ptrs_end > bytes.len() {
        return Err(table::Error::Corrupted("cell pointer array out of bounds"));
    }
    Ok(&bytes[header.cell_ptrs_start..cell_ptrs_end])
}

#[inline(always)]
fn cell_ptr_at(cell_ptrs: &[u8], idx: usize) -> Result<u16> {
    let offset =
        idx.checked_mul(2).ok_or(table::Error::Corrupted("cell pointer array overflow"))?;
    if offset + 1 >= cell_ptrs.len() {
        return Err(table::Error::Corrupted("cell pointer array out of bounds"));
    }
    Ok(u16::from_be_bytes([cell_ptrs[offset], cell_ptrs[offset + 1]]))
}

fn read_index_leaf_cell<'row>(
    pager: &'row Pager,
    page: &'row PageRef<'_>,
    offset: u16,
) -> Result<IndexCellRef<'row>> {
    read_index_cell(pager, page, offset, false)
}

fn read_index_interior_cell<'row>(
    pager: &'row Pager,
    page: &'row PageRef<'_>,
    offset: u16,
) -> Result<IndexCellRef<'row>> {
    read_index_cell(pager, page, offset, true)
}

fn read_index_cell<'row>(
    pager: &'row Pager,
    page: &'row PageRef<'_>,
    offset: u16,
    has_child: bool,
) -> Result<IndexCellRef<'row>> {
    let usable = page.usable_bytes();
    let mut pos = offset as usize;
    if pos >= usable.len() {
        return Err(table::Error::Corrupted("cell offset out of bounds"));
    }

    let child = if has_child {
        if pos + 4 > usable.len() {
            return Err(table::Error::Corrupted("cell child pointer truncated"));
        }
        let child =
            u32::from_be_bytes([usable[pos], usable[pos + 1], usable[pos + 2], usable[pos + 3]]);
        pos += 4;
        Some(PageId::try_new(child).ok_or(table::Error::Corrupted("child page id is zero"))?)
    } else {
        None
    };

    let payload_length = table::read_varint_at(usable, &mut pos, "cell payload length truncated")?;
    let payload_length = usize::try_from(payload_length)
        .map_err(|_| table::Error::Corrupted("payload is too large"))?;
    if payload_length > MAX_PAYLOAD_BYTES {
        return Err(table::Error::PayloadTooLarge(payload_length));
    }

    let start = pos;

    let usable_size = page.usable_size();
    let x = usable_size.checked_sub(35).ok_or(table::Error::Corrupted("usable size underflow"))?;
    if payload_length <= x {
        let end = start + payload_length;
        if end > usable.len() {
            return Err(table::Error::Corrupted("payload extends past page boundary"));
        }
        return Ok(IndexCellRef { child, payload: table::PayloadRef::Inline(&usable[start..end]) });
    }

    let local_len = table::local_payload_len(usable_size, payload_length)?;
    let end_local = start + local_len;
    if end_local > usable.len() {
        return Err(table::Error::Corrupted("payload extends past page boundary"));
    }

    let overflow_end = end_local + 4;
    if overflow_end > usable.len() {
        return Err(table::Error::Corrupted("overflow pointer out of bounds"));
    }
    let overflow_page = u32::from_be_bytes(usable[end_local..overflow_end].try_into().unwrap());
    if overflow_page == 0 {
        return Err(table::Error::OverflowChainTruncated);
    }
    let payload = table::OverflowPayload::new(
        pager,
        payload_length,
        &usable[start..end_local],
        overflow_page,
    );
    Ok(IndexCellRef { child, payload: table::PayloadRef::Overflow(payload) })
}

fn child_page_for_slot(
    pager: &Pager,
    page: &PageRef<'_>,
    header: &IndexHeader,
    child_slot: usize,
) -> Result<PageId> {
    if child_slot < header.cell_count as usize {
        let cell_ptrs = cell_ptrs(page, header)?;
        let offset = cell_ptr_at(cell_ptrs, child_slot)?;
        let cell = read_index_interior_cell(pager, page, offset)?;
        cell.child().ok_or(table::Error::Corrupted("missing child pointer"))
    } else {
        let right_most = header
            .right_most_child
            .ok_or(table::Error::Corrupted("missing right-most child pointer"))?;
        PageId::try_new(right_most).ok_or(table::Error::Corrupted("child page id is zero"))
    }
}
