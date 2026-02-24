use std::cmp::Ordering;

use crate::btree::{self, BTreeHeader, BTreeKind};
use crate::compare::compare_value_refs;
use crate::error::JoinError as Error;
use crate::pager::{PageId, PageRef, Pager};
use crate::table::{self, Corruption, ValueRef};

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
                return Err(table::Error::Corrupted(Corruption::BtreePageCycleDetected));
            }

            let page = self.pager.page(page_id)?;
            let header = btree::parse_index_header(&page)?;
            let cell_ptrs = btree::cell_ptrs(&page, &header)?;

            match header.kind {
                BTreeKind::IndexLeaf => {
                    let cell_count = header.cell_count as usize;
                    let idx = self.lower_bound_leaf(&page, cell_ptrs, cell_count, target)?;
                    self.leaf = Some(LeafPos { page_id, cell_index: idx });

                    if idx < cell_count {
                        return Ok(true);
                    }

                    return self.advance_from_leaf_end();
                }
                BTreeKind::IndexInterior => {
                    let cell_count = header.cell_count as usize;
                    let (slot, child) =
                        self.lower_bound_interior_child(&page, cell_ptrs, cell_count, target)?;

                    self.scratch.stack.push(StackEntry { page_id, child_slot: slot });

                    page_id = match child {
                        Some(child) => child,
                        None => page_id_from_u32(header.right_most_child.ok_or(
                            table::Error::Corrupted(Corruption::MissingRightMostChildPointer),
                        )?)?,
                    };
                }
                BTreeKind::TableLeaf | BTreeKind::TableInterior => {
                    unreachable!("table btree header returned for index page")
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
        let header = btree::parse_index_header(&page)?;
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
        let Some(eq) = self.with_current_leaf_cell_if_any(|cell, bytes, key_col| {
            let key = decode_key_from_payload(key_col, cell.payload(), bytes)?;

            if is_not_comparable(key) || is_not_comparable(target) {
                return Ok(false);
            }

            Ok(compare_value_refs(target, key) == Ordering::Equal)
        })?
        else {
            return Ok(false);
        };

        Ok(eq)
    }

    /// Return the rowid at the current cursor position.
    pub fn current_rowid(&mut self) -> Result<i64> {
        self.with_current_leaf_cell(|cell, bytes, _key_col| {
            decode_index_rowid(cell.payload(), bytes)
        })
    }

    /// Execute `f` with the payload and rowid at the current cursor position.
    pub fn with_current_payload_and_rowid<T, F>(&mut self, f: F) -> Result<T>
    where
        F: for<'row> FnOnce(table::PayloadRef<'row>, i64) -> Result<T>,
    {
        self.with_current_leaf_cell(|cell, bytes, _key_col| {
            let payload = cell.payload();
            let rowid = decode_index_rowid(payload, bytes)?;
            f(payload, rowid)
        })
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
            let offset = btree::cell_ptr_at(cell_ptrs, mid)?;
            let cell = read_index_leaf_cell(self.pager, page, offset)?;
            let key =
                decode_key_from_payload(self.key_col, cell.payload(), &mut self.scratch.bytes)?;
            let cmp = compare_value_refs(target, key);
            if cmp == Ordering::Greater {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        Ok(lo)
    }

    /// Like `lower_bound_leaf`, but for interior pages: returns
    /// (slot, child_for_slot_or_none_if_right_most).
    fn lower_bound_interior_child(
        &mut self,
        page: &PageRef<'_>,
        cell_ptrs: &[u8],
        cell_count: usize,
        target: ValueRef<'_>,
    ) -> Result<(usize, Option<PageId>)> {
        let mut lo = 0usize;
        let mut hi = cell_count;
        let mut candidate_child: Option<PageId> = None;

        while lo < hi {
            let mid = (lo + hi) / 2;
            let offset = btree::cell_ptr_at(cell_ptrs, mid)?;
            let cell = read_index_interior_cell(self.pager, page, offset)?;
            let child =
                cell.child().ok_or(table::Error::Corrupted(Corruption::MissingChildPointer))?;
            let key =
                decode_key_from_payload(self.key_col, cell.payload(), &mut self.scratch.bytes)?;
            let cmp = compare_value_refs(target, key);

            if cmp == Ordering::Greater {
                lo = mid + 1;
            } else {
                hi = mid;
                candidate_child = Some(child);
            }
        }

        if lo < cell_count {
            Ok((
                lo,
                Some(
                    candidate_child
                        .ok_or(table::Error::Corrupted(Corruption::MissingChildPointer))?,
                ),
            ))
        } else {
            Ok((lo, None)) // right-most
        }
    }

    fn compare_current_key(&mut self, target: ValueRef<'_>) -> Result<Ordering> {
        self.with_current_leaf_cell(|cell, bytes, key_col| {
            let key = decode_key_from_payload(key_col, cell.payload(), bytes)?;
            Ok(compare_value_refs(target, key))
        })
    }

    /// Strict: must be positioned and not past end.
    #[inline]
    fn with_current_leaf_cell<T, F>(&mut self, f: F) -> Result<T>
    where
        F: for<'row> FnOnce(IndexCellRef<'row>, &mut Vec<u8>, u16) -> Result<T>,
    {
        let Some(leaf) = self.leaf else {
            return Err(table::Error::Corrupted(Corruption::IndexCursorNotPositioned));
        };

        let page = self.pager.page(leaf.page_id)?;
        let header = btree::parse_index_header(&page)?;

        if leaf.cell_index >= header.cell_count as usize {
            return Err(table::Error::Corrupted(Corruption::IndexCursorPastEnd));
        }

        let cell_ptrs = btree::cell_ptrs(&page, &header)?;
        let offset = btree::cell_ptr_at(cell_ptrs, leaf.cell_index)?;
        let cell = read_index_leaf_cell(self.pager, &page, offset)?;

        let key_col = self.key_col;
        let bytes = &mut self.scratch.bytes;
        f(cell, bytes, key_col)
    }

    /// Lenient: returns `Ok(None)` if not positioned or past end.
    #[inline]
    fn with_current_leaf_cell_if_any<T, F>(&mut self, f: F) -> Result<Option<T>>
    where
        F: for<'row> FnOnce(IndexCellRef<'row>, &mut Vec<u8>, u16) -> Result<T>,
    {
        let Some(leaf) = self.leaf else {
            return Ok(None);
        };

        let page = self.pager.page(leaf.page_id)?;
        let header = btree::parse_index_header(&page)?;

        if leaf.cell_index >= header.cell_count as usize {
            return Ok(None);
        }

        let cell_ptrs = btree::cell_ptrs(&page, &header)?;
        let offset = btree::cell_ptr_at(cell_ptrs, leaf.cell_index)?;
        let cell = read_index_leaf_cell(self.pager, &page, offset)?;

        let key_col = self.key_col;
        let bytes = &mut self.scratch.bytes;
        Ok(Some(f(cell, bytes, key_col)?))
    }

    fn advance_from_leaf_end(&mut self) -> Result<bool> {
        while let Some(entry) = self.scratch.stack.pop() {
            let page = self.pager.page(entry.page_id)?;
            let header = btree::parse_index_header(&page)?;
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
                return Err(table::Error::Corrupted(Corruption::BtreePageCycleDetected));
            }

            let page = self.pager.page(page_id)?;
            let header = btree::parse_index_header(&page)?;
            if matches!(header.kind, BTreeKind::IndexLeaf) {
                return Ok(LeafPos { page_id, cell_index: 0 });
            }

            let cell_ptrs = btree::cell_ptrs(&page, &header)?;
            if header.cell_count == 0 {
                return Err(table::Error::Corrupted(Corruption::InteriorIndexPageHasNoCells));
            }
            let offset = btree::cell_ptr_at(cell_ptrs, 0)?;
            let cell = read_index_interior_cell(self.pager, &page, offset)?;
            let child =
                cell.child().ok_or(table::Error::Corrupted(Corruption::MissingChildPointer))?;
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
            return Err(table::Error::Corrupted(Corruption::BtreePageCycleDetected));
        }

        let page = pager.page(page_id)?;
        let header = btree::parse_index_header(&page)?;

        if header.cell_count == 0 {
            return match header.kind {
                BTreeKind::IndexLeaf => Ok(None),
                BTreeKind::IndexInterior => {
                    Err(table::Error::Corrupted(Corruption::InteriorIndexPageHasNoCells))
                }
                BTreeKind::TableLeaf | BTreeKind::TableInterior => {
                    unreachable!("table btree header returned for index page")
                }
            };
        }

        match header.kind {
            BTreeKind::IndexLeaf => {
                let cell_ptrs = btree::cell_ptrs(&page, &header)?;
                let offset = btree::cell_ptr_at(cell_ptrs, 0)?;
                let cell = read_index_leaf_cell(pager, &page, offset)?;
                let count = table::record_column_count(cell.payload())?;
                if count == 0 {
                    return Err(table::Error::Corrupted(Corruption::IndexRecordHasNoColumns));
                }
                return Ok(Some(count.saturating_sub(1)));
            }
            BTreeKind::IndexInterior => {
                let cell_ptrs = btree::cell_ptrs(&page, &header)?;
                let offset = btree::cell_ptr_at(cell_ptrs, 0)?;
                let cell = read_index_interior_cell(pager, &page, offset)?;
                page_id =
                    cell.child().ok_or(table::Error::Corrupted(Corruption::MissingChildPointer))?;
            }
            BTreeKind::TableLeaf | BTreeKind::TableInterior => {
                unreachable!("table btree header returned for index page")
            }
        }
    }
}

#[inline]
fn is_not_comparable(v: ValueRef<'_>) -> bool {
    matches!(v, ValueRef::Null) || matches!(v, ValueRef::Real(x) if x.is_nan())
}

#[inline]
fn page_id_from_u32(id: u32) -> Result<PageId> {
    PageId::try_new(id).ok_or(table::Error::Corrupted(Corruption::ChildPageIdZero))
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
    // SAFETY: `bytes` is borrowed for `'row`, and the returned ref may point into
    // it.
    Ok(unsafe { raw.as_value_ref_with_scratch(bytes.as_slice()) })
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
        return Err(table::Error::Corrupted(Corruption::CellOffsetOutOfBounds));
    }

    let child = if has_child {
        if pos + 4 > usable.len() {
            return Err(table::Error::Corrupted(Corruption::CellChildPointerTruncated));
        }
        let child =
            u32::from_be_bytes([usable[pos], usable[pos + 1], usable[pos + 2], usable[pos + 3]]);
        pos += 4;
        Some(page_id_from_u32(child)?)
    } else {
        None
    };

    let payload_length =
        table::read_varint_at(usable, &mut pos, Corruption::CellPayloadLengthTruncated)?;
    let payload_length = usize::try_from(payload_length)
        .map_err(|_| table::Error::Corrupted(Corruption::PayloadIsTooLarge))?;
    if payload_length > MAX_PAYLOAD_BYTES {
        return Err(table::Error::PayloadTooLarge(payload_length));
    }

    let start = pos;

    let usable_size = page.usable_size();
    let x = usable_size
        .checked_sub(35)
        .ok_or(table::Error::Corrupted(Corruption::UsableSizeUnderflow))?;
    if payload_length <= x {
        let end = start + payload_length;
        if end > usable.len() {
            return Err(table::Error::Corrupted(Corruption::PayloadExtendsPastPageBoundary));
        }
        return Ok(IndexCellRef { child, payload: table::PayloadRef::Inline(&usable[start..end]) });
    }

    let local_len = table::local_payload_len(usable_size, payload_length)?;
    let end_local = start + local_len;
    if end_local > usable.len() {
        return Err(table::Error::Corrupted(Corruption::PayloadExtendsPastPageBoundary));
    }

    let overflow_end = end_local + 4;
    if overflow_end > usable.len() {
        return Err(table::Error::Corrupted(Corruption::OverflowPointerOutOfBounds));
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
    header: &BTreeHeader,
    child_slot: usize,
) -> Result<PageId> {
    if child_slot < header.cell_count as usize {
        let cell_ptrs = btree::cell_ptrs(page, header)?;
        let offset = btree::cell_ptr_at(cell_ptrs, child_slot)?;
        let cell = read_index_interior_cell(pager, page, offset)?;
        cell.child().ok_or(table::Error::Corrupted(Corruption::MissingChildPointer))
    } else {
        let right_most = header
            .right_most_child
            .ok_or(table::Error::Corrupted(Corruption::MissingRightMostChildPointer))?;
        page_id_from_u32(right_most)
    }
}
#[inline]
fn decode_index_rowid<'row>(
    payload: table::PayloadRef<'row>,
    scratch_bytes: &mut Vec<u8>,
) -> Result<i64> {
    let count = table::record_column_count(payload)?;
    let last = count.checked_sub(1).ok_or_else(|| table::Error::from(Error::MissingIndexRowId))?;
    let last = u16::try_from(last).map_err(|_| table::Error::from(Error::MissingIndexRowId))?;

    let Some(slot) = table::decode_record_column(payload, last, scratch_bytes)? else {
        return Err(Error::MissingIndexRowId.into());
    };

    match slot {
        table::ValueSlot::Integer(rowid) => Ok(rowid),
        _ => Err(Error::MissingIndexRowId.into()),
    }
}
