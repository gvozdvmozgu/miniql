use std::cmp::Ordering;

use crate::join::JoinError;
use crate::pager::{PageId, PageRef, Pager};
use crate::table::{self, ValueRef, ValueRefRaw};

pub type Result<T> = table::Result<T>;

const MAX_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Default)]
pub struct IndexScratch {
    stack: Vec<StackEntry>,
    overflow_buf: Vec<u8>,
    decoded: Vec<ValueRefRaw>,
}

impl IndexScratch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(values: usize, overflow: usize) -> Self {
        Self {
            stack: Vec::new(),
            overflow_buf: Vec::with_capacity(overflow),
            decoded: Vec::with_capacity(values),
        }
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

pub struct IndexCursor<'db, 'scratch> {
    pager: &'db Pager,
    root: PageId,
    key_col: u16,
    scratch: &'scratch mut IndexScratch,
    leaf: Option<LeafPos>,
    cached_cell: Option<(PageId, usize)>,
}

impl<'db, 'scratch> IndexCursor<'db, 'scratch> {
    pub fn new(
        pager: &'db Pager,
        root: PageId,
        key_col: u16,
        scratch: &'scratch mut IndexScratch,
    ) -> Self {
        Self { pager, root, key_col, scratch, leaf: None, cached_cell: None }
    }

    pub fn seek_ge(&mut self, target: ValueRef<'_>) -> Result<bool> {
        self.scratch.stack.clear();
        self.leaf = None;
        self.cached_cell = None;

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
                    self.cached_cell = None;

                    if idx < cell_count {
                        return Ok(true);
                    }

                    return self.advance_from_leaf_end();
                }
                IndexKind::Interior => {
                    let cell_count = header.cell_count as usize;
                    let mut lo = 0usize;
                    let mut hi = cell_count;

                    while lo < hi {
                        let mid = (lo + hi) / 2;
                        let offset =
                            u16::from_be_bytes([cell_ptrs[mid * 2], cell_ptrs[mid * 2 + 1]]);
                        let (_child, payload) = read_index_interior_cell(
                            self.pager,
                            &page,
                            offset,
                            &mut self.scratch.overflow_buf,
                        )?;
                        let key = decode_key_from_payload(self.key_col, payload)?;
                        let cmp = compare_total(target, key);

                        if cmp == Ordering::Greater {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }

                    if lo < cell_count {
                        let offset = u16::from_be_bytes([cell_ptrs[lo * 2], cell_ptrs[lo * 2 + 1]]);
                        let (child, _payload) = read_index_interior_cell(
                            self.pager,
                            &page,
                            offset,
                            &mut self.scratch.overflow_buf,
                        )?;
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

    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<bool> {
        let Some(leaf) = self.leaf else {
            return Ok(false);
        };

        let page = self.pager.page(leaf.page_id)?;
        let header = parse_header(&page)?;
        let cell_count = header.cell_count as usize;

        if leaf.cell_index + 1 < cell_count {
            self.leaf = Some(LeafPos { page_id: leaf.page_id, cell_index: leaf.cell_index + 1 });
            self.cached_cell = None;
            return Ok(true);
        }

        self.leaf = Some(LeafPos { page_id: leaf.page_id, cell_index: cell_count });
        self.cached_cell = None;
        self.advance_from_leaf_end()
    }

    pub fn key_eq(&mut self, target: ValueRef<'_>) -> Result<bool> {
        let Some(leaf) = self.leaf else {
            return Ok(false);
        };

        let page = self.pager.page(leaf.page_id)?;
        let header = parse_header(&page)?;
        if leaf.cell_index >= header.cell_count as usize {
            return Ok(false);
        }

        self.ensure_decoded_current(&page, &header)?;
        let idx = self.key_col as usize;
        let Some(value) = self.scratch.decoded.get(idx).copied() else {
            return Err(JoinError::IndexKeyNotComparable.into());
        };
        let key = unsafe { value.as_value_ref() };
        if matches!(key, ValueRef::Null) || matches!(target, ValueRef::Null) {
            return Ok(false);
        }

        Ok(compare_total(target, key) == Ordering::Equal)
    }

    pub fn current_rowid(&mut self) -> Result<i64> {
        let Some(leaf) = self.leaf else {
            return Err(table::Error::Corrupted("index cursor not positioned"));
        };

        let page = self.pager.page(leaf.page_id)?;
        let header = parse_header(&page)?;
        if leaf.cell_index >= header.cell_count as usize {
            return Err(table::Error::Corrupted("index cursor past end"));
        }

        self.ensure_decoded_current(&page, &header)?;
        let Some(value) = self.scratch.decoded.last().copied() else {
            return Err(JoinError::MissingIndexRowId.into());
        };

        match unsafe { value.as_value_ref() } {
            ValueRef::Integer(rowid) => Ok(rowid),
            _ => Err(JoinError::MissingIndexRowId.into()),
        }
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
            let offset = u16::from_be_bytes([cell_ptrs[mid * 2], cell_ptrs[mid * 2 + 1]]);
            let payload =
                read_index_leaf_payload(self.pager, page, offset, &mut self.scratch.overflow_buf)?;
            let key = decode_key_from_payload(self.key_col, payload)?;
            let cmp = compare_total(target, key);
            if cmp == Ordering::Greater {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        Ok(lo)
    }

    fn ensure_decoded_current(&mut self, page: &PageRef<'_>, header: &IndexHeader) -> Result<()> {
        let Some(leaf) = self.leaf else {
            return Err(table::Error::Corrupted("index cursor not positioned"));
        };

        if self.cached_cell == Some((leaf.page_id, leaf.cell_index)) {
            return Ok(());
        }

        let cell_ptrs = cell_ptrs(page, header)?;
        if leaf.cell_index >= header.cell_count as usize {
            return Err(table::Error::Corrupted("index cursor past end"));
        }
        let offset = u16::from_be_bytes([
            cell_ptrs[leaf.cell_index * 2],
            cell_ptrs[leaf.cell_index * 2 + 1],
        ]);
        let payload =
            read_index_leaf_payload(self.pager, page, offset, &mut self.scratch.overflow_buf)?;
        table::decode_record_project_into(payload, None, &mut self.scratch.decoded)?;
        self.cached_cell = Some((leaf.page_id, leaf.cell_index));
        Ok(())
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

                let child = child_page_for_slot(
                    self.pager,
                    &page,
                    &header,
                    next_slot,
                    &mut self.scratch.overflow_buf,
                )?;
                let leaf = self.descend_leftmost(child)?;
                self.leaf = Some(leaf);
                self.cached_cell = None;
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
            let offset = u16::from_be_bytes([cell_ptrs[0], cell_ptrs[1]]);
            let (child, _payload) = read_index_interior_cell(
                self.pager,
                &page,
                offset,
                &mut self.scratch.overflow_buf,
            )?;
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
    scratch.decoded.clear();
    scratch.overflow_buf.clear();

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
                let offset = u16::from_be_bytes([cell_ptrs[0], cell_ptrs[1]]);
                let payload =
                    read_index_leaf_payload(pager, &page, offset, &mut scratch.overflow_buf)?;
                let count = table::decode_record_project_into(payload, None, &mut scratch.decoded)?;
                if count == 0 {
                    return Err(table::Error::Corrupted("index record has no columns"));
                }
                return Ok(Some(count.saturating_sub(1)));
            }
            IndexKind::Interior => {
                let cell_ptrs = cell_ptrs(&page, &header)?;
                let offset = u16::from_be_bytes([cell_ptrs[0], cell_ptrs[1]]);
                let (child, _payload) =
                    read_index_interior_cell(pager, &page, offset, &mut scratch.overflow_buf)?;
                page_id = child;
            }
        }
    }
}

fn compare_total(left: ValueRef<'_>, right: ValueRef<'_>) -> Ordering {
    let rank = |value: ValueRef<'_>| match value {
        ValueRef::Null => 0u8,
        ValueRef::Integer(_) | ValueRef::Real(_) => 1u8,
        ValueRef::TextBytes(_) => 2u8,
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
        (ValueRef::TextBytes(l), ValueRef::TextBytes(r)) => l.cmp(r),
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

fn decode_key_from_payload<'row>(key_col: u16, payload: &'row [u8]) -> Result<ValueRef<'row>> {
    let Some(raw) = table::decode_record_column(payload, key_col)? else {
        return Err(JoinError::IndexKeyNotComparable.into());
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

fn read_index_leaf_payload<'row>(
    pager: &Pager,
    page: &'row PageRef<'_>,
    offset: u16,
    overflow_buf: &'row mut Vec<u8>,
) -> Result<&'row [u8]> {
    read_index_payload(pager, page, offset, overflow_buf, false).map(|(_, payload)| payload)
}

fn read_index_interior_cell<'row>(
    pager: &Pager,
    page: &'row PageRef<'_>,
    offset: u16,
    overflow_buf: &'row mut Vec<u8>,
) -> Result<(PageId, &'row [u8])> {
    read_index_payload(pager, page, offset, overflow_buf, true)
}

fn read_index_payload<'row>(
    pager: &Pager,
    page: &'row PageRef<'_>,
    offset: u16,
    overflow_buf: &'row mut Vec<u8>,
    has_child: bool,
) -> Result<(PageId, &'row [u8])> {
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
        PageId::try_new(child).ok_or(table::Error::Corrupted("child page id is zero"))?
    } else {
        PageId::ROOT
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
        return Ok((child, &usable[start..end]));
    }

    let local_len = table::local_payload_len(usable_size, payload_length)?;
    let end_local = start + local_len;
    if end_local > usable.len() {
        return Err(table::Error::Corrupted("payload extends past page boundary"));
    }

    overflow_buf.clear();

    let overflow_end = end_local + 4;
    if overflow_end > usable.len() {
        return Err(table::Error::Corrupted("overflow pointer out of bounds"));
    }
    let overflow_page = u32::from_be_bytes(usable[end_local..overflow_end].try_into().unwrap());
    if overflow_page == 0 {
        return Err(table::Error::OverflowChainTruncated);
    }
    table::assemble_overflow_payload(
        pager,
        payload_length,
        local_len,
        overflow_page,
        &usable[start..end_local],
        overflow_buf,
    )?;
    let payload = overflow_buf.as_slice();
    Ok((child, payload))
}

fn child_page_for_slot(
    pager: &Pager,
    page: &PageRef<'_>,
    header: &IndexHeader,
    child_slot: usize,
    overflow_buf: &mut Vec<u8>,
) -> Result<PageId> {
    if child_slot < header.cell_count as usize {
        let cell_ptrs = cell_ptrs(page, header)?;
        let offset = u16::from_be_bytes([cell_ptrs[child_slot * 2], cell_ptrs[child_slot * 2 + 1]]);
        let (child, _payload) = read_index_interior_cell(pager, page, offset, overflow_buf)?;
        Ok(child)
    } else {
        let right_most = header
            .right_most_child
            .ok_or(table::Error::Corrupted("missing right-most child pointer"))?;
        PageId::try_new(right_most).ok_or(table::Error::Corrupted("child page id is zero"))
    }
}
