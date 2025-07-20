use alloc::boxed::Box;
use alloc::vec::Vec;
use core::alloc::{AllocError, Allocator};
use core::cell::OnceCell;
use core::num::NonZero;

use crate::decoder::Decoder;
use crate::fs::File;

pub const PAGE_SIZE: usize = 4096;

type Result<T, F> = core::result::Result<T, Error<<F as File>::Error>>;

pub struct Pager<F: File, A: Allocator + Copy> {
    file: F,
    pages: Vec<OnceCell<Page<A>>, A>,
    db_size: u32,
    allocator: A,
}

impl<F: File, A: Allocator + Copy> Pager<F, A> {
    pub fn new(file: F, allocator: A) -> Result<Self, F> {
        let file_size = file.size().map_err(Error::Io)? as usize;
        let db_size = file_size.div_ceil(PAGE_SIZE) as u32;

        let mut pager = Pager { file, pages: Vec::new_in(allocator), db_size, allocator };
        pager.read_page(PageId::ROOT)?;

        Ok(pager)
    }

    pub fn root(&self) -> &Page<A> {
        unsafe { self.pages.get_unchecked(0).get().unwrap_unchecked() }
    }

    pub fn count(&self) -> u32 {
        self.root().decoder().split_at(28).read_u32()
    }

    pub fn read_page(&mut self, page_id: PageId) -> Result<(), F> {
        let page_id1 = page_id.into_inner() as usize;
        let page_id0 = page_id1 - 1;

        // Ensure our vector holds at least `page_id1` entries.
        // After this call, `self.pages.len() ≥ page_id1`.
        self.resize_with(page_id1)?;

        // SAFETY: `resize_with(page_id1)` has ensured `self.pages.len() ≥ page_id1`.
        let page = unsafe { self.pages.get_unchecked_mut(page_id0) }
            .get_mut_or_try_init(|| {
                Box::try_new_in([0; PAGE_SIZE], self.allocator)
                    .map(|page| Page::new(page_id1, page))
            })
            .map_err(|AllocError| Error::OutOfMemory)?;

        if self.db_size > page_id0 as u32 {
            let offset = page_id0.checked_mul(PAGE_SIZE).ok_or(Error::TooManyPages)?;
            self.file.read(&mut page.bytes, offset).map_err(Error::Io)?;
        }

        Ok(())
    }

    fn resize_with(&mut self, new_size: usize) -> Result<(), F> {
        self.pages.try_reserve(new_size).map_err(|error| match error.kind() {
            alloc::collections::TryReserveErrorKind::CapacityOverflow => Error::CapacityOverflow,
            alloc::collections::TryReserveErrorKind::AllocError { .. } => Error::OutOfMemory,
        })?;

        let len = self.pages.len();
        if new_size > len {
            let additional = new_size - len;

            unsafe {
                let ptr = self.pages.as_mut_ptr().add(len);

                for i in 0..additional {
                    ptr.add(i).write(OnceCell::new());
                }

                self.pages.set_len(new_size);
            }
        }

        Ok(())
    }
}

enum PageKind {
    Root,
    Normal,
}

pub struct Page<A: Allocator> {
    bytes: Box<[u8], A>,
    kind: PageKind,
}

impl<A: Allocator> Page<A> {
    fn new(page_id: usize, bytes: Box<[u8; PAGE_SIZE], A>) -> Self {
        Self { bytes, kind: if page_id == 1 { PageKind::Root } else { PageKind::Normal } }
    }

    pub fn offset(&self) -> usize {
        match self.kind {
            PageKind::Root => 100,
            PageKind::Normal => 0,
        }
    }

    pub fn decoder(&self) -> Decoder<'_> {
        Decoder::new(&self.bytes)
    }

    pub fn decoder_after_header(&self) -> Decoder<'_> {
        self.decoder().split_at(self.offset())
    }
}

#[derive(Clone, Copy)]
pub struct PageId(NonZero<u32>);

impl PageId {
    pub const ROOT: PageId = unsafe { PageId::new_unchecked(1) };

    pub fn new(id: u32) -> Self {
        Self(NonZero::new(id).unwrap())
    }

    /// # Safety
    ///
    /// The value must not be zero.
    pub const unsafe fn new_unchecked(id: u32) -> Self {
        unsafe { Self(NonZero::new_unchecked(id)) }
    }

    pub fn into_inner(self) -> u32 {
        self.0.get()
    }
}

#[derive(Debug)]
pub enum Error<Io> {
    Io(Io),
    OutOfMemory,
    TooManyPages,
    CapacityOverflow,
}
