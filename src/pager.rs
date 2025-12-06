use std::boxed::Box;
use std::cell::OnceCell;
use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::num::NonZero;
use std::vec::Vec;

use crate::decoder::Decoder;

pub const PAGE_SIZE: usize = 4096;

type Result<T> = std::result::Result<T, Error>;

pub struct Pager {
    file: File,
    pages: Vec<OnceCell<Page>>,
    db_size: u32,
}

impl Pager {
    pub fn new(file: File) -> Result<Self> {
        let file_size = file.metadata().map_err(Error::Io)?.len() as usize;
        let db_size = file_size.div_ceil(PAGE_SIZE) as u32;

        let mut pager = Pager { file, pages: Vec::new(), db_size };
        pager.read_page(PageId::ROOT)?;

        Ok(pager)
    }

    #[inline]
    pub fn root(&self) -> &Page {
        unsafe { self.pages.get_unchecked(0).get().unwrap_unchecked() }
    }

    pub fn count(&self) -> u32 {
        self.root().decoder().split_at(28).read_u32()
    }

    pub fn read_page(&mut self, page_id: PageId) -> Result<()> {
        let page_id1 = page_id.into_inner() as usize;
        let page_id0 = page_id1 - 1;

        // Ensure our vector holds at least `page_id1` entries.
        // After this call, `self.pages.len() ≥ page_id1`.
        self.resize_with(page_id1)?;

        // SAFETY: `resize_with(page_id1)` has ensured `self.pages.len() ≥ page_id1`.
        let page = unsafe { self.pages.get_unchecked_mut(page_id0) };

        if page.get().is_none() {
            let mut bytes = Box::new([0u8; PAGE_SIZE]);

            if self.db_size > page_id0 as u32 {
                let offset = page_id0.checked_mul(PAGE_SIZE).ok_or(Error::TooManyPages)?;
                self.file.seek(SeekFrom::Start(offset as u64)).map_err(Error::Io)?;
                self.file.read_exact(&mut bytes[..]).map_err(Error::Io)?;
            }

            let _ = page.set(Page::new(page_id1, bytes));
        }

        Ok(())
    }

    fn resize_with(&mut self, new_size: usize) -> Result<()> {
        self.pages
            .try_reserve(new_size.saturating_sub(self.pages.len()))
            .map_err(|_| Error::OutOfMemory)?;

        let len = self.pages.len();
        if new_size > len {
            self.pages.resize_with(new_size, OnceCell::new);
        }

        Ok(())
    }

    pub fn page(&self, page_id: PageId) -> Option<&Page> {
        let index = (page_id.into_inner() - 1) as usize;
        self.pages.get(index).and_then(OnceCell::get)
    }
}

enum PageKind {
    Root,
    Normal,
}

pub struct Page {
    bytes: Box<[u8]>,
    kind: PageKind,
}

impl Page {
    fn new(page_id: usize, bytes: Box<[u8; PAGE_SIZE]>) -> Self {
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

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[derive(Clone, Copy)]
pub struct PageId(NonZero<u32>);

impl PageId {
    pub const ROOT: PageId = unsafe { PageId::new_unchecked(1) };

    pub fn new(id: u32) -> Self {
        Self(NonZero::new(id).unwrap())
    }

    pub fn try_new(id: u32) -> Option<Self> {
        NonZero::new(id).map(Self)
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
pub enum Error {
    Io(std::io::Error),
    OutOfMemory,
    TooManyPages,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::OutOfMemory => f.write_str("Not enough memory to load page"),
            Self::TooManyPages => f.write_str("Database contains more pages than supported"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}
