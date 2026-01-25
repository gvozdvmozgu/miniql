use std::fmt;
use std::fs::File;
use std::num::NonZero;

use memmap2::Mmap;

use crate::decoder::Decoder;

type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct DbHeader {
    pub page_size: usize,
    pub reserved: u8,
    pub usable_size: usize,
    pub encoding: u32,
    pub page_count_hint: u32,
    pub write_version: u8,
    pub read_version: u8,
}

impl DbHeader {
    pub fn parse(header: &[u8]) -> Result<Self> {
        if header.len() < 100 {
            return Err(Error::FileTooSmall);
        }

        if &header[..16] != b"SQLite format 3\0" {
            return Err(Error::InvalidMagic);
        }

        let page_size_raw = u16::from_be_bytes([header[16], header[17]]);
        let page_size = match page_size_raw {
            1 => 65536usize,
            size => size as usize,
        };

        if !is_valid_page_size(page_size) {
            return Err(Error::UnsupportedPageSize(page_size_raw));
        }

        let reserved = header[20];
        let usable_size = page_size
            .checked_sub(reserved as usize)
            .ok_or(Error::UnsupportedReservedSpace(reserved))?;

        let payload_fraction = (header[21], header[22], header[23]);
        if payload_fraction != (64, 32, 32) {
            return Err(Error::UnsupportedPayloadFractions(payload_fraction));
        }

        let write_version = header[18];
        let read_version = header[19];
        if write_version == 2 || read_version == 2 {
            return Err(Error::WalModeUnsupported { write_version, read_version });
        }
        if write_version != 1 || read_version != 1 {
            return Err(Error::InvalidFileFormatVersion { write_version, read_version });
        }

        let page_count_hint = u32::from_be_bytes([header[28], header[29], header[30], header[31]]);
        let encoding_raw = u32::from_be_bytes([header[56], header[57], header[58], header[59]]);
        let encoding = match encoding_raw {
            0 | 1 => 1,
            other => return Err(Error::UnsupportedEncoding(other)),
        };

        Ok(DbHeader {
            page_size,
            reserved,
            usable_size,
            encoding,
            page_count_hint,
            write_version,
            read_version,
        })
    }
}

pub struct Pager {
    header: DbHeader,
    mmap: Mmap,
    page_count: u32,
}

impl Pager {
    pub fn new(file: File) -> Result<Self> {
        let mmap = unsafe { Mmap::map(&file) }.map_err(Error::Io)?;
        if mmap.len() < 100 {
            return Err(Error::FileTooSmall);
        }

        let header = DbHeader::parse(&mmap[..100])?;

        let file_len = mmap.len();
        if file_len < header.page_size {
            return Err(Error::FileTooSmall);
        }

        if file_len % header.page_size != 0 {
            return Err(Error::TruncatedFile);
        }

        let page_count =
            (file_len / header.page_size).try_into().map_err(|_| Error::TooManyPages)?;

        Ok(Pager { header, mmap, page_count })
    }

    #[inline]
    pub fn header(&self) -> &DbHeader {
        &self.header
    }

    pub fn count(&self) -> u32 {
        self.header.page_count_hint
    }

    pub fn page_count(&self) -> u32 {
        self.page_count
    }

    pub fn page_bytes(&self, page_id: PageId) -> Result<&[u8]> {
        let index = (page_id.into_inner() - 1) as usize;
        if index >= self.page_count as usize {
            return Err(Error::PageOutOfRange);
        }

        let start = index.checked_mul(self.header.page_size).ok_or(Error::PageOutOfRange)?;
        let end = start.checked_add(self.header.page_size).ok_or(Error::PageOutOfRange)?;
        if end > self.mmap.len() {
            return Err(Error::PageOutOfRange);
        }

        Ok(&self.mmap[start..end])
    }

    pub fn page(&self, page_id: PageId) -> Result<PageRef<'_>> {
        let bytes = self.page_bytes(page_id)?;
        Ok(PageRef { bytes, page_id, header: &self.header })
    }
}

pub struct PageRef<'a> {
    bytes: &'a [u8],
    page_id: PageId,
    header: &'a DbHeader,
}

impl<'a> PageRef<'a> {
    pub fn offset(&self) -> usize {
        if self.page_id.into_inner() == 1 { 100 } else { 0 }
    }

    pub fn usable_size(&self) -> usize {
        self.header.usable_size
    }

    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    pub fn usable_bytes(&self) -> &'a [u8] {
        let end = self.header.usable_size.min(self.bytes.len());
        &self.bytes[..end]
    }

    pub fn decoder(&self) -> Decoder<'a> {
        Decoder::new(self.bytes)
    }

    pub fn decoder_after_header(&self) -> Decoder<'a> {
        self.decoder().split_at(self.offset())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    FileTooSmall,
    InvalidMagic,
    UnsupportedPageSize(u16),
    UnsupportedReservedSpace(u8),
    UnsupportedPayloadFractions((u8, u8, u8)),
    UnsupportedEncoding(u32),
    WalModeUnsupported { write_version: u8, read_version: u8 },
    InvalidFileFormatVersion { write_version: u8, read_version: u8 },
    TruncatedFile,
    TooManyPages,
    PageOutOfRange,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::FileTooSmall => f.write_str("Database file is too small"),
            Self::InvalidMagic => f.write_str("Invalid SQLite header magic"),
            Self::UnsupportedPageSize(size) => {
                write!(f, "Unsupported page size: {size}")
            }
            Self::UnsupportedReservedSpace(reserved) => {
                write!(f, "Unsupported reserved space: {reserved}")
            }
            Self::UnsupportedPayloadFractions((a, b, c)) => {
                write!(f, "Unsupported payload fractions: {a}/{b}/{c}")
            }
            Self::UnsupportedEncoding(encoding) => {
                write!(f, "Unsupported encoding: {encoding}")
            }
            Self::WalModeUnsupported { write_version, read_version } => {
                write!(
                    f,
                    "WAL mode unsupported (write_version={write_version}, \
                     read_version={read_version})"
                )
            }
            Self::InvalidFileFormatVersion { write_version, read_version } => {
                write!(
                    f,
                    "Invalid SQLite format version (write_version={write_version}, \
                     read_version={read_version})"
                )
            }
            Self::TruncatedFile => f.write_str("Database file is truncated"),
            Self::TooManyPages => f.write_str("Database contains more pages than supported"),
            Self::PageOutOfRange => f.write_str("Requested page is out of range"),
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

fn is_valid_page_size(page_size: usize) -> bool {
    match page_size {
        512..=32768 => page_size.is_power_of_two(),
        65536 => true,
        _ => false,
    }
}
