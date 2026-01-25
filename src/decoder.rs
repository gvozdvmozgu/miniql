use std::marker::PhantomData;
use std::ops::Range;

/// Lightweight byte decoder for SQLite structures.
pub struct Decoder<'bytes> {
    start: *const u8,
    current: *const u8,
    end: *const u8,
    phantom: PhantomData<&'bytes [u8]>,
}

impl<'bytes> Decoder<'bytes> {
    #[inline]
    /// Create a decoder over the provided bytes.
    pub fn new(data: &'bytes [u8]) -> Self {
        let Range { start, end } = data.as_ptr_range();
        Self { start, current: start, end, phantom: PhantomData }
    }

    #[inline]
    /// Read a single byte, panicking on EOF.
    pub fn read_u8(&mut self) -> u8 {
        if self.current == self.end {
            decoder_exhausted();
        }

        unsafe {
            let byte = *self.current;
            self.current = self.current.add(1);
            byte
        }
    }

    #[inline]
    /// Try to read a single byte, returning `None` on EOF.
    pub fn try_read_u8(&mut self) -> Option<u8> {
        if self.current == self.end {
            decoder_exhausted_cold();
            return None;
        }

        unsafe {
            let byte = *self.current;
            self.current = self.current.add(1);
            Some(byte)
        }
    }

    #[inline]
    /// Read a big-endian `u16`.
    pub fn read_u16(&mut self) -> u16 {
        u16::from_be_bytes(self.read_array())
    }

    #[inline]
    /// Try to read a big-endian `u16`.
    pub fn try_read_u16(&mut self) -> Option<u16> {
        self.try_read_array().map(u16::from_be_bytes)
    }

    #[inline]
    /// Read a big-endian `u32`.
    pub fn read_u32(&mut self) -> u32 {
        u32::from_be_bytes(self.read_array())
    }

    #[inline]
    /// Try to read a big-endian `u32`.
    pub fn try_read_u32(&mut self) -> Option<u32> {
        self.try_read_array().map(u32::from_be_bytes)
    }

    #[inline]
    /// Read a SQLite varint, panicking on EOF.
    pub fn read_varint(&mut self) -> u64 {
        let first = self.read_u8();
        if first & 0x80 == 0 {
            return u64::from(first);
        }

        let mut result = u64::from(first & 0x7F);
        for _ in 0..7 {
            let byte = self.read_u8();
            result = (result << 7) | u64::from(byte & 0x7F);

            if byte & 0x80 == 0 {
                return result;
            }
        }

        let byte = self.read_u8();
        (result << 8) | u64::from(byte)
    }

    #[inline]
    /// Try to read a SQLite varint, returning `None` on EOF.
    pub fn try_read_varint(&mut self) -> Option<u64> {
        let first = self.try_read_u8()?;
        if first & 0x80 == 0 {
            return Some(u64::from(first));
        }

        let mut result = u64::from(first & 0x7F);
        for _ in 0..7 {
            let byte = self.try_read_u8()?;
            result = (result << 7) | u64::from(byte & 0x7F);

            if byte & 0x80 == 0 {
                return Some(result);
            }
        }

        let byte = self.try_read_u8()?;
        Some((result << 8) | u64::from(byte))
    }

    #[inline]
    /// Read a fixed-size byte array.
    pub fn read_array<const N: usize>(&mut self) -> [u8; N] {
        self.read_bytes(N).try_into().unwrap()
    }

    #[inline]
    /// Try to read a fixed-size byte array.
    pub fn try_read_array<const N: usize>(&mut self) -> Option<[u8; N]> {
        let bytes = self.try_read_bytes(N)?;
        Some(bytes.try_into().unwrap())
    }

    #[inline]
    /// Read a byte slice of length `bytes`.
    pub fn read_bytes(&mut self, bytes: usize) -> &'bytes [u8] {
        if bytes > self.remaining() {
            decoder_exhausted();
        }

        unsafe {
            let slice = std::slice::from_raw_parts(self.current, bytes);
            self.current = self.current.add(bytes);
            slice
        }
    }

    #[inline]
    /// Try to read a byte slice of length `bytes`.
    pub fn try_read_bytes(&mut self, bytes: usize) -> Option<&'bytes [u8]> {
        if bytes > self.remaining() {
            decoder_exhausted_cold();
            return None;
        }

        unsafe {
            let slice = std::slice::from_raw_parts(self.current, bytes);
            self.current = self.current.add(bytes);
            Some(slice)
        }
    }

    #[inline]
    /// Split the decoder at a byte offset.
    pub fn split_at(&self, position: usize) -> Decoder<'bytes> {
        debug_assert!(position <= self.len());

        let current = unsafe { self.start.add(position) };
        Decoder { start: self.start, current, end: self.end, phantom: PhantomData }
    }

    #[allow(clippy::len_without_is_empty)]
    #[inline]
    /// Total length of the underlying byte slice.
    pub fn len(&self) -> usize {
        unsafe { self.end.offset_from_unsigned(self.start) }
    }

    #[inline]
    /// Remaining unread bytes.
    pub fn remaining(&self) -> usize {
        unsafe { self.end.offset_from_unsigned(self.current) }
    }
}

#[cold]
#[inline(never)]
fn decoder_exhausted() -> ! {
    panic!("Decoder exhausted")
}

#[cold]
#[inline(never)]
fn decoder_exhausted_cold() {}
