use std::marker::PhantomData;
use std::ops::Range;

pub struct Decoder<'bytes> {
    start: *const u8,
    current: *const u8,
    end: *const u8,
    phantom: PhantomData<&'bytes [u8]>,
}

impl<'bytes> Decoder<'bytes> {
    #[inline]
    pub fn new(data: &'bytes [u8]) -> Self {
        let Range { start, end } = data.as_ptr_range();
        Self { start, current: start, end, phantom: PhantomData }
    }

    #[inline]
    pub fn read_u8(&mut self) -> u8 {
        self.read_array::<1>()[0]
    }

    #[inline]
    pub fn read_u16(&mut self) -> u16 {
        u16::from_be_bytes(self.read_array())
    }

    #[inline]
    pub fn read_u32(&mut self) -> u32 {
        u32::from_be_bytes(self.read_array())
    }

    #[inline]
    pub fn read_varint(&mut self) -> u64 {
        let mut result = 0u64;

        for _ in 0..8 {
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
    pub fn read_array<const N: usize>(&mut self) -> [u8; N] {
        self.read_bytes(N).try_into().unwrap()
    }

    #[inline]
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
    pub fn split_at(&self, position: usize) -> Decoder<'bytes> {
        assert!(position <= self.len());

        let current = unsafe { self.start.add(position) };
        Decoder { start: self.start, current, end: self.end, phantom: PhantomData }
    }

    #[allow(clippy::len_without_is_empty)]
    #[inline]
    pub fn len(&self) -> usize {
        unsafe { self.end.offset_from_unsigned(self.start) }
    }

    #[inline]
    pub fn remaining(&self) -> usize {
        unsafe { self.end.offset_from_unsigned(self.current) }
    }
}

#[cold]
#[inline(never)]
fn decoder_exhausted() -> ! {
    panic!("Decoder exhausted")
}
