use core::marker::PhantomData;
use core::ops::Range;

pub struct Decoder<'bytes> {
    start: *const u8,
    current: *const u8,
    end: *const u8,
    phantom: PhantomData<&'bytes [u8]>,
}

impl<'bytes> Decoder<'bytes> {
    pub fn new(data: &'bytes [u8]) -> Self {
        let Range { start, end } = data.as_ptr_range();
        Self { start, current: start, end, phantom: PhantomData }
    }

    pub fn read_u32(&mut self) -> u32 {
        u32::from_be_bytes(self.read_array())
    }

    #[inline]
    pub fn read_array<const N: usize>(&mut self) -> [u8; N] {
        self.read_bytes(N).try_into().unwrap()
    }

    #[inline]
    fn read_bytes(&mut self, bytes: usize) -> &'bytes [u8] {
        if bytes > self.remaining() {
            decoder_exhausted();
        }

        unsafe {
            let slice = core::slice::from_raw_parts(self.current, bytes);
            self.current = self.current.add(bytes);
            slice
        }
    }

    pub fn split_at(&self, position: usize) -> Decoder<'bytes> {
        assert!(position <= self.len());

        let current = unsafe { self.start.add(position) };
        Decoder { start: self.start, current, end: self.end, phantom: PhantomData }
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        unsafe { self.end.offset_from_unsigned(self.start) }
    }

    pub fn remaining(&self) -> usize {
        unsafe { self.end.offset_from_unsigned(self.current) }
    }
}

#[cold]
#[inline(never)]
fn decoder_exhausted() -> ! {
    panic!("Decoder exhausted")
}
