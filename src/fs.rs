pub trait File: Sized {
    type Error;

    fn read(&mut self, buf: &mut [u8], offset: u64) -> Result<(), Self::Error>;
    fn write(&mut self, buf: &[u8], offset: u64) -> Result<(), Self::Error>;
    fn size(&self) -> Result<u64, Self::Error>;
}

#[cfg(feature = "std")]
impl File for std::fs::File {
    type Error = std::io::Error;

    fn read(&mut self, buf: &mut [u8], offset: u64) -> Result<(), Self::Error> {
        read_at(self, buf, offset).map(|_| ())
    }

    fn write(&mut self, buf: &[u8], offset: u64) -> Result<(), Self::Error> {
        write_at(self, buf, offset).map(|_| ())
    }

    fn size(&self) -> Result<u64, Self::Error> {
        self.metadata().map(|meta| meta.len())
    }
}

#[cfg(all(feature = "std", unix))]
fn read_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::unix::prelude::FileExt;
    file.read_at(buf, offset)
}

#[cfg(all(feature = "std", windows))]
fn read_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_read(buf, offset)
}

#[cfg(all(feature = "std", unix))]
fn write_at(file: &std::fs::File, buffer: &[u8], offset: u64) -> std::io::Result<usize> {
    use std::os::unix::prelude::FileExt;
    file.write_at(buffer, offset)
}

#[cfg(all(feature = "std", windows))]
fn write_at(file: &std::fs::File, buffer: &[u8], offset: u64) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_write(buffer, offset)
}
