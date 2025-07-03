pub trait File: Sized {
    type Error;

    fn read(&mut self, buf: &mut [u8], offset: usize) -> Result<(), Self::Error>;
    fn write(&mut self, buf: &[u8], offset: usize) -> Result<(), Self::Error>;
    fn size(&self) -> Result<u64, Self::Error>;
}

#[cfg(feature = "std")]
impl File for std::fs::File {
    type Error = std::io::Error;

    fn read(&mut self, buf: &mut [u8], offset: usize) -> Result<(), Self::Error> {
        use std::io;

        io::Seek::seek(self, std::io::SeekFrom::Start(offset as u64))?;
        io::Read::read_exact(self, buf)
    }

    fn write(&mut self, buf: &[u8], offset: usize) -> Result<(), Self::Error> {
        use std::io;

        io::Seek::seek(self, std::io::SeekFrom::Start(offset as u64))?;
        io::Write::write_all(self, buf)
    }

    fn size(&self) -> Result<u64, Self::Error> {
        self.metadata().map(|meta| meta.len())
    }
}
