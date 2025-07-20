#![feature(allocator_api)]

use std::alloc::Global;

use miniql::pager::{PageId, Pager};

fn main() {
    let file = std::env::args().nth(1).expect("Usage: miniql-cli <path-to-db-file>");
    let file = std::fs::File::open(file).expect("Failed to open file");

    let mut pager = Pager::new(file, Global).expect("Failed to create pager");
    pager.read_page(PageId::ROOT).expect("Failed to read root page");

    println!("Page count: {}", pager.count());
}
