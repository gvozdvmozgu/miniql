#![feature(allocator_api)]

use std::alloc::Global;

use miniql::pager::{PageId, Pager};

fn main() {
    let file = std::fs::File::open("app.db").expect("Failed to open file");

    let mut pager = Pager::new(file, Global).expect("Failed to create pager");
    pager.read_page(PageId::ROOT).expect("Failed to read root page");

    println!("Page count: {}", pager.count());
}
