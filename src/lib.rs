#![cfg_attr(not(feature = "std"), no_std)]
#![feature(allocator_api)]
#![feature(once_cell_get_mut)]
#![feature(try_reserve_kind)]

extern crate alloc;

pub mod fs;
pub mod pager;
