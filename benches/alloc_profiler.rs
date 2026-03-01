#![allow(dead_code)]

use std::alloc::System;
use std::fs::OpenOptions;
use std::io::Write;

use stats_alloc::{INSTRUMENTED_SYSTEM, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

pub(crate) fn with_alloc_log<T>(benchmark_id: &str, f: impl FnOnce() -> T) -> T {
    let before = GLOBAL.stats();
    let out = f();
    let after = GLOBAL.stats();
    let delta = after - before;
    append_log(benchmark_id, delta);
    out
}

fn append_log(benchmark_id: &str, delta: stats_alloc::Stats) {
    let Some(path) = std::env::var_os("MINIQL_ALLOC_LOG") else {
        return;
    };

    let path = std::path::PathBuf::from(path);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let file = OpenOptions::new().create(true).append(true).open(path);
    let Ok(mut file) = file else {
        return;
    };

    let _ = writeln!(
        file,
        "{benchmark_id}\t{}\t{}\t{}\t{}\t{}\t{}",
        delta.allocations,
        delta.deallocations,
        delta.reallocations,
        delta.bytes_allocated,
        delta.bytes_deallocated,
        delta.bytes_reallocated
    );
}
