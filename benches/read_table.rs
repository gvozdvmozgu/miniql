use std::fs::File;
use std::path::{Path, PathBuf};

use codspeed_criterion_compat::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use miniql::pager::{PageId, Pager};
use miniql::query::{Scan, ScanScratch, col, lit_bytes, lit_i64};
use miniql::table::{self, RowScratch};

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn bench_read_sqlite_schema_ref_hot(c: &mut Criterion) {
    let db_path = fixture_path("users.db");
    let file = File::open(&db_path).expect("open fixture");
    let pager = Pager::new(file).expect("create pager");
    let mut scratch = RowScratch::with_capacity(8, 0);
    c.bench_function("read_sqlite_schema_ref_hot", |b| {
        b.iter(|| {
            let mut rows = 0usize;
            table::scan_table_ref_with_scratch(&pager, PageId::ROOT, &mut scratch, |_, _| {
                rows += 1;
                Ok(())
            })
            .expect("read sqlite_schema");
            black_box(rows);
        });
    });
}

fn bench_read_users_table_ref_hot(c: &mut Criterion) {
    let db_path = fixture_path("users.db");
    let file = File::open(&db_path).expect("open fixture");
    let pager = Pager::new(file).expect("create pager");
    let mut scratch = RowScratch::with_capacity(8, 0);
    c.bench_function("read_users_table_ref_hot", |b| {
        b.iter(|| {
            let mut rows = 0usize;
            table::scan_table_ref_with_scratch(&pager, PageId::new(2), &mut scratch, |_, _| {
                rows += 1;
                Ok(())
            })
            .expect("read users table");
            black_box(rows);
        });
    });
}

fn bench_read_sqlite_schema_ref_cold(c: &mut Criterion) {
    let db_path = fixture_path("users.db");
    c.bench_function("read_sqlite_schema_ref_cold", |b| {
        b.iter_batched(
            || {
                let file = File::open(&db_path).expect("open fixture");
                Pager::new(file).expect("create pager")
            },
            |pager| {
                let mut rows = 0usize;
                let mut scratch = RowScratch::with_capacity(8, 0);
                table::scan_table_ref_with_scratch(&pager, PageId::ROOT, &mut scratch, |_, _| {
                    rows += 1;
                    Ok(())
                })
                .expect("read sqlite_schema");
                black_box(rows);
            },
            BatchSize::PerIteration,
        );
    });
}

fn bench_read_users_table_ref_cold(c: &mut Criterion) {
    let db_path = fixture_path("users.db");
    c.bench_function("read_users_table_ref_cold", |b| {
        b.iter_batched(
            || {
                let file = File::open(&db_path).expect("open fixture");
                Pager::new(file).expect("create pager")
            },
            |pager| {
                let mut rows = 0usize;
                let mut scratch = RowScratch::with_capacity(8, 0);
                table::scan_table_ref_with_scratch(&pager, PageId::new(2), &mut scratch, |_, _| {
                    rows += 1;
                    Ok(())
                })
                .expect("read users table");
                black_box(rows);
            },
            BatchSize::PerIteration,
        );
    });
}

fn bench_read_overflow_table_ref_hot(c: &mut Criterion) {
    let db_path = fixture_path("overflow.db");
    let file = File::open(&db_path).expect("open fixture");
    let pager = Pager::new(file).expect("create pager");
    let mut scratch = RowScratch::with_capacity(4, 10008);
    c.bench_function("read_overflow_table_ref_hot", |b| {
        b.iter(|| {
            let mut rows = 0usize;
            table::scan_table_ref_with_scratch(&pager, PageId::new(2), &mut scratch, |_, _| {
                rows += 1;
                Ok(())
            })
            .expect("read overflow table");
            black_box(rows);
        });
    });
}

fn bench_query_scan_full_projection_hot(c: &mut Criterion) {
    let db_path = fixture_path("users.db");
    let file = File::open(&db_path).expect("open fixture");
    let pager = Pager::new(file).expect("create pager");
    let mut scratch = ScanScratch::with_capacity(8, 0);
    let mut scan =
        Scan::table(&pager, PageId::new(2)).project([0, 1, 2]).compile().expect("compile scan");
    c.bench_function("query_scan_full_projection_hot", |b| {
        b.iter(|| {
            let mut rows = 0usize;
            scan.for_each(&mut scratch, |_, _| {
                rows += 1;
                Ok(())
            })
            .expect("scan users table");
            black_box(rows);
        });
    });
}

fn bench_query_scan_filter_int_hot(c: &mut Criterion) {
    let db_path = fixture_path("users.db");
    let file = File::open(&db_path).expect("open fixture");
    let pager = Pager::new(file).expect("create pager");
    let mut scratch = ScanScratch::with_capacity(8, 0);
    let mut scan = Scan::table(&pager, PageId::new(2))
        .project([1, 2])
        .filter(col(2).gt(lit_i64(25)))
        .compile()
        .expect("compile scan");
    c.bench_function("query_scan_filter_int_hot", |b| {
        b.iter(|| {
            let mut rows = 0usize;
            scan.for_each(&mut scratch, |_, _| {
                rows += 1;
                Ok(())
            })
            .expect("scan users table");
            black_box(rows);
        });
    });
}

fn bench_query_scan_filter_text_hot(c: &mut Criterion) {
    let db_path = fixture_path("users.db");
    let file = File::open(&db_path).expect("open fixture");
    let pager = Pager::new(file).expect("create pager");
    let mut scratch = ScanScratch::with_capacity(8, 0);
    let mut scan = Scan::table(&pager, PageId::new(2))
        .project([1])
        .filter(col(1).eq(lit_bytes(b"alice")))
        .compile()
        .expect("compile scan");
    c.bench_function("query_scan_filter_text_hot", |b| {
        b.iter(|| {
            let mut rows = 0usize;
            scan.for_each(&mut scratch, |_, _| {
                rows += 1;
                Ok(())
            })
            .expect("scan users table");
            black_box(rows);
        });
    });
}

criterion_group!(
    benches,
    bench_read_sqlite_schema_ref_hot,
    bench_read_users_table_ref_hot,
    bench_read_sqlite_schema_ref_cold,
    bench_read_users_table_ref_cold,
    bench_read_overflow_table_ref_hot,
    bench_query_scan_full_projection_hot,
    bench_query_scan_filter_int_hot,
    bench_query_scan_filter_text_hot
);
criterion_main!(benches);
