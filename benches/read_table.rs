use std::fs::File;
use std::path::{Path, PathBuf};

use codspeed_criterion_compat::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use miniql::pager::{PageId, Pager};
use miniql::table;

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/users.db")
}

fn bench_read_sqlite_schema_ref_hot(c: &mut Criterion) {
    let db_path = fixture_path();
    let file = File::open(&db_path).expect("open fixture");
    let pager = Pager::new(file).expect("create pager");
    let mut scratch = Vec::with_capacity(8);
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
    let db_path = fixture_path();
    let file = File::open(&db_path).expect("open fixture");
    let pager = Pager::new(file).expect("create pager");
    let mut scratch = Vec::with_capacity(8);
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
    let db_path = fixture_path();
    c.bench_function("read_sqlite_schema_ref_cold", |b| {
        b.iter_batched(
            || {
                let file = File::open(&db_path).expect("open fixture");
                Pager::new(file).expect("create pager")
            },
            |pager| {
                let mut rows = 0usize;
                let mut scratch = Vec::with_capacity(8);
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
    let db_path = fixture_path();
    c.bench_function("read_users_table_ref_cold", |b| {
        b.iter_batched(
            || {
                let file = File::open(&db_path).expect("open fixture");
                Pager::new(file).expect("create pager")
            },
            |pager| {
                let mut rows = 0usize;
                let mut scratch = Vec::with_capacity(8);
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

criterion_group!(
    benches,
    bench_read_sqlite_schema_ref_hot,
    bench_read_users_table_ref_hot,
    bench_read_sqlite_schema_ref_cold,
    bench_read_users_table_ref_cold
);
criterion_main!(benches);
