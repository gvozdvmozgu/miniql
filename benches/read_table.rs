use std::fs::File;
use std::path::{Path, PathBuf};

use codspeed_criterion_compat::{Criterion, black_box, criterion_group, criterion_main};
use miniql::pager::{PageId, Pager};
use miniql::table;

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/users.db")
}

fn bench_read_sqlite_schema(c: &mut Criterion) {
    let db_path = fixture_path();
    c.bench_function("read_sqlite_schema", |b| {
        b.iter(|| {
            let file = File::open(&db_path).expect("open fixture");
            let mut pager = Pager::new(file).expect("create pager");
            let rows = table::read_table(&mut pager, PageId::ROOT).expect("read sqlite_schema");
            black_box(&rows);
        });
    });
}

fn bench_read_users_table(c: &mut Criterion) {
    let db_path = fixture_path();
    c.bench_function("read_users_table", |b| {
        b.iter(|| {
            let file = File::open(&db_path).expect("open fixture");
            let mut pager = Pager::new(file).expect("create pager");
            let rows = table::read_table(&mut pager, PageId::new(2)).expect("read users table");
            black_box(&rows);
        });
    });
}

criterion_group!(benches, bench_read_sqlite_schema, bench_read_users_table);
criterion_main!(benches);
