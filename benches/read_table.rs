use std::fs::File;
use std::path::{Path, PathBuf};

use codspeed_criterion_compat::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use miniql::pager::{PageId, Pager};
use miniql::query::{Scan, ScanScratch, col, lit_bytes, lit_i64};
use miniql::table::{self, DecodeRecord, Null, RecordDecoder};

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

struct UserRow<'row> {
    _null: Null,
    name: &'row str,
    age: i64,
}

struct UserRowSpec;

impl DecodeRecord for UserRowSpec {
    type Row<'row> = UserRow<'row>;
    const COLS: usize = 3;

    fn decode<'row>(decoder: &mut RecordDecoder<'row>) -> table::Result<Self::Row<'row>> {
        Ok(UserRow {
            _null: decoder.read::<Null>()?,
            name: decoder.read_text()?,
            age: decoder.read_i64()?,
        })
    }
}

struct OverflowRow<'row> {
    id: i64,
    text: &'row str,
}

struct OverflowRowSpec;

impl DecodeRecord for OverflowRowSpec {
    type Row<'row> = OverflowRow<'row>;
    const COLS: usize = 2;

    fn decode<'row>(decoder: &mut RecordDecoder<'row>) -> table::Result<Self::Row<'row>> {
        Ok(OverflowRow { id: decoder.read_i64()?, text: decoder.read_text()? })
    }
}

fn bench_read_sqlite_schema_ref_hot(c: &mut Criterion) {
    let db_path = fixture_path("users.db");
    let file = File::open(&db_path).expect("open fixture");
    let pager = Pager::new(file).expect("create pager");
    c.bench_function("read_sqlite_schema_ref_hot", |b| {
        b.iter(|| {
            let mut rows = 0usize;
            table::scan_table(&pager, PageId::ROOT, |_, _| {
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
    c.bench_function("read_users_table_ref_hot", |b| {
        b.iter(|| {
            let mut rows = 0usize;
            table::scan_table(&pager, PageId::new(2), |_, _| {
                rows += 1;
                Ok(())
            })
            .expect("read users table");
            black_box(rows);
        });
    });
}

fn bench_read_users_table_rowview_decode_hot(c: &mut Criterion) {
    let db_path = fixture_path("users.db");
    let file = File::open(&db_path).expect("open fixture");
    let pager = Pager::new(file).expect("create pager");
    c.bench_function("read_users_table_rowview_decode_hot", |b| {
        b.iter(|| {
            let mut rows = 0usize;
            let mut age_sum = 0i64;
            let mut name_len = 0usize;
            table::scan_table(&pager, PageId::new(2), |_, row| {
                let null = row.get(0)?;
                black_box(null);
                let name = row.get_text(1)?;
                let age = row.get_i64(2)?;
                age_sum += age;
                name_len += name.len();
                rows += 1;
                Ok(())
            })
            .expect("read users table");
            black_box(rows);
            black_box(age_sum);
            black_box(name_len);
        });
    });
}

fn bench_read_users_table_typed_hot(c: &mut Criterion) {
    let db_path = fixture_path("users.db");
    let file = File::open(&db_path).expect("open fixture");
    let pager = Pager::new(file).expect("create pager");
    c.bench_function("read_users_table_typed_hot", |b| {
        b.iter(|| {
            let mut rows = 0usize;
            let mut age_sum = 0i64;
            let mut name_len = 0usize;
            table::scan_table_typed::<UserRowSpec, _>(&pager, PageId::new(2), |_, row| {
                black_box(row._null);
                age_sum += row.age;
                name_len += row.name.len();
                rows += 1;
                Ok(())
            })
            .expect("read users table typed");
            black_box(rows);
            black_box(age_sum);
            black_box(name_len);
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
                table::scan_table(&pager, PageId::ROOT, |_, _| {
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
                table::scan_table(&pager, PageId::new(2), |_, _| {
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
    c.bench_function("read_overflow_table_ref_hot", |b| {
        b.iter(|| {
            let mut rows = 0usize;
            table::scan_table_cells_with_scratch(&pager, PageId::new(2), |cell| {
                let payload = cell.payload().to_vec()?;
                let row = table::RowView::from_inline(&payload)?;
                let _ = row.get(0)?;
                rows += 1;
                Ok(())
            })
            .expect("read overflow table");
            black_box(rows);
        });
    });
}

fn bench_read_overflow_table_rowview_decode_hot(c: &mut Criterion) {
    let db_path = fixture_path("overflow.db");
    let file = File::open(&db_path).expect("open fixture");
    let pager = Pager::new(file).expect("create pager");
    c.bench_function("read_overflow_table_rowview_decode_hot", |b| {
        b.iter(|| {
            let mut rows = 0usize;
            let mut id_sum = 0i64;
            let mut text_len = 0usize;
            table::scan_table_cells_with_scratch(&pager, PageId::new(2), |cell| {
                let payload = cell.payload().to_vec()?;
                let row = table::RowView::from_inline(&payload)?;
                let id = row.get_i64(0)?;
                let text = row.get_text(1)?;
                id_sum += id;
                text_len += text.len();
                rows += 1;
                Ok(())
            })
            .expect("read overflow table");
            black_box(rows);
            black_box(id_sum);
            black_box(text_len);
        });
    });
}

fn bench_read_overflow_table_typed_hot(c: &mut Criterion) {
    let db_path = fixture_path("overflow.db");
    let file = File::open(&db_path).expect("open fixture");
    let pager = Pager::new(file).expect("create pager");
    c.bench_function("read_overflow_table_typed_hot", |b| {
        b.iter(|| {
            let mut rows = 0usize;
            let mut id_sum = 0i64;
            let mut text_len = 0usize;
            table::scan_table_typed::<OverflowRowSpec, _>(&pager, PageId::new(2), |_, row| {
                id_sum += row.id;
                text_len += row.text.len();
                rows += 1;
                Ok(())
            })
            .expect("read overflow table typed");
            black_box(rows);
            black_box(id_sum);
            black_box(text_len);
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
    bench_read_users_table_rowview_decode_hot,
    bench_read_users_table_typed_hot,
    bench_read_sqlite_schema_ref_cold,
    bench_read_users_table_ref_cold,
    bench_read_overflow_table_ref_hot,
    bench_read_overflow_table_rowview_decode_hot,
    bench_read_overflow_table_typed_hot,
    bench_query_scan_full_projection_hot,
    bench_query_scan_filter_int_hot,
    bench_query_scan_filter_text_hot
);
criterion_main!(benches);
