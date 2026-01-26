use std::path::{Path, PathBuf};

use codspeed_criterion_compat::{Criterion, black_box, criterion_group, criterion_main};
use miniql::join::{Join, JoinKey, JoinScratch, JoinStrategy, JoinType};
use miniql::pager::{PageId, Pager};
use miniql::query::Scan;
use miniql::table::{self, TableRow};

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn open_pager(name: &str) -> Pager {
    let file = std::fs::File::open(fixture_path(name)).expect("open fixture database");
    Pager::new(file).expect("create pager")
}

fn schema_root(rows: &[TableRow], kind: &str, name: &str) -> Option<u32> {
    rows.iter().find_map(|row| {
        if row.values.len() < 4 {
            return None;
        }

        let row_type = row.values[0].as_text()?;
        let row_name = row.values[1].as_text()?;
        let rootpage = row.values[3].as_integer()?;

        if row_type == kind && row_name == name { u32::try_from(rootpage).ok() } else { None }
    })
}

fn join_roots(pager: &Pager) -> (PageId, PageId, PageId) {
    let rows = table::read_table(pager, PageId::ROOT).expect("read sqlite_schema");
    let users_root = schema_root(&rows, "table", "users").expect("users table root");
    let orders_root = schema_root(&rows, "table", "orders").expect("orders table root");
    let index_root = schema_root(&rows, "index", "orders_user_id_idx").expect("orders index root");
    (PageId::new(users_root), PageId::new(orders_root), PageId::new(index_root))
}

fn bench_join_inlj_hot(c: &mut Criterion) {
    let pager = open_pager("join.db");
    let (users_root, orders_root, index_root) = join_roots(&pager);
    let mut scratch = JoinScratch::with_capacity(4, 4, 0);
    let mut prepared = Join::new(
        JoinType::Inner,
        Scan::table(&pager, users_root),
        Scan::table(&pager, orders_root),
    )
    .on(JoinKey::RowId, JoinKey::Col(0))
    .project_left([0])
    .project_right([1])
    .strategy(JoinStrategy::IndexNestedLoop { index_root, index_key_col: 0 })
    .compile()
    .expect("compile join");

    c.bench_function("join_inlj_hot", |b| {
        b.iter(|| {
            let mut rows = 0usize;
            prepared
                .for_each(&mut scratch, |_jr| {
                    rows += 1;
                    Ok(())
                })
                .expect("join");
            black_box(rows);
        });
    });
}

fn bench_join_hash_hot(c: &mut Criterion) {
    let pager = open_pager("join.db");
    let (users_root, orders_root, _index_root) = join_roots(&pager);
    let mut scratch = JoinScratch::with_capacity(4, 4, 0);
    let mut prepared = Join::new(
        JoinType::Inner,
        Scan::table(&pager, users_root),
        Scan::table(&pager, orders_root),
    )
    .on(JoinKey::RowId, JoinKey::Col(0))
    .project_left([0])
    .project_right([1])
    .strategy(JoinStrategy::Hash)
    .compile()
    .expect("compile join");

    c.bench_function("join_hash_hot", |b| {
        b.iter(|| {
            let mut rows = 0usize;
            prepared
                .for_each(&mut scratch, |_jr| {
                    rows += 1;
                    Ok(())
                })
                .expect("join");
            black_box(rows);
        });
    });
}

criterion_group!(benches, bench_join_inlj_hot, bench_join_hash_hot);
criterion_main!(benches);
