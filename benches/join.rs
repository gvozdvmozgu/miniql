use std::path::{Path, PathBuf};

use codspeed_criterion_compat::{Criterion, black_box, criterion_group, criterion_main};
use miniql::join::{Join, JoinKey, JoinScratch, JoinStrategy, JoinType};
use miniql::pager::{PageId, Pager};
use miniql::query::Scan;
use miniql::table::{self, ValueRef};

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn open_pager(name: &str) -> Pager {
    let file = std::fs::File::open(fixture_path(name)).expect("open fixture database");
    Pager::new(file).expect("create pager")
}

fn schema_root(pager: &Pager, kind: &str, name: &str) -> Option<u32> {
    let mut found = None;
    table::scan_table(pager, PageId::ROOT, |_, row| {
        let row_type = match row.get(0)? {
            Some(ValueRef::Text(bytes)) => std::str::from_utf8(bytes).ok(),
            _ => None,
        };
        if row_type != Some(kind) {
            return Ok(());
        }
        let row_name = match row.get(1)? {
            Some(ValueRef::Text(bytes)) => std::str::from_utf8(bytes).ok(),
            _ => None,
        };
        if row_name != Some(name) {
            return Ok(());
        }
        let rootpage = match row.get(3)? {
            Some(ValueRef::Integer(value)) => Some(value),
            _ => None,
        };
        found = rootpage.and_then(|value| u32::try_from(value).ok());
        Ok(())
    })
    .expect("read sqlite_schema");
    found
}

fn join_roots(pager: &Pager) -> (PageId, PageId, PageId) {
    let users_root = schema_root(pager, "table", "users").expect("users table root");
    let orders_root = schema_root(pager, "table", "orders").expect("orders table root");
    let index_root = schema_root(pager, "index", "orders_user_id_idx").expect("orders index root");
    (PageId::new(users_root), PageId::new(orders_root), PageId::new(index_root))
}

fn bench_join_inlj_hot(c: &mut Criterion) {
    let pager = open_pager("join.db");
    let (users_root, orders_root, index_root) = join_roots(&pager);
    let mut scratch = JoinScratch::with_capacity(4, 4, 0);
    let mut prepared = Join::new(
        JoinType::Inner,
        Scan::from_root(&pager, users_root),
        Scan::from_root(&pager, orders_root),
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
        Scan::from_root(&pager, users_root),
        Scan::from_root(&pager, orders_root),
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
