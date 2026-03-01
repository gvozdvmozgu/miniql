mod util;

use miniql::join::{Join, JoinKey, JoinScratch, JoinStrategy, JoinType};
use miniql::pager::{PageId, Pager};
use miniql::query::Scan;
use miniql::table::{self, ValueRef};
use rusqlite::params;

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

fn join_roots(pager: &Pager, left: &str, right: &str, index: &str) -> (PageId, PageId, PageId) {
    let left_root = schema_root(pager, "table", left).expect("left table root");
    let right_root = schema_root(pager, "table", right).expect("right table root");
    let index_root = schema_root(pager, "index", index).expect("index root");
    (PageId::new(left_root), PageId::new(right_root), PageId::new(index_root))
}

fn join_count(
    pager: &Pager,
    left_root: PageId,
    right_root: PageId,
    index_root: PageId,
    strategy: JoinStrategy,
) -> usize {
    let mut scratch = JoinScratch::with_capacity(2, 2, 0);
    let mut rows = 0usize;
    Join::new(
        JoinType::Inner,
        Scan::from_root(pager, left_root),
        Scan::from_root(pager, right_root),
    )
    .on(JoinKey::Col(0), JoinKey::Col(0))
    .strategy(match strategy {
        JoinStrategy::IndexNestedLoop { .. } => {
            JoinStrategy::IndexNestedLoop { index_root, index_key_col: 0 }
        }
        other => other,
    })
    .for_each(&mut scratch, |_jr| {
        rows += 1;
        Ok(())
    })
    .expect("join");
    rows
}

fn join_count_on_cols(
    pager: &Pager,
    left_root: PageId,
    right_root: PageId,
    left_col: u16,
    right_col: u16,
    strategy: JoinStrategy,
) -> usize {
    let mut scratch = JoinScratch::with_capacity(2, 2, 0);
    let mut rows = 0usize;
    Join::new(
        JoinType::Inner,
        Scan::from_root(pager, left_root),
        Scan::from_root(pager, right_root),
    )
    .on(JoinKey::Col(left_col), JoinKey::Col(right_col))
    .strategy(strategy)
    .for_each(&mut scratch, |_jr| {
        rows += 1;
        Ok(())
    })
    .expect("join");
    rows
}

fn assert_all_strategies(
    pager: &Pager,
    left_root: PageId,
    right_root: PageId,
    index_root: PageId,
    expected: usize,
) {
    let hash = join_count(pager, left_root, right_root, index_root, JoinStrategy::Hash);
    let nested = join_count(pager, left_root, right_root, index_root, JoinStrategy::NestedLoopScan);
    let inlj = join_count(
        pager,
        left_root,
        right_root,
        index_root,
        JoinStrategy::IndexNestedLoop { index_root, index_key_col: 0 },
    );

    assert_eq!(hash, expected, "hash join mismatch");
    assert_eq!(nested, expected, "nested loop scan mismatch");
    assert_eq!(inlj, expected, "index nested loop mismatch");
}

#[test]
fn hash_join_numeric_collision_large_integers() {
    let a: i64 = 9_007_199_254_740_992;
    let b: i64 = a + 1;

    let file = util::make_db(|conn| {
        conn.execute_batch(
            "CREATE TABLE left_t (k INTEGER);
             CREATE TABLE right_t (k INTEGER);
             CREATE INDEX right_k_idx ON right_t(k);",
        )
        .unwrap();
        conn.execute("INSERT INTO left_t (k) VALUES (?1)", params![a]).unwrap();
        conn.execute("INSERT INTO right_t (k) VALUES (?1)", params![b]).unwrap();
    });

    let pager = Pager::new(std::fs::File::open(file.path()).unwrap()).unwrap();
    let (left_root, right_root, index_root) =
        join_roots(&pager, "left_t", "right_t", "right_k_idx");
    assert_all_strategies(&pager, left_root, right_root, index_root, 0);
}

#[test]
fn join_matches_negative_zero_and_zero() {
    let file = util::make_db(|conn| {
        conn.execute_batch(
            "CREATE TABLE left_t (k REAL);
             CREATE TABLE right_t (k REAL);
             CREATE INDEX right_k_idx ON right_t(k);",
        )
        .unwrap();
        conn.execute("INSERT INTO left_t (k) VALUES (?1)", params![-0.0_f64]).unwrap();
        conn.execute("INSERT INTO right_t (k) VALUES (?1)", params![0.0_f64]).unwrap();
    });

    let pager = Pager::new(std::fs::File::open(file.path()).unwrap()).unwrap();
    let (left_root, right_root, index_root) =
        join_roots(&pager, "left_t", "right_t", "right_k_idx");
    assert_all_strategies(&pager, left_root, right_root, index_root, 1);
}

#[test]
fn nan_never_matches() {
    let file = util::make_db(|conn| {
        conn.execute_batch(
            "CREATE TABLE left_t (k REAL);
             CREATE TABLE right_t (k REAL);
             CREATE INDEX right_k_idx ON right_t(k);",
        )
        .unwrap();
        conn.execute("INSERT INTO left_t (k) VALUES (?1)", params![f64::NAN]).unwrap();
        conn.execute("INSERT INTO right_t (k) VALUES (?1)", params![f64::NAN]).unwrap();
    });

    let pager = Pager::new(std::fs::File::open(file.path()).unwrap()).unwrap();
    let (left_root, right_root, index_root) =
        join_roots(&pager, "left_t", "right_t", "right_k_idx");
    assert_all_strategies(&pager, left_root, right_root, index_root, 0);
}

#[test]
fn auto_ignores_non_leading_index_for_join() {
    let file = util::make_db(|conn| {
        conn.execute_batch(
            "CREATE TABLE left_t (b INTEGER);
             CREATE TABLE right_t (a INTEGER, b INTEGER);
             CREATE INDEX right_ab_idx ON right_t(a, b);",
        )
        .unwrap();
        conn.execute("INSERT INTO left_t (b) VALUES (?1)", params![2_i64]).unwrap();
        conn.execute("INSERT INTO right_t (a, b) VALUES (?1, ?2)", params![1_i64, 2_i64]).unwrap();
        conn.execute("INSERT INTO right_t (a, b) VALUES (?1, ?2)", params![2_i64, 1_i64]).unwrap();
        conn.execute("INSERT INTO right_t (a, b) VALUES (?1, ?2)", params![3_i64, 2_i64]).unwrap();
    });

    let pager = Pager::new(std::fs::File::open(file.path()).unwrap()).unwrap();
    let (left_root, right_root, _index_root) =
        join_roots(&pager, "left_t", "right_t", "right_ab_idx");

    let auto = join_count_on_cols(&pager, left_root, right_root, 0, 1, JoinStrategy::Auto);
    let hash = join_count_on_cols(&pager, left_root, right_root, 0, 1, JoinStrategy::Hash);
    let nested =
        join_count_on_cols(&pager, left_root, right_root, 0, 1, JoinStrategy::NestedLoopScan);

    assert_eq!(hash, 2, "hash join expected 2 matches");
    assert_eq!(nested, 2, "nested loop expected 2 matches");
    assert_eq!(auto, 2, "auto join must not use non-leading index");
}
