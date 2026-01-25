mod util;

use miniql::join::{
    Error as JoinError, Join, JoinKey, JoinOrderBy, JoinScratch, JoinStrategy, JoinType,
};
use miniql::pager::{PageId, Pager};
use miniql::query::{col, lit_f64, lit_i64};
use miniql::table::{self, TableRow, ValueRef};
use miniql::{Db, Error as TableError, OrderDir};
use rusqlite::params;

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

fn join_roots(
    pager: &Pager,
    left: &str,
    right: &str,
    index: Option<&str>,
) -> (PageId, PageId, Option<PageId>) {
    let rows = table::read_table(pager, PageId::ROOT).expect("read sqlite_schema");
    let left_root = schema_root(&rows, "table", left).expect("left table root");
    let right_root = schema_root(&rows, "table", right).expect("right table root");
    let index_root = index.and_then(|name| schema_root(&rows, "index", name)).map(PageId::new);
    (PageId::new(left_root), PageId::new(right_root), index_root)
}

fn new_scratch() -> JoinScratch {
    JoinScratch::with_capacity(4, 4, 4096)
}

fn make_int_join_db(
    left_keys: &[Option<i64>],
    right_keys: &[Option<i64>],
    with_index: bool,
) -> tempfile::NamedTempFile {
    util::make_db(|conn| {
        conn.execute_batch(
            "CREATE TABLE left_t (k INTEGER, v INTEGER);
             CREATE TABLE right_t (k INTEGER, v INTEGER);",
        )
        .unwrap();
        if with_index {
            conn.execute_batch("CREATE INDEX right_k_idx ON right_t(k);").unwrap();
        }
        for (idx, key) in left_keys.iter().enumerate() {
            conn.execute("INSERT INTO left_t (k, v) VALUES (?1, ?2)", params![key, idx as i64])
                .unwrap();
        }
        for (idx, key) in right_keys.iter().enumerate() {
            conn.execute("INSERT INTO right_t (k, v) VALUES (?1, ?2)", params![key, idx as i64])
                .unwrap();
        }
    })
}

fn collect_pairs(
    db_path: &std::path::Path,
    strategy: JoinStrategy,
    index_root: Option<PageId>,
    join_type: JoinType,
) -> Vec<(i64, i64)> {
    let db = Db::open(db_path).expect("open db");
    let pager = Pager::new(std::fs::File::open(db_path).unwrap()).unwrap();
    let (left_root, right_root, _) = join_roots(&pager, "left_t", "right_t", None);
    let left = db.table_root(left_root);
    let right = db.table_root(right_root);

    let strategy = match (strategy, index_root) {
        (JoinStrategy::IndexNestedLoop { .. }, Some(index_root)) => {
            JoinStrategy::IndexNestedLoop { index_root, index_key_col: 0 }
        }
        (JoinStrategy::IndexNestedLoop { .. }, None) => {
            panic!("index root required for IndexNestedLoop strategy");
        }
        (other, _) => other,
    };

    let mut out = Vec::new();
    let mut scratch = new_scratch();
    Join::new(join_type, left.scan(), right.scan())
        .on(JoinKey::Col(0), JoinKey::Col(0))
        .strategy(strategy)
        .for_each(&mut scratch, |jr| {
            out.push((jr.left_rowid, jr.right_rowid));
            Ok(())
        })
        .expect("join");
    out.sort();
    out
}

fn collect_left_rows(
    db_path: &std::path::Path,
    strategy: JoinStrategy,
    index_root: Option<PageId>,
) -> Vec<(i64, i64, bool)> {
    let db = Db::open(db_path).expect("open db");
    let left = db.table("left_t").expect("left table");
    let right = db.table("right_t").expect("right table");

    let strategy = match (strategy, index_root) {
        (JoinStrategy::IndexNestedLoop { .. }, Some(index_root)) => {
            JoinStrategy::IndexNestedLoop { index_root, index_key_col: 0 }
        }
        (JoinStrategy::IndexNestedLoop { .. }, None) => {
            panic!("index root required for IndexNestedLoop strategy");
        }
        (other, _) => other,
    };

    let mut out = Vec::new();
    let mut scratch = new_scratch();
    Join::left(left.scan(), right.scan())
        .on(JoinKey::Col(0), JoinKey::Col(0))
        .strategy(strategy)
        .for_each(&mut scratch, |jr| {
            let right_all_null =
                (0..jr.right.len()).all(|idx| matches!(jr.right.get(idx), Some(ValueRef::Null)));
            out.push((jr.left_rowid, jr.right_rowid, right_all_null));
            Ok(())
        })
        .expect("join");
    out.sort();
    out
}

#[test]
fn join_compile_missing_on_errors() {
    let file = make_int_join_db(&[Some(1)], &[Some(1)], false);
    let db = Db::open(file.path()).unwrap();
    let left = db.table("left_t").unwrap();
    let right = db.table("right_t").unwrap();
    let err = Join::new(JoinType::Inner, left.scan(), right.scan()).compile();
    assert!(matches!(err, Err(TableError::Join(JoinError::MissingJoinCondition))));
}

#[test]
fn join_compile_index_nested_loop_requires_right_col_key() {
    let file = make_int_join_db(&[Some(1)], &[Some(1)], true);
    let pager = Pager::new(std::fs::File::open(file.path()).unwrap()).unwrap();
    let (left_root, right_root, index_root) =
        join_roots(&pager, "left_t", "right_t", Some("right_k_idx"));
    let index_root = index_root.expect("index root");
    let db = Db::open(file.path()).unwrap();
    let left = db.table_root(left_root);
    let right = db.table_root(right_root);
    let err = Join::new(JoinType::Inner, left.scan(), right.scan())
        .on(JoinKey::Col(0), JoinKey::RowId)
        .strategy(JoinStrategy::IndexNestedLoop { index_root, index_key_col: 0 })
        .compile();
    assert!(matches!(err, Err(TableError::Join(JoinError::UnsupportedJoinKeyType))));
}

#[test]
fn join_order_by_left_right_columns() {
    let file = util::make_db(|conn| {
        conn.execute_batch(
            "CREATE TABLE left_t (k INTEGER, name TEXT);
             CREATE TABLE right_t (k INTEGER, score INTEGER, tag TEXT);",
        )
        .unwrap();
        conn.execute("INSERT INTO left_t (k, name) VALUES (?1, ?2)", params![1, "bob"]).unwrap();
        conn.execute("INSERT INTO left_t (k, name) VALUES (?1, ?2)", params![2, "alice"]).unwrap();
        conn.execute("INSERT INTO right_t (k, score, tag) VALUES (?1, ?2, ?3)", params![1, 5, "x"])
            .unwrap();
        conn.execute("INSERT INTO right_t (k, score, tag) VALUES (?1, ?2, ?3)", params![1, 3, "y"])
            .unwrap();
        conn.execute("INSERT INTO right_t (k, score, tag) VALUES (?1, ?2, ?3)", params![2, 4, "z"])
            .unwrap();
    });

    let db = Db::open(file.path()).expect("open db");
    let left = db.table("left_t").expect("left table");
    let right = db.table("right_t").expect("right table");
    let mut scratch = new_scratch();
    let mut seen = Vec::new();

    Join::inner(left.scan(), right.scan())
        .on(JoinKey::Col(0), JoinKey::Col(0))
        .project_left([1])
        .project_right([2])
        .order_by([JoinOrderBy::left(0, OrderDir::Asc), JoinOrderBy::right(1, OrderDir::Desc)])
        .for_each(&mut scratch, |jr| {
            let name = jr.left.get_text(0)?.to_owned();
            let tag = jr.right.get_text(0)?.to_owned();
            seen.push((name, tag));
            Ok(())
        })
        .expect("join");

    assert_eq!(
        seen,
        vec![
            ("bob".to_string(), "x".to_string()),
            ("bob".to_string(), "y".to_string()),
            ("alice".to_string(), "z".to_string())
        ]
    );
}

#[test]
fn inner_join_one_to_one() {
    let file = make_int_join_db(&[Some(1), Some(2), Some(3)], &[Some(1), Some(2), Some(3)], true);
    let pager = Pager::new(std::fs::File::open(file.path()).unwrap()).unwrap();
    let (_, _, index_root) = join_roots(&pager, "left_t", "right_t", Some("right_k_idx"));
    let index_root = index_root.expect("index root");

    let expected = vec![(1, 1), (2, 2), (3, 3)];
    let hash = collect_pairs(file.path(), JoinStrategy::Hash, None, JoinType::Inner);
    let nested = collect_pairs(file.path(), JoinStrategy::NestedLoopScan, None, JoinType::Inner);
    let inlj = collect_pairs(
        file.path(),
        JoinStrategy::IndexNestedLoop { index_root, index_key_col: 0 },
        Some(index_root),
        JoinType::Inner,
    );
    let auto = collect_pairs(file.path(), JoinStrategy::Auto, Some(index_root), JoinType::Inner);
    assert_eq!(hash, expected);
    assert_eq!(nested, expected);
    assert_eq!(inlj, expected);
    assert_eq!(auto, expected);
}

#[test]
fn inner_join_one_to_many() {
    let file = make_int_join_db(&[Some(1)], &[Some(1), Some(1), Some(1)], true);
    let pager = Pager::new(std::fs::File::open(file.path()).unwrap()).unwrap();
    let (_, _, index_root) = join_roots(&pager, "left_t", "right_t", Some("right_k_idx"));
    let index_root = index_root.expect("index root");

    let expected = vec![(1, 1), (1, 2), (1, 3)];
    let hash = collect_pairs(file.path(), JoinStrategy::Hash, None, JoinType::Inner);
    let nested = collect_pairs(file.path(), JoinStrategy::NestedLoopScan, None, JoinType::Inner);
    let inlj = collect_pairs(
        file.path(),
        JoinStrategy::IndexNestedLoop { index_root, index_key_col: 0 },
        Some(index_root),
        JoinType::Inner,
    );
    let auto = collect_pairs(file.path(), JoinStrategy::Auto, Some(index_root), JoinType::Inner);
    assert_eq!(hash, expected);
    assert_eq!(nested, expected);
    assert_eq!(inlj, expected);
    assert_eq!(auto, expected);
}

#[test]
fn inner_join_many_to_one() {
    let file = make_int_join_db(&[Some(1), Some(1), Some(1)], &[Some(1)], true);
    let pager = Pager::new(std::fs::File::open(file.path()).unwrap()).unwrap();
    let (_, _, index_root) = join_roots(&pager, "left_t", "right_t", Some("right_k_idx"));
    let index_root = index_root.expect("index root");

    let expected = vec![(1, 1), (2, 1), (3, 1)];
    let hash = collect_pairs(file.path(), JoinStrategy::Hash, None, JoinType::Inner);
    let nested = collect_pairs(file.path(), JoinStrategy::NestedLoopScan, None, JoinType::Inner);
    let inlj = collect_pairs(
        file.path(),
        JoinStrategy::IndexNestedLoop { index_root, index_key_col: 0 },
        Some(index_root),
        JoinType::Inner,
    );
    let auto = collect_pairs(file.path(), JoinStrategy::Auto, Some(index_root), JoinType::Inner);
    assert_eq!(hash, expected);
    assert_eq!(nested, expected);
    assert_eq!(inlj, expected);
    assert_eq!(auto, expected);
}

#[test]
fn inner_join_many_to_many() {
    let file = make_int_join_db(&[Some(1), Some(1)], &[Some(1), Some(1), Some(1)], true);
    let pager = Pager::new(std::fs::File::open(file.path()).unwrap()).unwrap();
    let (_, _, index_root) = join_roots(&pager, "left_t", "right_t", Some("right_k_idx"));
    let index_root = index_root.expect("index root");

    let expected = vec![(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)];
    let hash = collect_pairs(file.path(), JoinStrategy::Hash, None, JoinType::Inner);
    let nested = collect_pairs(file.path(), JoinStrategy::NestedLoopScan, None, JoinType::Inner);
    let inlj = collect_pairs(
        file.path(),
        JoinStrategy::IndexNestedLoop { index_root, index_key_col: 0 },
        Some(index_root),
        JoinType::Inner,
    );
    let auto = collect_pairs(file.path(), JoinStrategy::Auto, Some(index_root), JoinType::Inner);
    assert_eq!(hash, expected);
    assert_eq!(nested, expected);
    assert_eq!(inlj, expected);
    assert_eq!(auto, expected);
}

#[test]
fn null_join_keys_never_match() {
    let file = make_int_join_db(&[None, Some(1)], &[None, Some(1)], true);
    let pager = Pager::new(std::fs::File::open(file.path()).unwrap()).unwrap();
    let (_, _, index_root) = join_roots(&pager, "left_t", "right_t", Some("right_k_idx"));
    let index_root = index_root.expect("index root");

    let expected = vec![(2, 2)];
    let hash = collect_pairs(file.path(), JoinStrategy::Hash, None, JoinType::Inner);
    let nested = collect_pairs(file.path(), JoinStrategy::NestedLoopScan, None, JoinType::Inner);
    let inlj = collect_pairs(
        file.path(),
        JoinStrategy::IndexNestedLoop { index_root, index_key_col: 0 },
        Some(index_root),
        JoinType::Inner,
    );
    let auto = collect_pairs(file.path(), JoinStrategy::Auto, Some(index_root), JoinType::Inner);
    assert_eq!(hash, expected);
    assert_eq!(nested, expected);
    assert_eq!(inlj, expected);
    assert_eq!(auto, expected);
}

#[test]
fn left_join_emits_nulls_for_unmatched() {
    let file = make_int_join_db(&[Some(1), Some(2), Some(3)], &[Some(1)], true);
    let db = Db::open(file.path()).unwrap();
    let left = db.table("left_t").unwrap();
    let right = db.table("right_t").unwrap();
    let mut scratch = new_scratch();
    let mut seen = Vec::new();

    Join::left(left.scan(), right.scan())
        .on(JoinKey::Col(0), JoinKey::Col(0))
        .strategy(JoinStrategy::Hash)
        .for_each(&mut scratch, |jr| {
            let right_is_null = matches!(jr.right.get(0), Some(ValueRef::Null));
            seen.push((jr.left_rowid, jr.right_rowid, right_is_null));
            Ok(())
        })
        .expect("left join");

    assert_eq!(seen, vec![(1, 1, false), (2, 0, true), (3, 0, true)]);
}

#[test]
fn left_join_null_left_key_emits_nulls() {
    let file = make_int_join_db(&[None, Some(1)], &[Some(1)], true);
    let db = Db::open(file.path()).unwrap();
    let left = db.table("left_t").unwrap();
    let right = db.table("right_t").unwrap();
    let mut scratch = new_scratch();
    let mut seen = Vec::new();

    Join::left(left.scan(), right.scan())
        .on(JoinKey::Col(0), JoinKey::Col(0))
        .strategy(JoinStrategy::Hash)
        .for_each(&mut scratch, |jr| {
            let right_is_null = matches!(jr.right.get(0), Some(ValueRef::Null));
            seen.push((jr.left_rowid, jr.right_rowid, right_is_null));
            Ok(())
        })
        .expect("left join");

    assert_eq!(seen, vec![(1, 0, true), (2, 1, false)]);
}

#[test]
fn left_join_no_matches_emits_nulls_for_every_row() {
    let file = make_int_join_db(&[Some(1), Some(2), Some(3)], &[Some(4)], true);
    let pager = Pager::new(std::fs::File::open(file.path()).unwrap()).unwrap();
    let (_, _, index_root) = join_roots(&pager, "left_t", "right_t", Some("right_k_idx"));
    let index_root = index_root.expect("index root");

    let expected = vec![(1, 0, true), (2, 0, true), (3, 0, true)];
    let hash = collect_left_rows(file.path(), JoinStrategy::Hash, None);
    let nested = collect_left_rows(file.path(), JoinStrategy::NestedLoopScan, None);
    let inlj = collect_left_rows(
        file.path(),
        JoinStrategy::IndexNestedLoop { index_root, index_key_col: 0 },
        Some(index_root),
    );
    let auto = collect_left_rows(file.path(), JoinStrategy::Auto, Some(index_root));
    assert_eq!(hash, expected);
    assert_eq!(nested, expected);
    assert_eq!(inlj, expected);
    assert_eq!(auto, expected);
}

#[test]
fn join_key_not_in_projection_still_works() {
    let file = util::make_db(|conn| {
        conn.execute_batch(
            "CREATE TABLE left_t (k INTEGER, a INTEGER, b INTEGER);
             CREATE TABLE right_t (k INTEGER, x INTEGER);",
        )
        .unwrap();
        conn.execute("INSERT INTO left_t (k, a, b) VALUES (1, 10, 20)", []).unwrap();
        conn.execute("INSERT INTO right_t (k, x) VALUES (1, 99)", []).unwrap();
    });

    let db = Db::open(file.path()).unwrap();
    let left = db.table("left_t").unwrap();
    let right = db.table("right_t").unwrap();
    let mut scratch = new_scratch();
    let mut seen = Vec::new();

    Join::inner(left.scan().project([1]), right.scan().project([1]))
        .on(JoinKey::Col(0), JoinKey::Col(0))
        .strategy(JoinStrategy::Hash)
        .for_each(&mut scratch, |jr| {
            assert_eq!(jr.left.len(), 1);
            assert_eq!(jr.right.len(), 1);
            seen.push((jr.left.get_i64(0)?, jr.right.get_i64(0)?));
            Ok(())
        })
        .expect("join");

    assert_eq!(seen, vec![(10, 99)]);
}

#[test]
fn projection_reorder_and_duplicates() {
    let file = util::make_db(|conn| {
        conn.execute_batch(
            "CREATE TABLE left_t (k INTEGER, a INTEGER, b INTEGER);
             CREATE TABLE right_t (k INTEGER, x INTEGER);",
        )
        .unwrap();
        conn.execute("INSERT INTO left_t (k, a, b) VALUES (1, 10, 20)", []).unwrap();
        conn.execute("INSERT INTO right_t (k, x) VALUES (1, 99)", []).unwrap();
    });

    let db = Db::open(file.path()).unwrap();
    let left = db.table("left_t").unwrap();
    let right = db.table("right_t").unwrap();
    let mut scratch = new_scratch();

    Join::inner(left.scan().project([2, 0]), right.scan().project([1]))
        .on(JoinKey::Col(0), JoinKey::Col(0))
        .strategy(JoinStrategy::Hash)
        .for_each(&mut scratch, |jr| {
            assert_eq!(jr.left.get_i64(0)?, 20);
            assert_eq!(jr.left.get_i64(1)?, 1);
            assert_eq!(jr.right.get_i64(0)?, 99);
            Ok(())
        })
        .expect("join");

    Join::inner(left.scan().project([1, 1]), right.scan().project([1]))
        .on(JoinKey::Col(0), JoinKey::Col(0))
        .strategy(JoinStrategy::Hash)
        .for_each(&mut scratch, |jr| {
            assert_eq!(jr.left.len(), 2);
            assert_eq!(jr.left.get_i64(0)?, 10);
            assert_eq!(jr.left.get_i64(1)?, 10);
            Ok(())
        })
        .expect("join");
}

#[test]
fn filter_on_left_and_right() {
    let file = util::make_db(|conn| {
        conn.execute_batch(
            "CREATE TABLE left_t (k INTEGER, flag INTEGER);
             CREATE TABLE right_t (k INTEGER, flag INTEGER);
             CREATE INDEX right_k_idx ON right_t(k);",
        )
        .unwrap();
        conn.execute("INSERT INTO left_t (k, flag) VALUES (1, 1)", []).unwrap();
        conn.execute("INSERT INTO left_t (k, flag) VALUES (2, 0)", []).unwrap();
        conn.execute("INSERT INTO right_t (k, flag) VALUES (1, 1)", []).unwrap();
        conn.execute("INSERT INTO right_t (k, flag) VALUES (2, 1)", []).unwrap();
    });

    let db = Db::open(file.path()).unwrap();
    let left = db.table("left_t").unwrap();
    let right = db.table("right_t").unwrap();
    let mut scratch = new_scratch();
    let mut seen = Vec::new();

    Join::inner(
        left.scan().filter(col(1).eq(lit_i64(1))),
        right.scan().filter(col(1).eq(lit_i64(1))),
    )
    .on(JoinKey::Col(0), JoinKey::Col(0))
    .strategy(JoinStrategy::Hash)
    .for_each(&mut scratch, |jr| {
        seen.push((jr.left_rowid, jr.right_rowid));
        Ok(())
    })
    .expect("join");

    assert_eq!(seen, vec![(1, 1)]);
}

#[test]
fn filter_uses_non_projected_columns() {
    let file = util::make_db(|conn| {
        conn.execute_batch(
            "CREATE TABLE left_t (k INTEGER, flag INTEGER);
             CREATE TABLE right_t (k INTEGER, v INTEGER);",
        )
        .unwrap();
        conn.execute("INSERT INTO left_t (k, flag) VALUES (1, 1)", []).unwrap();
        conn.execute("INSERT INTO left_t (k, flag) VALUES (2, 0)", []).unwrap();
        conn.execute("INSERT INTO right_t (k, v) VALUES (1, 10)", []).unwrap();
        conn.execute("INSERT INTO right_t (k, v) VALUES (2, 20)", []).unwrap();
    });

    let db = Db::open(file.path()).unwrap();
    let left = db.table("left_t").unwrap();
    let right = db.table("right_t").unwrap();
    let mut scratch = new_scratch();
    let mut seen = Vec::new();

    Join::inner(left.scan().project([0]).filter(col(1).eq(lit_i64(1))), right.scan())
        .on(JoinKey::Col(0), JoinKey::Col(0))
        .strategy(JoinStrategy::Hash)
        .for_each(&mut scratch, |jr| {
            seen.push((jr.left.get_i64(0)?, jr.right.get_i64(1)?));
            Ok(())
        })
        .expect("join");

    assert_eq!(seen, vec![(1, 10)]);
}

#[test]
fn filter_on_right_uses_non_projected_columns() {
    let file = util::make_db(|conn| {
        conn.execute_batch(
            "CREATE TABLE left_t (k INTEGER);
             CREATE TABLE right_t (k INTEGER, flag INTEGER);",
        )
        .unwrap();
        conn.execute("INSERT INTO left_t (k) VALUES (1)", []).unwrap();
        conn.execute("INSERT INTO left_t (k) VALUES (2)", []).unwrap();
        conn.execute("INSERT INTO right_t (k, flag) VALUES (1, 1)", []).unwrap();
        conn.execute("INSERT INTO right_t (k, flag) VALUES (2, 0)", []).unwrap();
    });

    let db = Db::open(file.path()).unwrap();
    let left = db.table("left_t").unwrap();
    let right = db.table("right_t").unwrap();
    let mut scratch = new_scratch();
    let mut seen = Vec::new();

    Join::inner(left.scan(), right.scan().project([0]).filter(col(1).eq(lit_i64(1))))
        .on(JoinKey::Col(0), JoinKey::Col(0))
        .strategy(JoinStrategy::Hash)
        .for_each(&mut scratch, |jr| {
            seen.push((jr.left_rowid, jr.right_rowid, jr.right.get_i64(0)?));
            Ok(())
        })
        .expect("join");

    assert_eq!(seen, vec![(1, 1, 1)]);
}

#[test]
fn limit_interaction_left_and_right() {
    let file = make_int_join_db(&[Some(1), Some(2), Some(3)], &[Some(1), Some(2), Some(3)], true);
    let db = Db::open(file.path()).unwrap();
    let left = db.table("left_t").unwrap();
    let right = db.table("right_t").unwrap();
    let mut scratch = new_scratch();
    let mut seen = Vec::new();

    Join::inner(left.scan().limit(2), right.scan())
        .on(JoinKey::Col(0), JoinKey::Col(0))
        .strategy(JoinStrategy::Hash)
        .for_each(&mut scratch, |jr| {
            seen.push((jr.left_rowid, jr.right_rowid));
            Ok(())
        })
        .expect("join");
    assert_eq!(seen, vec![(1, 1), (2, 2)]);

    let mut scratch = new_scratch();
    let mut seen = Vec::new();
    Join::inner(left.scan(), right.scan().limit(1))
        .on(JoinKey::Col(0), JoinKey::Col(0))
        .strategy(JoinStrategy::Hash)
        .for_each(&mut scratch, |jr| {
            seen.push((jr.left_rowid, jr.right_rowid));
            Ok(())
        })
        .expect("join");
    assert_eq!(seen, vec![(1, 1)]);
}

#[test]
fn left_join_limit_respects_left_and_right() {
    let file = make_int_join_db(&[Some(1), Some(2), Some(3)], &[Some(1), Some(2), Some(3)], true);
    let db = Db::open(file.path()).unwrap();
    let left = db.table("left_t").unwrap();
    let right = db.table("right_t").unwrap();
    let mut scratch = new_scratch();
    let mut seen = Vec::new();

    Join::left(left.scan().limit(2), right.scan())
        .on(JoinKey::Col(0), JoinKey::Col(0))
        .strategy(JoinStrategy::Hash)
        .for_each(&mut scratch, |jr| {
            let right_all_null =
                (0..jr.right.len()).all(|idx| matches!(jr.right.get(idx), Some(ValueRef::Null)));
            seen.push((jr.left_rowid, jr.right_rowid, right_all_null));
            Ok(())
        })
        .expect("join");
    assert_eq!(seen, vec![(1, 1, false), (2, 2, false)]);

    let mut scratch = new_scratch();
    let mut seen = Vec::new();
    Join::left(left.scan(), right.scan().limit(1))
        .on(JoinKey::Col(0), JoinKey::Col(0))
        .strategy(JoinStrategy::Hash)
        .for_each(&mut scratch, |jr| {
            let right_all_null =
                (0..jr.right.len()).all(|idx| matches!(jr.right.get(idx), Some(ValueRef::Null)));
            seen.push((jr.left_rowid, jr.right_rowid, right_all_null));
            Ok(())
        })
        .expect("join");
    assert_eq!(seen, vec![(1, 1, false), (2, 0, true), (3, 0, true)]);
}

#[test]
fn auto_rowid_join_ignores_hash_mem_limit() {
    let file = util::make_db(|conn| {
        conn.execute_batch(
            "CREATE TABLE left_t (rid INTEGER);
             CREATE TABLE right_t (v INTEGER);",
        )
        .unwrap();
        conn.execute("INSERT INTO right_t (v) VALUES (10)", []).unwrap();
        conn.execute("INSERT INTO right_t (v) VALUES (20)", []).unwrap();
        conn.execute("INSERT INTO left_t (rid) VALUES (1)", []).unwrap();
        conn.execute("INSERT INTO left_t (rid) VALUES (2)", []).unwrap();
    });

    let db = Db::open(file.path()).unwrap();
    let left = db.table("left_t").unwrap();
    let right = db.table("right_t").unwrap();
    let mut scratch = new_scratch();
    let mut seen = Vec::new();

    Join::inner(left.scan(), right.scan())
        .on(JoinKey::Col(0), JoinKey::RowId)
        .strategy(JoinStrategy::Auto)
        .hash_mem_limit(0)
        .for_each(&mut scratch, |jr| {
            seen.push((jr.left_rowid, jr.right_rowid));
            Ok(())
        })
        .expect("auto rowid join");

    assert_eq!(seen, vec![(1, 1), (2, 2)]);
}

#[test]
fn auto_without_index_uses_hash() {
    let file = make_int_join_db(&[Some(1)], &[Some(1)], false);
    let db = Db::open(file.path()).unwrap();
    let left = db.table("left_t").unwrap();
    let right = db.table("right_t").unwrap();
    let mut scratch = new_scratch();
    let err = Join::inner(left.scan(), right.scan())
        .on(JoinKey::Col(0), JoinKey::Col(0))
        .strategy(JoinStrategy::Auto)
        .hash_mem_limit(0)
        .for_each(&mut scratch, |_jr| Ok(()))
        .unwrap_err();
    assert!(matches!(err, TableError::Join(JoinError::HashMemoryLimitExceeded)));
}

#[test]
fn hash_mem_limit_enforced() {
    let keys = vec!["a".repeat(64), "b".repeat(64), "c".repeat(64)];
    let file = util::make_db(|conn| {
        conn.execute_batch(
            "CREATE TABLE left_t (k TEXT);
             CREATE TABLE right_t (k TEXT);",
        )
        .unwrap();
        conn.execute("INSERT INTO left_t (k) VALUES (?1)", params![keys[0]]).unwrap();
        for key in &keys {
            conn.execute("INSERT INTO right_t (k) VALUES (?1)", params![key]).unwrap();
        }
    });

    let pager = Pager::new(std::fs::File::open(file.path()).unwrap()).unwrap();
    let (left_root, right_root, _) = join_roots(&pager, "left_t", "right_t", None);
    let db = Db::open(file.path()).unwrap();
    let left = db.table_root(left_root);
    let right = db.table_root(right_root);

    let mut payload_total = 0usize;
    table::scan_table_cells_with_scratch(&pager, right_root, |cell| {
        payload_total += cell.payload().to_vec()?.len();
        Ok(())
    })
    .unwrap();
    let key_total: usize = keys.iter().map(|k| k.len()).sum();
    let total = payload_total + key_total;

    let mut scratch = new_scratch();
    let err = Join::inner(left.scan(), right.scan())
        .on(JoinKey::Col(0), JoinKey::Col(0))
        .strategy(JoinStrategy::Hash)
        .hash_mem_limit(total.saturating_sub(1))
        .for_each(&mut scratch, |_jr| Ok(()))
        .unwrap_err();
    assert!(matches!(err, TableError::Join(JoinError::HashMemoryLimitExceeded)));

    let mut scratch = new_scratch();
    Join::inner(left.scan(), right.scan())
        .on(JoinKey::Col(0), JoinKey::Col(0))
        .strategy(JoinStrategy::Hash)
        .hash_mem_limit(total)
        .for_each(&mut scratch, |_jr| Ok(()))
        .expect("hash join under limit");
}

#[test]
fn filter_on_real_keys_and_zero_neg_zero() {
    let file = util::make_db(|conn| {
        conn.execute_batch(
            "CREATE TABLE left_t (k REAL);
             CREATE TABLE right_t (k REAL);",
        )
        .unwrap();
        conn.execute("INSERT INTO left_t (k) VALUES (?1)", params![-0.0_f64]).unwrap();
        conn.execute("INSERT INTO right_t (k) VALUES (?1)", params![0.0_f64]).unwrap();
    });

    let db = Db::open(file.path()).unwrap();
    let left = db.table("left_t").unwrap();
    let right = db.table("right_t").unwrap();
    let mut scratch = new_scratch();
    let mut seen = Vec::new();

    Join::inner(left.scan().filter(col(0).eq(lit_f64(0.0))), right.scan())
        .on(JoinKey::Col(0), JoinKey::Col(0))
        .strategy(JoinStrategy::Hash)
        .for_each(&mut scratch, |jr| {
            seen.push((jr.left_rowid, jr.right_rowid));
            Ok(())
        })
        .expect("join");

    assert_eq!(seen, vec![(1, 1)]);
}
