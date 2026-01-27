mod util;

use miniql::{
    Db, Error as TableError, ScanScratch, ValueRef, avg, col, count, count_star, group, lit_i64,
    max, min, sum,
};
use rusqlite::params;

#[test]
fn group_by_aggregates_with_nulls() {
    let file = util::make_db(|conn| {
        conn.execute_batch("CREATE TABLE items (grp TEXT, val INTEGER);").unwrap();
        let rows: [(Option<&str>, Option<i64>); 6] = [
            (Some("a"), Some(1)),
            (Some("a"), None),
            (Some("b"), Some(2)),
            (Some("b"), Some(2)),
            (None, Some(5)),
            (None, None),
        ];
        for (grp, val) in rows {
            conn.execute("INSERT INTO items (grp, val) VALUES (?1, ?2)", params![grp, val])
                .unwrap();
        }
    });

    let db = Db::open(file.path()).expect("open temp db");
    let items = db.table("items").expect("items table");
    let mut scratch = ScanScratch::with_capacity(2, 0);
    let mut seen = Vec::new();

    items
        .scan()
        .aggregate([
            group(col(0)),
            count_star(),
            count(col(1)),
            sum(col(1)),
            avg(col(1)),
            min(col(1)),
            max(col(1)),
        ])
        .group_by([col(0)])
        .for_each(&mut scratch, |row| {
            let group = match row.get(0) {
                Some(ValueRef::Text(bytes)) => Some(String::from_utf8_lossy(bytes).to_string()),
                Some(ValueRef::Null) | None => None,
                other => panic!("unexpected group value: {other:?}"),
            };
            let count_all = match row.get(1) {
                Some(ValueRef::Integer(v)) => v,
                other => panic!("unexpected count(*): {other:?}"),
            };
            let count_val = match row.get(2) {
                Some(ValueRef::Integer(v)) => v,
                other => panic!("unexpected count(val): {other:?}"),
            };
            let sum_val = match row.get(3) {
                Some(ValueRef::Integer(v)) => Some(v),
                Some(ValueRef::Null) | None => None,
                other => panic!("unexpected sum(val): {other:?}"),
            };
            let avg_val = match row.get(4) {
                Some(ValueRef::Real(v)) => Some(v),
                Some(ValueRef::Null) | None => None,
                other => panic!("unexpected avg(val): {other:?}"),
            };
            let min_val = match row.get(5) {
                Some(ValueRef::Integer(v)) => Some(v),
                Some(ValueRef::Null) | None => None,
                other => panic!("unexpected min(val): {other:?}"),
            };
            let max_val = match row.get(6) {
                Some(ValueRef::Integer(v)) => Some(v),
                Some(ValueRef::Null) | None => None,
                other => panic!("unexpected max(val): {other:?}"),
            };

            seen.push((group, count_all, count_val, sum_val, avg_val, min_val, max_val));
            Ok(())
        })
        .expect("aggregate items");

    seen.sort_by(|left, right| left.0.cmp(&right.0));

    let expected = vec![
        (None, 2, 1, Some(5), Some(5.0), Some(5), Some(5)),
        (Some("a".to_string()), 2, 1, Some(1), Some(1.0), Some(1), Some(1)),
        (Some("b".to_string()), 2, 2, Some(4), Some(2.0), Some(2), Some(2)),
    ];

    assert_eq!(seen, expected);
}

#[test]
fn aggregates_without_group_by_single_row() {
    let file = util::make_db(|conn| {
        conn.execute_batch("CREATE TABLE nums (val INTEGER);").unwrap();
        conn.execute("INSERT INTO nums (val) VALUES (1), (2), (NULL);", []).unwrap();
    });

    let db = Db::open(file.path()).expect("open temp db");
    let nums = db.table("nums").expect("nums table");
    let mut scratch = ScanScratch::with_capacity(1, 0);
    let mut seen = Vec::new();

    nums.scan()
        .aggregate([count_star(), sum(col(0)), avg(col(0))])
        .for_each(&mut scratch, |row| {
            let count_all = match row.get(0) {
                Some(ValueRef::Integer(v)) => v,
                other => panic!("unexpected count(*): {other:?}"),
            };
            let sum_val = match row.get(1) {
                Some(ValueRef::Integer(v)) => Some(v),
                Some(ValueRef::Null) | None => None,
                other => panic!("unexpected sum(val): {other:?}"),
            };
            let avg_val = match row.get(2) {
                Some(ValueRef::Real(v)) => Some(v),
                Some(ValueRef::Null) | None => None,
                other => panic!("unexpected avg(val): {other:?}"),
            };
            seen.push((count_all, sum_val, avg_val));
            Ok(())
        })
        .expect("aggregate nums");

    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0], (3, Some(3), Some(1.5)));
}

#[test]
fn aggregates_with_empty_input_sets() {
    let file = util::make_db(|conn| {
        conn.execute_batch("CREATE TABLE nums (val INTEGER);").unwrap();
        conn.execute("INSERT INTO nums (val) VALUES (1), (2), (NULL);", []).unwrap();
    });

    let db = Db::open(file.path()).expect("open temp db");
    let nums = db.table("nums").expect("nums table");
    let mut scratch = ScanScratch::with_capacity(1, 0);
    let mut seen = Vec::new();

    nums.scan()
        .filter(col(0).gt(lit_i64(99)))
        .aggregate([count_star(), sum(col(0)), avg(col(0)), min(col(0)), max(col(0))])
        .for_each(&mut scratch, |row| {
            let count_all = match row.get(0) {
                Some(ValueRef::Integer(v)) => v,
                other => panic!("unexpected count(*): {other:?}"),
            };
            let sum_val: Option<i64> = match row.get(1) {
                Some(ValueRef::Null) | None => None,
                other => panic!("unexpected sum(val): {other:?}"),
            };
            let avg_val: Option<f64> = match row.get(2) {
                Some(ValueRef::Null) | None => None,
                other => panic!("unexpected avg(val): {other:?}"),
            };
            let min_val: Option<i64> = match row.get(3) {
                Some(ValueRef::Null) | None => None,
                other => panic!("unexpected min(val): {other:?}"),
            };
            let max_val: Option<i64> = match row.get(4) {
                Some(ValueRef::Null) | None => None,
                other => panic!("unexpected max(val): {other:?}"),
            };
            seen.push((count_all, sum_val, avg_val, min_val, max_val));
            Ok(())
        })
        .expect("aggregate nums with empty input");

    assert_eq!(seen, vec![(0, None, None, None, None)]);
}

#[test]
fn group_by_multiple_keys() {
    let file = util::make_db(|conn| {
        conn.execute_batch("CREATE TABLE pairs (a TEXT, b TEXT);").unwrap();
        let rows = [("a", "x"), ("a", "x"), ("a", "y"), ("b", "x")];
        for (a, b) in rows {
            conn.execute("INSERT INTO pairs (a, b) VALUES (?1, ?2)", params![a, b]).unwrap();
        }
    });

    let db = Db::open(file.path()).expect("open temp db");
    let pairs = db.table("pairs").expect("pairs table");
    let mut scratch = ScanScratch::with_capacity(2, 0);
    let mut seen = Vec::new();

    pairs
        .scan()
        .aggregate([group(col(0)), group(col(1)), count_star()])
        .group_by([col(0), col(1)])
        .for_each(&mut scratch, |row| {
            let a = match row.get(0) {
                Some(ValueRef::Text(bytes)) => String::from_utf8_lossy(bytes).to_string(),
                other => panic!("unexpected a value: {other:?}"),
            };
            let b = match row.get(1) {
                Some(ValueRef::Text(bytes)) => String::from_utf8_lossy(bytes).to_string(),
                other => panic!("unexpected b value: {other:?}"),
            };
            let count_all = match row.get(2) {
                Some(ValueRef::Integer(v)) => v,
                other => panic!("unexpected count(*): {other:?}"),
            };
            seen.push((a, b, count_all));
            Ok(())
        })
        .expect("aggregate pairs");

    seen.sort();
    assert_eq!(
        seen,
        vec![
            ("a".to_string(), "x".to_string(), 2),
            ("a".to_string(), "y".to_string(), 1),
            ("b".to_string(), "x".to_string(), 1),
        ]
    );
}

#[test]
fn having_filters_groups() {
    let file = util::make_db(|conn| {
        conn.execute_batch("CREATE TABLE items (grp TEXT, val INTEGER);").unwrap();
        let rows: [(Option<&str>, Option<i64>); 6] = [
            (Some("a"), Some(1)),
            (Some("a"), None),
            (Some("b"), Some(2)),
            (Some("b"), Some(2)),
            (None, Some(5)),
            (None, None),
        ];
        for (grp, val) in rows {
            conn.execute("INSERT INTO items (grp, val) VALUES (?1, ?2)", params![grp, val])
                .unwrap();
        }
    });

    let db = Db::open(file.path()).expect("open temp db");
    let items = db.table("items").expect("items table");
    let mut scratch = ScanScratch::with_capacity(2, 0);
    let mut seen = Vec::new();

    items
        .scan()
        .aggregate([group(col(0)), sum(col(1))])
        .group_by([col(0)])
        .having(col(1).gt(lit_i64(2)))
        .for_each(&mut scratch, |row| {
            let group = match row.get(0) {
                Some(ValueRef::Text(bytes)) => Some(String::from_utf8_lossy(bytes).to_string()),
                Some(ValueRef::Null) | None => None,
                other => panic!("unexpected group value: {other:?}"),
            };
            let sum_val = match row.get(1) {
                Some(ValueRef::Integer(v)) => v,
                other => panic!("unexpected sum(val): {other:?}"),
            };
            seen.push((group, sum_val));
            Ok(())
        })
        .expect("aggregate items with having");

    seen.sort_by(|left, right| left.0.cmp(&right.0));
    assert_eq!(seen, vec![(None, 5), (Some("b".to_string()), 4)]);
}

#[test]
fn validates_non_grouped_expression() {
    let file = util::make_db(|conn| {
        conn.execute_batch("CREATE TABLE items (grp TEXT, val INTEGER);").unwrap();
        conn.execute("INSERT INTO items (grp, val) VALUES ('a', 1);", []).unwrap();
    });

    let db = Db::open(file.path()).expect("open temp db");
    let items = db.table("items").expect("items table");

    let err = items.scan().aggregate([group(col(0)), group(col(1))]).group_by([col(0)]).compile();

    assert!(matches!(err, Err(TableError::Query(_))));
}
