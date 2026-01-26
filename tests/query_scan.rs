mod util;

use std::path::{Path, PathBuf};

use miniql::{Db, ScanScratch, asc, col, desc, lit_bytes, lit_i64};
use rusqlite::params;

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn open_db(name: &str) -> Db {
    Db::open(fixture_path(name)).expect("open fixture database")
}

#[test]
fn filters_rows_with_predicate() {
    let db = open_db("users.db");
    let users = db.table("users").expect("users table");
    let mut scratch = ScanScratch::with_capacity(4, 0);
    let mut seen = Vec::new();

    let mut scan = users.scan().filter(col(2).gt(lit_i64(25))).compile().expect("compile scan");
    scan.for_each(&mut scratch, |rowid, row| {
        let name = row.get_text(1)?;
        let age = row.get_i64(2)?;
        seen.push((rowid, name.to_owned(), age));
        Ok(())
    })
    .expect("scan users table");

    assert_eq!(seen, vec![(1, "alice".to_string(), 30)]);
}

#[test]
fn null_predicate_behavior() {
    let db = open_db("users.db");
    let users = db.table("users").expect("users table");
    let mut scratch = ScanScratch::with_capacity(4, 0);
    let mut count = 0usize;

    let mut scan = users.scan().filter(col(0).is_null()).compile().expect("compile scan");
    scan.for_each(&mut scratch, |_, _| {
        count += 1;
        Ok(())
    })
    .expect("scan users table");

    assert_eq!(count, 2);

    let mut scratch = ScanScratch::with_capacity(4, 0);
    let mut count = 0usize;

    let mut scan = users.scan().filter(col(0).eq(lit_i64(1))).compile().expect("compile scan");
    scan.for_each(&mut scratch, |_, _| {
        count += 1;
        Ok(())
    })
    .expect("scan users table");

    assert_eq!(count, 0);
}

#[test]
fn filter_with_text_column() {
    let db = open_db("users.db");
    let users = db.table("users").expect("users table");
    let mut scratch = ScanScratch::with_capacity(4, 0);
    let mut first = None;

    let mut scan =
        users.scan().filter(col(1).eq(lit_bytes(b"alice"))).compile().expect("compile scan");
    scan.for_each(&mut scratch, |rowid, row| {
        let name = row.get_text(1)?;
        let age = row.get_i64(2)?;
        first = Some((rowid, name.to_owned(), age));
        Ok(())
    })
    .expect("scan users table");

    assert_eq!(first, Some((1, "alice".to_string(), 30)));
}

#[test]
fn scan_reads_all_columns() {
    let db = open_db("users.db");
    let users = db.table("users").expect("users table");
    let mut scratch = ScanScratch::with_capacity(4, 0);
    let mut first = None;

    let mut scan = users.scan().filter(col(2).gt(lit_i64(25))).compile().expect("compile scan");
    scan.for_each(&mut scratch, |rowid, row| {
        assert!(matches!(row.get(0), Ok(Some(miniql::ValueRef::Null)))); // col 0 is NULL
        let name = row.get_text(1)?;
        let age = row.get_i64(2)?;
        first = Some((rowid, name.to_owned(), age));
        Ok(())
    })
    .expect("scan users table");

    assert_eq!(first, Some((1, "alice".to_string(), 30)));
}

#[test]
fn order_by_multi_column_with_limit() {
    let file = util::make_db(|conn| {
        conn.execute_batch("CREATE TABLE people (name TEXT, age INTEGER, city TEXT);").unwrap();
        let rows = [
            ("alice", 30, "austin"),
            ("bob", 25, "boston"),
            ("carol", 25, "chicago"),
            ("dave", 40, "denver"),
            ("eve", 35, "elpaso"),
        ];
        for (name, age, city) in rows {
            conn.execute(
                "INSERT INTO people (name, age, city) VALUES (?1, ?2, ?3)",
                params![name, age, city],
            )
            .unwrap();
        }
    });

    let db = Db::open(file.path()).expect("open temp db");
    let people = db.table("people").expect("people table");
    let mut scratch = ScanScratch::with_capacity(3, 0);
    let mut seen = Vec::new();

    people
        .scan()
        .order_by([asc(1), desc(0)])
        .limit(3)
        .for_each(&mut scratch, |_, row| {
            seen.push(row.get_text(0)?.to_owned());
            Ok(())
        })
        .expect("scan people table");

    assert_eq!(seen, vec!["carol".to_string(), "bob".to_string(), "alice".to_string()]);
}
