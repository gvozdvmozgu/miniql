use std::path::{Path, PathBuf};

use miniql::{Db, ScanScratch, ValueRef, col, lit_bytes, lit_i64};

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

    let mut scan = users
        .scan()
        .project([1, 2])
        .filter(col(2).gt(lit_i64(25)))
        .compile()
        .expect("compile scan");
    scan.for_each(&mut scratch, |rowid, row| {
        let name = row.get_text(0)?;
        let age = row.get_i64(1)?;
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

    let mut scan =
        users.scan().project([1]).filter(col(0).is_null()).compile().expect("compile scan");
    scan.for_each(&mut scratch, |_, _| {
        count += 1;
        Ok(())
    })
    .expect("scan users table");

    assert_eq!(count, 2);

    let mut scratch = ScanScratch::with_capacity(4, 0);
    let mut count = 0usize;

    let mut scan =
        users.scan().project([1]).filter(col(0).eq(lit_i64(1))).compile().expect("compile scan");
    scan.for_each(&mut scratch, |_, _| {
        count += 1;
        Ok(())
    })
    .expect("scan users table");

    assert_eq!(count, 0);
}

#[test]
fn projection_remaps_columns() {
    let db = open_db("users.db");
    let users = db.table("users").expect("users table");
    let mut scratch = ScanScratch::with_capacity(4, 0);
    let mut first = None;

    let mut scan = users
        .scan()
        .project([2, 1])
        .filter(col(1).eq(lit_bytes(b"alice")))
        .compile()
        .expect("compile scan");
    scan.for_each(&mut scratch, |rowid, row| {
        let age = row.get_i64(0)?;
        let name = row.get_text(1)?;
        first = Some((rowid, name.to_owned(), age));
        Ok(())
    })
    .expect("scan users table");

    assert_eq!(first, Some((1, "alice".to_string(), 30)));
}

#[test]
fn scan_without_projection_decodes_all_columns() {
    let db = open_db("users.db");
    let users = db.table("users").expect("users table");
    let mut scratch = ScanScratch::with_capacity(4, 0);
    let mut first = None;

    let mut scan = users.scan().filter(col(2).gt(lit_i64(25))).compile().expect("compile scan");
    scan.for_each(&mut scratch, |rowid, row| {
        assert!(matches!(row.get(0), Some(ValueRef::Null)));
        let name = row.get_text(1)?;
        let age = row.get_i64(2)?;
        first = Some((rowid, name.to_owned(), age));
        Ok(())
    })
    .expect("scan users table");

    assert_eq!(first, Some((1, "alice".to_string(), 30)));
}
