use std::path::{Path, PathBuf};

use miniql::pager::{PageId, Pager};
use miniql::query::{Scan, ScanScratch, col, lit_bytes, lit_i64};

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn open_pager(name: &str) -> Pager {
    let file = std::fs::File::open(fixture_path(name)).expect("open fixture database");
    Pager::new(file).expect("create pager")
}

#[test]
fn filters_rows_with_predicate() {
    let pager = open_pager("users.db");
    let mut scratch = ScanScratch::with_capacity(4, 0);
    let mut seen = Vec::new();

    Scan::table(&pager, PageId::new(2))
        .project([1, 2])
        .filter(col(2).gt(lit_i64(25)))
        .for_each(&mut scratch, |rowid, row| {
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
    let pager = open_pager("users.db");
    let mut scratch = ScanScratch::with_capacity(4, 0);
    let mut count = 0usize;

    Scan::table(&pager, PageId::new(2))
        .project([1])
        .filter(col(0).is_null())
        .for_each(&mut scratch, |_, _| {
            count += 1;
            Ok(())
        })
        .expect("scan users table");

    assert_eq!(count, 2);

    let mut scratch = ScanScratch::with_capacity(4, 0);
    let mut count = 0usize;

    Scan::table(&pager, PageId::new(2))
        .project([1])
        .filter(col(0).eq(lit_i64(1)))
        .for_each(&mut scratch, |_, _| {
            count += 1;
            Ok(())
        })
        .expect("scan users table");

    assert_eq!(count, 0);
}

#[test]
fn projection_remaps_columns() {
    let pager = open_pager("users.db");
    let mut scratch = ScanScratch::with_capacity(4, 0);
    let mut first = None;

    Scan::table(&pager, PageId::new(2))
        .project([2, 1])
        .filter(col(1).eq(lit_bytes(b"alice")))
        .for_each(&mut scratch, |rowid, row| {
            let age = row.get_i64(0)?;
            let name = row.get_text(1)?;
            first = Some((rowid, name.to_owned(), age));
            Ok(())
        })
        .expect("scan users table");

    assert_eq!(first, Some((1, "alice".to_string(), 30)));
}
