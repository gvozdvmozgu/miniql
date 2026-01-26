use std::path::{Path, PathBuf};

use miniql::pager::{PageId, Pager};
use miniql::table::{self, ValueRef};

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn open_pager(name: &str) -> Pager {
    let file = std::fs::File::open(fixture_path(name)).expect("open fixture database");
    Pager::new(file).expect("create pager")
}

fn row_text<'row>(row: table::RowView<'row>, idx: usize) -> table::Result<Option<&'row str>> {
    match row.get(idx)? {
        Some(ValueRef::Text(bytes)) => Ok(std::str::from_utf8(bytes).ok()),
        _ => Ok(None),
    }
}

fn row_integer(row: table::RowView<'_>, idx: usize) -> table::Result<Option<i64>> {
    match row.get(idx)? {
        Some(ValueRef::Integer(value)) => Ok(Some(value)),
        _ => Ok(None),
    }
}

#[test]
fn reads_schema_and_finds_users_root() {
    let pager = open_pager("users.db");
    let mut found = None;
    table::scan_table(&pager, PageId::ROOT, |_, row| {
        let row_type = match row_text(row, 0)? {
            Some(value) => value,
            None => return Ok(()),
        };
        if !row_type.eq_ignore_ascii_case("table") {
            return Ok(());
        }
        let name = match row_text(row, 1)? {
            Some(value) => value,
            None => return Ok(()),
        };
        if name != "users" {
            return Ok(());
        }
        let rootpage = match row_integer(row, 3)? {
            Some(value) => value,
            None => return Ok(()),
        };
        found = u32::try_from(rootpage).ok();
        Ok(())
    })
    .expect("read sqlite_schema");
    let root = found.expect("users table entry in sqlite_schema");
    assert_eq!(root, 2);
}

#[test]
fn scans_users_table_rows() {
    let pager = open_pager("users.db");
    let mut seen = 0usize;
    table::scan_table(&pager, PageId::new(2), |rowid, row| {
        match seen {
            0 => {
                assert_eq!(rowid, 1);
                assert_eq!(row.column_count(), 3);
                assert!(matches!(row.get(0)?, Some(ValueRef::Null)));
                assert_eq!(row.get(1)?.and_then(|v| v.as_text()), Some("alice"));
                assert_eq!(row.get(2)?.and_then(|v| v.as_integer()), Some(30));
            }
            1 => {
                assert_eq!(rowid, 2);
                assert_eq!(row.column_count(), 3);
                assert!(matches!(row.get(0)?, Some(ValueRef::Null)));
                assert_eq!(row.get(1)?.and_then(|v| v.as_text()), Some("bob"));
                assert_eq!(row.get(2)?.and_then(|v| v.as_integer()), Some(25));
            }
            _ => panic!("unexpected extra row"),
        }
        seen += 1;
        Ok(())
    })
    .expect("scan users table");

    assert_eq!(seen, 2);
}

#[test]
fn opens_with_different_page_sizes() {
    let pager_1024 = open_pager("users_1024.db");
    assert_eq!(pager_1024.header().page_size, 1024);
    let pager_8192 = open_pager("users_8192.db");
    assert_eq!(pager_8192.header().page_size, 8192);
}

#[test]
fn reads_overflow_payloads() {
    let pager = open_pager("overflow.db");
    let mut seen = 0usize;
    table::scan_table_cells_with_scratch(&pager, PageId::new(2), |cell| {
        assert_eq!(cell.rowid(), 1);
        let payload = cell.payload().to_vec()?;
        let row = table::RowView::from_inline(&payload)?;
        assert_eq!(row.column_count(), 2);
        assert_eq!(row.get(0)?.and_then(|v| v.as_integer()), Some(1));
        let text = row.get(1)?.and_then(|v| v.as_text()).expect("utf8 text");
        assert!(text.len() > 5000);
        assert!(text.starts_with("payload-"));
        seen += 1;
        Ok(())
    })
    .expect("scan overflow table");
    assert_eq!(seen, 1);
}

#[test]
fn scans_tables_with_interior_pages() {
    let pager = open_pager("interior.db");
    let mut rows = 0usize;
    table::scan_table(&pager, PageId::new(2), |_, row| {
        assert_eq!(row.column_count(), 2);
        rows += 1;
        Ok(())
    })
    .expect("scan interior table");
    assert!(rows > 200);
}
