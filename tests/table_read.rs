use std::path::{Path, PathBuf};

use miniql::pager::{PageId, Pager};
use miniql::table::{self, TableRow, Value, ValueRef};

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn open_pager(name: &str) -> Pager {
    let file = std::fs::File::open(fixture_path(name)).expect("open fixture database");
    Pager::new(file).expect("create pager")
}

fn users_root_page(rows: &[TableRow]) -> Option<u32> {
    rows.iter().find_map(|row| {
        if row.values.len() < 4 {
            return None;
        }

        let row_type = row.values[0].as_text()?;
        let name = row.values[1].as_text()?;
        let rootpage = row.values[3].as_integer()?;

        if row_type == "table" && name == "users" { u32::try_from(rootpage).ok() } else { None }
    })
}

#[test]
fn reads_schema_and_finds_users_root() {
    let pager = open_pager("users.db");
    let rows = table::read_table(&pager, PageId::ROOT).expect("read sqlite_schema");
    let root = users_root_page(&rows).expect("users table entry in sqlite_schema");
    assert_eq!(root, 2);
}

#[test]
fn reads_users_table_rows() {
    let pager = open_pager("users.db");
    let rows = table::read_table(&pager, PageId::new(2)).expect("read users table");

    assert_eq!(rows.len(), 2);

    let first = &rows[0];
    assert_eq!(first.rowid, 1);
    assert_eq!(first.values.len(), 3);
    assert!(matches!(first.values[0], Value::Null));
    assert_eq!(first.values[1].as_text(), Some("alice"));
    assert_eq!(first.values[2].as_integer(), Some(30));

    let second = &rows[1];
    assert_eq!(second.rowid, 2);
    assert_eq!(second.values.len(), 3);
    assert!(matches!(second.values[0], Value::Null));
    assert_eq!(second.values[1].as_text(), Some("bob"));
    assert_eq!(second.values[2].as_integer(), Some(25));
}

#[test]
fn reads_users_table_rows_by_ref() {
    let pager = open_pager("users.db");
    let rows = table::read_table_ref(&pager, PageId::new(2)).expect("read users table");

    assert_eq!(rows.len(), 2);

    let first = &rows[0];
    assert_eq!(first.rowid, 1);
    assert!(matches!(first.values[0], Value::Null));
    assert_eq!(first.values[1].as_text(), Some("alice"));
    assert_eq!(first.values[2].as_integer(), Some(30));
}

#[test]
fn scans_users_table_rows() {
    let pager = open_pager("users.db");
    let mut seen = 0usize;
    table::scan_table_ref(&pager, PageId::new(2), |rowid, values| {
        match seen {
            0 => {
                assert_eq!(rowid, 1);
                assert_eq!(values.len(), 3);
                assert!(matches!(values[0], ValueRef::Null));
                assert_eq!(values[1].as_text(), Some("alice"));
                assert_eq!(values[2].as_integer(), Some(30));
            }
            1 => {
                assert_eq!(rowid, 2);
                assert_eq!(values.len(), 3);
                assert!(matches!(values[0], ValueRef::Null));
                assert_eq!(values[1].as_text(), Some("bob"));
                assert_eq!(values[2].as_integer(), Some(25));
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
    let mut rows = 0usize;
    table::scan_table_ref(&pager, PageId::new(2), |rowid, values| {
        assert_eq!(rowid, 1);
        assert_eq!(values.len(), 2);
        assert_eq!(values[0].as_integer(), Some(1));
        let text = values[1].as_text().expect("utf8 text");
        assert!(text.len() > 5000);
        assert!(text.starts_with("payload-"));
        rows += 1;
        Ok(())
    })
    .expect("scan overflow table");
    assert_eq!(rows, 1);
}

#[test]
fn scans_tables_with_interior_pages() {
    let pager = open_pager("interior.db");
    let mut rows = 0usize;
    table::scan_table_ref(&pager, PageId::new(2), |_, values| {
        assert_eq!(values.len(), 2);
        rows += 1;
        Ok(())
    })
    .expect("scan interior table");
    assert!(rows > 200);
}
