use std::path::Path;

use miniql::pager::{PageId, Pager};
use miniql::table::{self, TableRow, Value};

fn open_pager() -> Pager {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/users.db");
    let file = std::fs::File::open(path).expect("open fixture database");
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
    let mut pager = open_pager();
    let rows = table::read_table(&mut pager, PageId::ROOT).expect("read sqlite_schema");
    let root = users_root_page(&rows).expect("users table entry in sqlite_schema");
    assert_eq!(root, 2);
}

#[test]
fn reads_users_table_rows() {
    let mut pager = open_pager();
    let rows = table::read_table(&mut pager, PageId::new(2)).expect("read users table");

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
