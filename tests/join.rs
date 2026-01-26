use std::path::{Path, PathBuf};

use miniql::index::{IndexCursor, IndexScratch};
use miniql::join::{Join, JoinKey, JoinScratch, JoinStrategy};
use miniql::pager::Pager;
use miniql::table::{self, ValueRef};
use miniql::{Db, PageId};

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn open_pager(name: &str) -> Pager {
    let file = std::fs::File::open(fixture_path(name)).expect("open fixture database");
    Pager::new(file).expect("create pager")
}

fn open_db(name: &str) -> Db {
    Db::open(fixture_path(name)).expect("open fixture database")
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

#[test]
fn index_cursor_seek_duplicates() {
    let pager = open_pager("join.db");
    let (_users_root, _orders_root, index_root) = join_roots(&pager);
    let mut scratch = IndexScratch::with_capacity(4, 0);
    let mut cursor = IndexCursor::new(&pager, index_root, 0, &mut scratch);

    let key = ValueRef::Integer(1);
    assert!(cursor.seek_ge(key).expect("seek"));

    let mut rowids = Vec::new();
    while cursor.key_eq(key).expect("key_eq") {
        rowids.push(cursor.current_rowid().expect("rowid"));
        if !cursor.next().expect("next") {
            break;
        }
    }

    assert_eq!(rowids, vec![10, 11]);
}

#[test]
fn lookup_rowid_payload_matches_scan() {
    let pager = open_pager("join.db");
    let (users_root, _orders_root, _index_root) = join_roots(&pager);

    let mut payloads = Vec::new();
    table::scan_table_cells_with_scratch(&pager, users_root, |cell| {
        let bytes = cell.payload().to_vec()?;
        payloads.push((cell.rowid(), bytes));
        Ok(())
    })
    .expect("scan users table");

    for (rowid, payload) in payloads {
        let found = table::lookup_rowid_cell(&pager, users_root, rowid)
            .expect("lookup rowid")
            .expect("row exists")
            .payload()
            .to_vec()
            .expect("materialize payload");
        assert_eq!(found, payload);
    }
}

#[test]
fn inner_join_users_orders_indexed() {
    let pager = open_pager("join.db");
    let db = open_db("join.db");
    let (users_root, orders_root, index_root) = join_roots(&pager);
    let mut scratch = JoinScratch::with_capacity(4, 4, 0);
    let mut seen = Vec::new();

    let users = db.table_root(users_root);
    let orders = db.table_root(orders_root);
    let mut join = Join::inner(users.scan(), orders.scan())
        .on(JoinKey::RowId, JoinKey::Col(0))
        .project_left([0])
        .project_right([1])
        .strategy(JoinStrategy::IndexNestedLoop { index_root, index_key_col: 0 })
        .compile()
        .expect("compile join");

    join.for_each(&mut scratch, |jr| {
        let name = jr.left.get_text(0)?.to_string();
        let amount = jr.right.get_i64(0)?;
        seen.push((jr.left_rowid, name, jr.right_rowid, amount));
        Ok(())
    })
    .expect("join");

    assert_eq!(
        seen,
        vec![
            (1, "alice".to_string(), 10, 100),
            (1, "alice".to_string(), 11, 120),
            (2, "bob".to_string(), 12, 75),
            (3, "cara".to_string(), 14, 10),
        ]
    );
}

#[test]
fn auto_discovers_index_for_join() {
    let pager = open_pager("join.db");
    let db = open_db("join.db");
    let (users_root, orders_root, _index_root) = join_roots(&pager);
    let mut scratch = JoinScratch::with_capacity(4, 4, 0);
    let mut seen = Vec::new();

    let users = db.table_root(users_root);
    let orders = db.table_root(orders_root);
    let mut join = Join::inner(users.scan(), orders.scan())
        .on(JoinKey::RowId, JoinKey::Col(0))
        .project_left([0])
        .project_right([1])
        .strategy(JoinStrategy::Auto)
        .hash_mem_limit(0)
        .compile()
        .expect("compile join");

    join.for_each(&mut scratch, |jr| {
        let name = jr.left.get_text(0)?.to_string();
        let amount = jr.right.get_i64(0)?;
        seen.push((jr.left_rowid, name, jr.right_rowid, amount));
        Ok(())
    })
    .expect("auto join");

    assert_eq!(
        seen,
        vec![
            (1, "alice".to_string(), 10, 100),
            (1, "alice".to_string(), 11, 120),
            (2, "bob".to_string(), 12, 75),
            (3, "cara".to_string(), 14, 10),
        ]
    );
}

#[test]
fn null_join_keys_do_not_match() {
    let pager = open_pager("join.db");
    let db = open_db("join.db");
    let (users_root, orders_root, _index_root) = join_roots(&pager);
    let mut scratch = JoinScratch::with_capacity(4, 4, 0);
    let mut count = 0usize;

    let orders = db.table_root(orders_root);
    let users = db.table_root(users_root);
    let mut join = Join::inner(orders.scan(), users.scan())
        .on(JoinKey::Col(0), JoinKey::RowId)
        .project_left([0])
        .project_right([0])
        .strategy(JoinStrategy::Hash)
        .compile()
        .expect("compile join");

    join.for_each(&mut scratch, |_jr| {
        count += 1;
        Ok(())
    })
    .expect("hash join");

    assert_eq!(count, 4);
}

#[test]
fn left_join_emits_null_right_rows() {
    let pager = open_pager("join.db");
    let db = open_db("join.db");
    let (users_root, orders_root, _index_root) = join_roots(&pager);
    let mut scratch = JoinScratch::with_capacity(4, 4, 0);
    let mut seen = Vec::new();

    let orders = db.table_root(orders_root);
    let users = db.table_root(users_root);
    let mut join = Join::left(orders.scan(), users.scan())
        .on(JoinKey::Col(0), JoinKey::RowId)
        .project_left([0, 1])
        .project_right([1])
        .strategy(JoinStrategy::Hash)
        .compile()
        .expect("compile join");

    join.for_each(&mut scratch, |jr| {
        let user_id = jr.left.get(0).and_then(|v| v.as_integer());
        let amount = jr.left.get_i64(1)?;
        let right_is_null = matches!(jr.right.get(0), Some(ValueRef::Null));
        seen.push((jr.left_rowid, user_id, amount, right_is_null));
        Ok(())
    })
    .expect("left join");

    let null_row = seen.iter().find(|(rowid, _, _, _)| *rowid == 13).expect("rowid 13 present");
    assert_eq!(null_row.1, None);
    assert_eq!(null_row.2, 50);
    assert!(null_row.3);
}

#[test]
fn auto_discovers_unique_autoindex() {
    let pager = open_pager("join_unique.db");
    let db = open_db("join_unique.db");
    let users_root = schema_root(&pager, "table", "users_unique").expect("users_unique root");
    let orders_root = schema_root(&pager, "table", "orders_unique").expect("orders_unique root");

    let mut scratch = JoinScratch::with_capacity(4, 4, 0);
    let mut seen = Vec::new();

    let orders = db.table_root(PageId::new(orders_root));
    let users = db.table_root(PageId::new(users_root));
    let mut join = Join::inner(orders.scan(), users.scan())
        .on(JoinKey::Col(1), JoinKey::Col(1))
        .project_left([1])
        .project_right([2])
        .strategy(JoinStrategy::Auto)
        .hash_mem_limit(0)
        .compile()
        .expect("compile join");

    join.for_each(&mut scratch, |jr| {
        let user_name = jr.left.get_text(0)?.to_string();
        let full_name = jr.right.get_text(0)?.to_string();
        seen.push((jr.left_rowid, user_name, full_name));
        Ok(())
    })
    .expect("auto join");

    assert_eq!(
        seen,
        vec![
            (10, "alice".to_string(), "Alice A".to_string()),
            (11, "alice".to_string(), "Alice A".to_string()),
            (12, "bob".to_string(), "Bob B".to_string()),
            (14, "cara".to_string(), "Cara C".to_string()),
        ]
    );
}

#[test]
fn auto_discovers_composite_primary_key_autoindex() {
    let pager = open_pager("join_pk.db");
    let db = open_db("join_pk.db");
    let pk_root = schema_root(&pager, "table", "composite_pk").expect("composite_pk root");
    let orders_root = schema_root(&pager, "table", "pk_orders").expect("pk_orders root");

    let mut scratch = JoinScratch::with_capacity(4, 4, 0);
    let mut seen = Vec::new();

    let orders = db.table_root(PageId::new(orders_root));
    let pk = db.table_root(PageId::new(pk_root));
    let mut join = Join::inner(orders.scan(), pk.scan())
        .on(JoinKey::Col(1), JoinKey::Col(0))
        .project_left([1])
        .project_right([2])
        .strategy(JoinStrategy::Auto)
        .hash_mem_limit(0)
        .compile()
        .expect("compile join");

    join.for_each(&mut scratch, |jr| {
        let a = jr.left.get_i64(0)?;
        let payload = jr.right.get_text(0)?.to_string();
        seen.push((jr.left_rowid, a, payload));
        Ok(())
    })
    .expect("auto join");

    assert_eq!(
        seen,
        vec![
            (10, 1, "a1".to_string()),
            (10, 1, "a2".to_string()),
            (11, 1, "a1".to_string()),
            (11, 1, "a2".to_string()),
            (12, 2, "b1".to_string()),
        ]
    );
}
