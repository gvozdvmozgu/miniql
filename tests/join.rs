use std::path::{Path, PathBuf};

use miniql::index::{IndexCursor, IndexScratch};
use miniql::join::{Join, JoinKey, JoinScratch, JoinStrategy, JoinType};
use miniql::pager::{PageId, Pager};
use miniql::query::Scan;
use miniql::table::{self, TableRow, ValueRef};

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn open_pager(name: &str) -> Pager {
    let file = std::fs::File::open(fixture_path(name)).expect("open fixture database");
    Pager::new(file).expect("create pager")
}

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

fn join_roots(pager: &Pager) -> (PageId, PageId, PageId) {
    let rows = table::read_table(pager, PageId::ROOT).expect("read sqlite_schema");
    let users_root = schema_root(&rows, "table", "users").expect("users table root");
    let orders_root = schema_root(&rows, "table", "orders").expect("orders table root");
    let index_root = schema_root(&rows, "index", "orders_user_id_idx").expect("orders index root");
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
    table::scan_table_cells_with_scratch(&pager, users_root, |rowid, payload| {
        let bytes = payload.to_vec()?;
        payloads.push((rowid, bytes));
        Ok(())
    })
    .expect("scan users table");

    for (rowid, payload) in payloads {
        let found = table::lookup_rowid_payload(&pager, users_root, rowid)
            .expect("lookup rowid")
            .expect("row exists")
            .to_vec()
            .expect("materialize payload");
        assert_eq!(found, payload);
    }
}

#[test]
fn inner_join_users_orders_indexed() {
    let pager = open_pager("join.db");
    let (users_root, orders_root, index_root) = join_roots(&pager);
    let mut scratch = JoinScratch::with_capacity(4, 4, 0);
    let mut seen = Vec::new();

    Join::new(JoinType::Inner, Scan::table(&pager, users_root), Scan::table(&pager, orders_root))
        .on(JoinKey::RowId, JoinKey::Col(0))
        .project_left([0])
        .project_right([1])
        .strategy(JoinStrategy::IndexNestedLoop { index_root, index_key_col: 0 })
        .for_each(&mut scratch, |jr| {
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
    let (users_root, orders_root, _index_root) = join_roots(&pager);
    let mut scratch = JoinScratch::with_capacity(4, 4, 0);
    let mut seen = Vec::new();

    Join::new(JoinType::Inner, Scan::table(&pager, users_root), Scan::table(&pager, orders_root))
        .on(JoinKey::RowId, JoinKey::Col(0))
        .project_left([0])
        .project_right([1])
        .strategy(JoinStrategy::Auto)
        .hash_mem_limit(0)
        .for_each(&mut scratch, |jr| {
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
    let (users_root, orders_root, _index_root) = join_roots(&pager);
    let mut scratch = JoinScratch::with_capacity(4, 4, 0);
    let mut count = 0usize;

    Join::new(JoinType::Inner, Scan::table(&pager, orders_root), Scan::table(&pager, users_root))
        .on(JoinKey::Col(0), JoinKey::RowId)
        .project_left([0])
        .project_right([0])
        .strategy(JoinStrategy::HashJoin)
        .for_each(&mut scratch, |_jr| {
            count += 1;
            Ok(())
        })
        .expect("hash join");

    assert_eq!(count, 4);
}

#[test]
fn left_join_emits_null_right_rows() {
    let pager = open_pager("join.db");
    let (users_root, orders_root, _index_root) = join_roots(&pager);
    let mut scratch = JoinScratch::with_capacity(4, 4, 0);
    let mut seen = Vec::new();

    Join::new(JoinType::Left, Scan::table(&pager, orders_root), Scan::table(&pager, users_root))
        .on(JoinKey::Col(0), JoinKey::RowId)
        .project_left([0, 1])
        .project_right([1])
        .strategy(JoinStrategy::HashJoin)
        .for_each(&mut scratch, |jr| {
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
    let rows = table::read_table(&pager, PageId::ROOT).expect("read sqlite_schema");
    let users_root = schema_root(&rows, "table", "users_unique").expect("users_unique root");
    let orders_root = schema_root(&rows, "table", "orders_unique").expect("orders_unique root");

    let mut scratch = JoinScratch::with_capacity(4, 4, 0);
    let mut seen = Vec::new();

    Join::new(
        JoinType::Inner,
        Scan::table(&pager, PageId::new(orders_root)),
        Scan::table(&pager, PageId::new(users_root)),
    )
    .on(JoinKey::Col(1), JoinKey::Col(1))
    .project_left([1])
    .project_right([2])
    .strategy(JoinStrategy::Auto)
    .hash_mem_limit(0)
    .for_each(&mut scratch, |jr| {
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
    let rows = table::read_table(&pager, PageId::ROOT).expect("read sqlite_schema");
    let pk_root = schema_root(&rows, "table", "composite_pk").expect("composite_pk root");
    let orders_root = schema_root(&rows, "table", "pk_orders").expect("pk_orders root");

    let mut scratch = JoinScratch::with_capacity(4, 4, 0);
    let mut seen = Vec::new();

    Join::new(
        JoinType::Inner,
        Scan::table(&pager, PageId::new(orders_root)),
        Scan::table(&pager, PageId::new(pk_root)),
    )
    .on(JoinKey::Col(1), JoinKey::Col(0))
    .project_left([1])
    .project_right([2])
    .strategy(JoinStrategy::Auto)
    .hash_mem_limit(0)
    .for_each(&mut scratch, |jr| {
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
