use std::path::Path;

use miniql::{Db, Error as TableError, ScanScratch, ValueRef};
use rusqlite::Connection;
use rusqlite::types::ValueRef as SqliteValueRef;
use tempfile::NamedTempFile;

pub fn make_db<F: FnOnce(&Connection)>(f: F) -> NamedTempFile {
    let file = NamedTempFile::new().expect("create temp db file");
    init_db(file.path(), f);
    file
}

fn init_db<F: FnOnce(&Connection)>(path: &Path, f: F) {
    let conn = Connection::open(path).expect("open temp sqlite db");
    conn.execute_batch("PRAGMA journal_mode=DELETE; PRAGMA synchronous=OFF;")
        .expect("set sqlite pragmas");
    f(&conn);
    drop(conn);
}

#[allow(dead_code)]
pub fn collect_miniql_rows(
    db: &Db,
    query: &str,
    scratch_cols: usize,
) -> Result<Vec<Vec<QueryValue>>, TableError> {
    let mut rows = Vec::new();
    let mut scratch = ScanScratch::with_capacity(scratch_cols, 0);
    db.query(query, &mut scratch, |row| {
        let mut out = Vec::with_capacity(row.len());
        for idx in 0..row.len() {
            out.push(query_value_from_miniql(row.get(idx)));
        }
        rows.push(out);
        Ok(())
    })?;
    Ok(rows)
}

#[allow(dead_code)]
pub fn collect_sqlite_rows(
    conn: &Connection,
    query: &str,
) -> rusqlite::Result<Vec<Vec<QueryValue>>> {
    let mut stmt = conn.prepare(query)?;
    let col_count = stmt.column_count();
    let mut rows_iter = stmt.query([])?;
    let mut rows = Vec::new();
    while let Some(row) = rows_iter.next()? {
        let mut out = Vec::with_capacity(col_count);
        for idx in 0..col_count {
            let value = row.get_ref(idx)?;
            out.push(query_value_from_sqlite(value));
        }
        rows.push(out);
    }
    Ok(rows)
}

#[allow(dead_code)]
pub fn execute_sqlite_query(conn: &Connection, sql: &str) -> rusqlite::Result<()> {
    let mut stmt = conn.prepare(sql)?;
    let col_count = stmt.column_count();
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        for idx in 0..col_count {
            let _ = row.get_ref(idx)?;
        }
    }
    Ok(())
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryValue {
    Null,
    Integer(i64),
    Real(u64),
    Text(Vec<u8>),
    Blob(Vec<u8>),
}

#[allow(dead_code)]
pub fn query_value_from_miniql(value: Option<ValueRef<'_>>) -> QueryValue {
    match value {
        Some(ValueRef::Null) | None => QueryValue::Null,
        Some(ValueRef::Integer(v)) => QueryValue::Integer(v),
        Some(ValueRef::Real(v)) => QueryValue::Real(normalized_real_bits(v)),
        Some(ValueRef::Text(bytes)) => QueryValue::Text(bytes.to_vec()),
        Some(ValueRef::Blob(bytes)) => QueryValue::Blob(bytes.to_vec()),
    }
}

#[allow(dead_code)]
pub fn query_value_from_sqlite(value: SqliteValueRef<'_>) -> QueryValue {
    match value {
        SqliteValueRef::Null => QueryValue::Null,
        SqliteValueRef::Integer(v) => QueryValue::Integer(v),
        SqliteValueRef::Real(v) => QueryValue::Real(normalized_real_bits(v)),
        SqliteValueRef::Text(bytes) => QueryValue::Text(bytes.to_vec()),
        SqliteValueRef::Blob(bytes) => QueryValue::Blob(bytes.to_vec()),
    }
}

#[allow(dead_code)]
fn normalized_real_bits(value: f64) -> u64 {
    if value == 0.0 {
        0.0f64.to_bits()
    } else if value.is_nan() {
        0x7ff8_0000_0000_0000
    } else {
        value.to_bits()
    }
}
