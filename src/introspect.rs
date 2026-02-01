use crate::pager::{PageId, Pager};
use crate::table::{self, ValueRef};

pub struct SchemaRow<'row> {
    pub kind: &'row str,
    pub name: &'row str,
    pub tbl_name: &'row str,
    pub root: PageId,
    pub sql: SchemaSql<'row>,
}

#[derive(Clone, Copy, Debug)]
pub enum SchemaSql<'row> {
    Text(&'row str),
    Null,
    InvalidUtf8,
}

impl<'row> SchemaSql<'row> {
    #[inline]
    pub fn as_str(self) -> Option<&'row str> {
        match self {
            Self::Text(sql) => Some(sql),
            Self::Null | Self::InvalidUtf8 => None,
        }
    }
}

pub fn scan_sqlite_schema_until<T, F>(pager: &Pager, mut f: F) -> table::Result<Option<T>>
where
    F: for<'row> FnMut(SchemaRow<'row>) -> table::Result<Option<T>>,
{
    let mut stack = Vec::with_capacity(64);
    let mut overflow_buf = Vec::new();
    let mut cache = table::RowCache::with_capacity(5);
    table::scan_table_cells_with_scratch_and_stack_until(pager, PageId::ROOT, &mut stack, |cell| {
        let payload = cell.payload();
        match payload {
            table::PayloadRef::Inline(bytes) => {
                let row = table::RowView::from_inline(bytes)?;
                let row = row.cached(&mut cache)?;
                if let Some(schema_row) = decode_schema_row(row)? {
                    return f(schema_row);
                }
                Ok(None)
            }
            table::PayloadRef::Overflow(_) => {
                overflow_buf = payload.to_vec()?;
                let row = table::RowView::from_inline(&overflow_buf)?;
                let row = row.cached(&mut cache)?;
                if let Some(schema_row) = decode_schema_row(row)? {
                    return f(schema_row);
                }
                Ok(None)
            }
        }
    })
}

pub fn scan_sqlite_schema<F>(pager: &Pager, mut f: F) -> table::Result<()>
where
    F: for<'row> FnMut(SchemaRow<'row>) -> table::Result<()>,
{
    scan_sqlite_schema_until::<(), _>(pager, |row| {
        f(row)?;
        Ok(None::<()>)
    })?;
    Ok(())
}

fn decode_schema_row<'row>(
    row: table::CachedRowView<'row, '_>,
) -> table::Result<Option<SchemaRow<'row>>> {
    let kind = decode_text(&row, 0)?;
    let name = decode_text(&row, 1)?;
    let tbl_name = decode_text(&row, 2)?;
    let (Some(kind), Some(name), Some(tbl_name)) = (kind, name, tbl_name) else {
        return Ok(None);
    };

    let rootpage = match row.get(3)? {
        Some(ValueRef::Integer(value)) => value,
        _ => return Ok(None),
    };
    let rootpage = match u32::try_from(rootpage).ok().and_then(PageId::try_new) {
        Some(value) => value,
        None => return Ok(None),
    };

    let sql = match row.get(4)? {
        Some(ValueRef::Text(bytes)) => match std::str::from_utf8(bytes) {
            Ok(value) => SchemaSql::Text(value),
            Err(_) => SchemaSql::InvalidUtf8,
        },
        Some(ValueRef::Null) | None => SchemaSql::Null,
        _ => SchemaSql::Null,
    };

    Ok(Some(SchemaRow { kind, name, tbl_name, root: rootpage, sql }))
}

fn decode_text<'row>(
    row: &table::CachedRowView<'row, '_>,
    col: usize,
) -> table::Result<Option<&'row str>> {
    match row.get(col)? {
        Some(ValueRef::Text(bytes)) => Ok(std::str::from_utf8(bytes).ok()),
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use rusqlite::Connection;
    use tempfile::NamedTempFile;

    use super::{SchemaSql, scan_sqlite_schema_until};
    use crate::db::Db;

    fn make_db<F: FnOnce(&Connection)>(f: F) -> NamedTempFile {
        let file = NamedTempFile::new().expect("create temp db file");
        let conn = Connection::open(file.path()).expect("open temp sqlite db");
        conn.execute_batch("PRAGMA journal_mode=DELETE; PRAGMA synchronous=OFF;")
            .expect("set sqlite pragmas");
        f(&conn);
        drop(conn);
        file
    }

    #[test]
    fn sqlite_schema_null_sql_for_autoindex() {
        let file = make_db(|conn| {
            conn.execute_batch("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT UNIQUE);")
                .expect("create table");
        });
        let db = Db::open(file.path()).expect("open db");
        let found = scan_sqlite_schema_until(db.pager(), |row| {
            if row.kind.eq_ignore_ascii_case("index") && matches!(row.sql, SchemaSql::Null) {
                return Ok(Some(()));
            }
            Ok(None)
        })
        .expect("scan sqlite_schema");
        assert!(found.is_some(), "expected an autoindex with NULL sql");
    }

    #[test]
    fn sqlite_schema_invalid_utf8_sql_is_reported() {
        let file = make_db(|conn| {
            conn.execute_batch("CREATE TABLE t (id INTEGER);").expect("create table");
            conn.execute_batch("PRAGMA writable_schema=ON;").expect("enable writable_schema");
            conn.execute_batch(
                "UPDATE sqlite_schema SET sql = CAST(X'80' AS TEXT) WHERE type='table' AND \
                 name='t';",
            )
            .expect("update sqlite_schema");
            let typeof_sql: String = conn
                .query_row(
                    "SELECT typeof(sql) FROM sqlite_schema WHERE type='table' AND name='t';",
                    [],
                    |row| row.get(0),
                )
                .expect("read typeof(sql)");
            assert_eq!(typeof_sql, "text");
            conn.execute_batch("PRAGMA writable_schema=OFF;").expect("disable writable_schema");
        });
        let db = Db::open(file.path()).expect("open db");
        let found = scan_sqlite_schema_until(db.pager(), |row| {
            if row.kind.eq_ignore_ascii_case("table") && row.name.eq_ignore_ascii_case("t") {
                return Ok(Some(matches!(row.sql, SchemaSql::InvalidUtf8)));
            }
            Ok(None)
        })
        .expect("scan sqlite_schema");
        assert_eq!(found, Some(true));
    }
}
