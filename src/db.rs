use std::cell::OnceCell;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use crate::introspect::{SchemaRow, scan_sqlite_schema, scan_sqlite_schema_until};
use crate::pager::{PageId, Pager};
use crate::query::{Scan, ScanScratch};
use crate::schema::parse_table_schema;
use crate::table;

/// Read-only handle to a SQLite database file.
pub struct Db {
    pager: Pager,
    schema: OnceCell<SchemaCache>,
}

#[derive(Clone)]
struct TableInfo {
    name: String,
    root: PageId,
    column_count: Option<usize>,
}

struct SchemaCache {
    tables: HashMap<String, TableInfo>,
}

impl SchemaCache {
    fn load(pager: &Pager) -> table::Result<Self> {
        let mut tables = HashMap::new();

        scan_sqlite_schema(pager, |row| {
            if !row.kind.eq_ignore_ascii_case("table") {
                return Ok(());
            }

            let column_count = column_count_from_schema_sql(row.sql.as_str());
            let info = TableInfo { name: row.name.to_owned(), root: row.root, column_count };

            tables.insert(row.name.to_ascii_lowercase(), info);
            Ok(())
        })?;

        Ok(Self { tables })
    }

    fn table(&self, name: &str) -> Option<&TableInfo> {
        self.tables.get(&name.to_ascii_lowercase())
    }
}

#[inline]
fn column_count_from_schema_sql(sql: Option<&str>) -> Option<usize> {
    let sql = sql?;
    let schema = parse_table_schema(sql);
    (!schema.columns.is_empty()).then_some(schema.columns.len())
}

impl Db {
    /// Open a SQLite database file.
    ///
    /// ```rust
    /// use std::path::Path;
    ///
    /// use miniql::Db;
    ///
    /// let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/users.db");
    /// let db = Db::open(path).unwrap();
    /// let _ = db.table("users").unwrap();
    /// ```
    pub fn open(path: impl AsRef<Path>) -> table::Result<Self> {
        let file =
            File::open(path).map_err(|err| table::Error::Pager(crate::pager::Error::Io(err)))?;
        let pager = Pager::new(file)?;
        Ok(Self { pager, schema: OnceCell::new() })
    }

    /// Look up a table by name using `sqlite_schema`.
    pub fn table(&self, name: &str) -> table::Result<Table<'_>> {
        let cache = self.schema_cache()?;
        let info = cache.table(name).ok_or(table::Error::TableNotFound)?;
        Ok(Table {
            db: self,
            root: info.root,
            name: info.name.clone(),
            column_count: info.column_count,
        })
    }

    /// Create a table handle from a root page id.
    pub fn table_root(&self, root: PageId) -> Table<'_> {
        Table { db: self, root, name: String::new(), column_count: None }
    }

    /// Create an index handle from a root page id.
    pub fn index_root(&self, root: PageId) -> Index<'_> {
        Index { db: self, root }
    }

    pub(crate) fn pager(&self) -> &Pager {
        &self.pager
    }

    /// Visit rows in `sqlite_schema` with a borrowed view of each row.
    pub fn scan_schema<F>(&self, f: F) -> table::Result<()>
    where
        F: for<'row> FnMut(SchemaRow<'row>) -> table::Result<()>,
    {
        scan_sqlite_schema(self.pager(), f)
    }

    /// Visit rows in `sqlite_schema` until the callback returns `Some`.
    pub fn scan_schema_until<T, F>(&self, f: F) -> table::Result<Option<T>>
    where
        F: for<'row> FnMut(SchemaRow<'row>) -> table::Result<Option<T>>,
    {
        scan_sqlite_schema_until(self.pager(), f)
    }

    fn schema_cache(&self) -> table::Result<&SchemaCache> {
        if self.schema.get().is_none() {
            let cache = SchemaCache::load(&self.pager)?;
            // If set fails, someone else set it first (still fine).
            let _ = self.schema.set(cache);
        }

        Ok(self.schema.get().expect("schema cache initialized"))
    }
}

/// Handle to a table b-tree.
pub struct Table<'db> {
    db: &'db Db,
    root: PageId,
    name: String,
    column_count: Option<usize>,
}

impl<'db> Table<'db> {
    /// Return the root page id for this table.
    pub fn root(&self) -> PageId {
        self.root
    }

    /// Create a row scan for this table.
    pub fn scan(&self) -> Scan<'db> {
        Scan::from_root_with_hint(self.db.pager(), self.root, self.column_count)
    }

    /// Create a scan that yields raw table cells.
    pub fn scan_cells(&self) -> CellScan<'db> {
        CellScan { pager: self.db.pager(), root: self.root }
    }

    /// Return the table name if known.
    pub fn name(&self) -> Option<&str> {
        (!self.name.is_empty()).then_some(self.name.as_str())
    }
}

/// Handle to an index b-tree.
pub struct Index<'db> {
    db: &'db Db,
    root: PageId,
}

impl<'db> Index<'db> {
    /// Return the root page id for this index.
    pub fn root(&self) -> PageId {
        self.root
    }

    /// Return the parent database handle.
    pub fn table(&self) -> &'db Db {
        self.db
    }
}

/// Scan that yields raw table cells.
pub struct CellScan<'db> {
    pager: &'db Pager,
    root: PageId,
}

impl<'db> CellScan<'db> {
    /// Visit each cell in a table b-tree.
    pub fn for_each<F>(self, scratch: &mut ScanScratch, mut cb: F) -> table::Result<()>
    where
        F: for<'row> FnMut(table::CellRef<'row>) -> table::Result<()>,
    {
        let (_, _, _, _, stack) = scratch.split_mut();
        table::scan_table_cells_with_scratch_and_stack(self.pager, self.root, stack, |cell| {
            cb(cell)
        })
    }
}
