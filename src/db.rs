use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::OnceLock;

use crate::pager::{PageId, Pager};
use crate::query::{Scan, ScanScratch};
use crate::schema::parse_table_schema;
use crate::table;

/// Read-only handle to a SQLite database file.
pub struct Db {
    pager: Pager,
    schema: OnceLock<SchemaCache>,
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
        let rows = table::read_table(pager, PageId::ROOT)?;
        let mut tables = HashMap::new();

        for row in rows {
            if row.values.len() < 5 {
                continue;
            }
            let row_type = row.values[0].as_text();
            let Some(row_type) = row_type else { continue };
            if !row_type.eq_ignore_ascii_case("table") {
                continue;
            }
            let Some(name) = row.values[1].as_text() else { continue };
            let Some(rootpage) = row.values[3].as_integer() else { continue };
            let Ok(rootpage) = u32::try_from(rootpage) else { continue };
            let Some(root) = PageId::try_new(rootpage) else { continue };
            let sql = row.values[4].as_text();
            let column_count = sql.and_then(|sql| {
                let schema = parse_table_schema(sql);
                if schema.columns.is_empty() { None } else { Some(schema.columns.len()) }
            });

            let info = TableInfo { name: name.to_owned(), root, column_count };
            tables.insert(name.to_ascii_lowercase(), info);
        }

        Ok(Self { tables })
    }

    fn table(&self, name: &str) -> Option<&TableInfo> {
        self.tables.get(&name.to_ascii_lowercase())
    }
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
        Ok(Self { pager, schema: OnceLock::new() })
    }

    /// Look up a table by name using `sqlite_schema`.
    pub fn table(&self, name: &str) -> table::Result<Table<'_>> {
        let cache = self.schema_cache()?;
        let info = cache.table(name).ok_or_else(|| table::Error::TableNotFound(name.to_owned()))?;
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

    fn schema_cache(&self) -> table::Result<&SchemaCache> {
        if let Some(cache) = self.schema.get() {
            return Ok(cache);
        }
        let cache = SchemaCache::load(&self.pager)?;
        let _ = self.schema.set(cache);
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
        if self.name.is_empty() { None } else { Some(self.name.as_str()) }
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
        table::scan_table_cells_with_scratch_and_stack_until::<_, ()>(
            self.pager,
            self.root,
            stack,
            |cell| {
                cb(cell)?;
                Ok(None)
            },
        )?;
        Ok(())
    }
}
