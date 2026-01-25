use std::path::Path;

use rusqlite::Connection;
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
