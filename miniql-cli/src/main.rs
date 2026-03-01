use std::process;

use miniql::{Db, PageId, Row, ScanScratch, ValueRef};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(usage);
    let mode = args.next();

    let db = Db::open(&path).unwrap_or_else(|err| {
        eprintln!("Failed to open '{path}': {err}");
        process::exit(1);
    });

    let result = match mode.as_deref() {
        Some("--sql") => {
            let query = args.collect::<Vec<_>>().join(" ");
            if query.trim().is_empty() {
                eprintln!("Missing SQL query after --sql");
                process::exit(1);
            }
            print_sql(&db, &query)
        }
        Some(table_name) => print_table(&db, table_name),
        None => print_table(&db, "sqlite_schema"),
    };

    if let Err(err) = result {
        eprintln!("Error: {err}");
        process::exit(1);
    }
}

fn print_table(db: &Db, table_name: &str) -> Result<(), miniql::Error> {
    let table = if table_name == "sqlite_schema" {
        db.table_from_root(PageId::ROOT)
    } else {
        db.table(table_name)?
    };

    println!("table: {table_name} (root page {})", table.root().into_inner());
    let mut scan = table.scan().compile()?;
    let mut scratch = ScanScratch::with_capacity(8, 0);
    let mut row_cache = miniql::RowCache::new();
    scan.for_each(&mut scratch, |rowid, row| {
        let cached = row.cached(&mut row_cache)?;
        print_row(rowid, &cached);
        Ok(())
    })?;

    Ok(())
}

fn print_row(rowid: i64, row: &miniql::CachedRowView<'_, '_>) {
    print!("{:>6} |", rowid);
    for idx in 0..row.column_count() {
        match row.get(idx) {
            Ok(Some(value)) => print!(" {value} |"),
            Ok(None) => print!(" <missing> |"),
            Err(_) => print!(" <error> |"),
        }
    }
    println!();
}

fn print_sql(db: &Db, query: &str) -> Result<(), miniql::Error> {
    let mut scratch = ScanScratch::with_capacity(8, 0);
    db.query(query, &mut scratch, |row| {
        print_sql_row(row);
        Ok(())
    })?;
    Ok(())
}

fn print_sql_row(row: Row<'_>) {
    print!("|");
    for idx in 0..row.len() {
        match row.get(idx) {
            Some(ValueRef::Null) => print!(" NULL |"),
            Some(value) => print!(" {value} |"),
            None => print!(" <missing> |"),
        }
    }
    println!();
}

fn usage() -> String {
    eprintln!("Usage: miniql <path-to-db-file> [table-name]");
    eprintln!("   or: miniql <path-to-db-file> --sql \"<select-query>\"");
    process::exit(1);
}
