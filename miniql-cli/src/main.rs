use std::process;

use miniql::{Db, PageId, ScanScratch};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(usage);
    let table_name = args.next().unwrap_or_else(|| "sqlite_schema".to_string());

    let db = Db::open(&path).unwrap_or_else(|err| {
        eprintln!("Failed to open '{path}': {err}");
        process::exit(1);
    });

    if let Err(err) = print_table(&db, &table_name) {
        eprintln!("Error: {err}");
        process::exit(1);
    }
}

fn print_table(db: &Db, table_name: &str) -> Result<(), miniql::Error> {
    let table = if table_name == "sqlite_schema" {
        db.table_root(PageId::ROOT)
    } else {
        db.table(table_name)?
    };

    println!("table: {table_name} (root page {})", table.root().into_inner());
    let mut scan = table.scan().compile()?;
    let mut scratch = ScanScratch::with_capacity(8, 0);
    scan.for_each(&mut scratch, |rowid, row| {
        print_row(rowid, row);
        Ok(())
    })?;

    Ok(())
}

fn print_row(rowid: i64, row: miniql::RowView<'_>) {
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

fn usage() -> String {
    eprintln!("Usage: miniql <path-to-db-file> [table-name]");
    process::exit(1);
}
