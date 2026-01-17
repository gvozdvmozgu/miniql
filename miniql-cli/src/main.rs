use std::fs::File;
use std::process;

use miniql::pager::{PageId, Pager};
use miniql::table::{self, TableRowRef};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(usage);
    let table_name = args.next().unwrap_or_else(|| "sqlite_schema".to_string());

    let file = File::open(&path).unwrap_or_else(|err| {
        eprintln!("Failed to open '{path}': {err}");
        process::exit(1);
    });

    let pager = Pager::new(file).unwrap_or_else(|err| {
        eprintln!("Failed to create pager: {err}");
        process::exit(1);
    });

    if let Err(err) = print_table(&pager, &table_name) {
        eprintln!("Error: {err}");
        process::exit(1);
    }
}

fn print_table(pager: &Pager, table_name: &str) -> Result<(), table::Error> {
    let root_page = if table_name == "sqlite_schema" {
        PageId::ROOT
    } else {
        let schema_rows = table::read_table_ref(pager, PageId::ROOT)?;
        let root_page = find_root_page(&schema_rows, table_name)
            .ok_or_else(|| table::Error::TableNotFound(table_name.to_owned()))?;

        PageId::try_new(root_page).ok_or(table::Error::Corrupted("table root page is zero"))?
    };

    let rows = table::read_table_ref(pager, root_page)?;
    println!("table: {table_name} (root page {})", root_page.into_inner());
    for row in rows {
        print_row(&row);
    }

    Ok(())
}

fn find_root_page(rows: &[TableRowRef<'_>], table_name: &str) -> Option<u32> {
    rows.iter().find_map(|row| {
        if row.values.len() < 4 {
            return None;
        }

        let row_type = row.values.first()?.as_text()?;
        let name = row.values.get(1)?.as_text()?;
        let root_page = row.values.get(3)?.as_integer()?;

        if row_type == "table" && name == table_name { u32::try_from(root_page).ok() } else { None }
    })
}

fn print_row(row: &TableRowRef<'_>) {
    print!("{:>6} |", row.rowid);
    for value in &row.values {
        print!(" {value} |");
    }
    println!();
}

fn usage() -> String {
    eprintln!("Usage: miniql <path-to-db-file> [table-name]");
    process::exit(1);
}
