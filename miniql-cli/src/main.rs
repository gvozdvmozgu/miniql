use std::fs::File;
use std::process;

use miniql::pager::{PageId, Pager};
use miniql::table::{self, RowView};

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
        find_root_page(pager, table_name)?
    };

    println!("table: {table_name} (root page {})", root_page.into_inner());
    table::scan_table_ref(pager, root_page, |rowid, row| {
        print_row(rowid, row);
        Ok(())
    })?;

    Ok(())
}

fn find_root_page(pager: &Pager, table_name: &str) -> Result<PageId, table::Error> {
    let found = table::scan_table_ref_until(pager, PageId::ROOT, |_, row| {
        if row.len() < 4 {
            return Ok(None);
        }

        let row_type = row.get(0).and_then(|v| v.text_bytes());
        let name = row.get(1).and_then(|v| v.text_bytes());
        let root_page = row.get(3).and_then(|v| v.as_integer());

        if let (Some(row_type), Some(name), Some(root_page)) = (row_type, name, root_page)
            && row_type == b"table"
            && name == table_name.as_bytes()
            && let Ok(root_page) = u32::try_from(root_page)
        {
            return match PageId::try_new(root_page) {
                Some(page_id) => Ok(Some(page_id)),
                None => Err(table::Error::Corrupted("table root page is zero")),
            };
        }

        Ok(None)
    })?;

    found.ok_or_else(|| table::Error::TableNotFound(table_name.to_owned()))
}

fn print_row(rowid: i64, row: RowView<'_>) {
    print!("{:>6} |", rowid);
    for value in row.iter() {
        print!(" {value} |");
    }
    println!();
}

fn usage() -> String {
    eprintln!("Usage: miniql <path-to-db-file> [table-name]");
    process::exit(1);
}
