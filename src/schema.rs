use sqlparser::ast::{
    ColumnDef, ColumnOption, CreateIndex, CreateTable, DataType, Expr, IndexColumn, Statement,
    TableConstraint,
};
use sqlparser::dialect::{GenericDialect, SQLiteDialect};
use sqlparser::parser::Parser;

#[derive(Clone, Debug)]
pub(crate) struct TableSchema {
    pub(crate) columns: Vec<String>,
    pub(crate) unique_indexes: Vec<Vec<String>>,
    pub(crate) without_rowid: bool,
}

pub(crate) fn parse_table_schema(sql: &str) -> TableSchema {
    let Some(create) = parse_create_table(sql) else {
        return TableSchema {
            columns: Vec::new(),
            unique_indexes: Vec::new(),
            without_rowid: false,
        };
    };

    let mut columns = Vec::with_capacity(create.columns.len());
    let mut unique_indexes = Vec::new();

    for column in &create.columns {
        let name = column.name.value.to_ascii_lowercase();
        let has_primary = column
            .options
            .iter()
            .any(|option| matches!(&option.option, ColumnOption::PrimaryKey(_)));
        let has_unique =
            column.options.iter().any(|option| matches!(&option.option, ColumnOption::Unique(_)));
        let integer_primary = is_integer_type(column) && has_primary;
        columns.push(name.clone());
        if (has_primary || has_unique) && !integer_primary {
            unique_indexes.push(vec![name]);
        }
    }

    for constraint in &create.constraints {
        if let Some(cols) = parse_table_constraint_index_cols(constraint) {
            unique_indexes.push(cols);
        }
    }

    TableSchema { columns, unique_indexes, without_rowid: create.without_rowid }
}

pub(crate) fn parse_index_columns(sql: &str) -> Option<Vec<String>> {
    let create = parse_create_index(sql)?;
    if create.predicate.is_some() {
        return None;
    }
    parse_index_column_names(&create.columns)
}

pub(crate) fn parse_index_is_unique(sql: &str) -> bool {
    parse_create_index(sql).is_some_and(|create| create.unique)
}

fn parse_create_table(sql: &str) -> Option<CreateTable> {
    parse_statements(sql)?.into_iter().find_map(|statement| match statement {
        Statement::CreateTable(create) => Some(create),
        _ => None,
    })
}

fn parse_create_index(sql: &str) -> Option<CreateIndex> {
    parse_statements(sql)?.into_iter().find_map(|statement| match statement {
        Statement::CreateIndex(create) => Some(create),
        _ => None,
    })
}

fn parse_statements(sql: &str) -> Option<Vec<Statement>> {
    let sqlite = SQLiteDialect {};
    Parser::parse_sql(&sqlite, sql).ok().or_else(|| {
        let generic = GenericDialect {};
        Parser::parse_sql(&generic, sql).ok()
    })
}

fn is_integer_type(column: &ColumnDef) -> bool {
    matches!(column.data_type, DataType::Integer(_))
}

fn parse_table_constraint_index_cols(constraint: &TableConstraint) -> Option<Vec<String>> {
    match constraint {
        TableConstraint::PrimaryKey(constraint) => parse_index_column_names(&constraint.columns),
        TableConstraint::Unique(constraint) => parse_index_column_names(&constraint.columns),
        _ => None,
    }
}

fn parse_index_column_names(columns: &[IndexColumn]) -> Option<Vec<String>> {
    let cols = columns.iter().map(parse_index_column_name).collect::<Option<Vec<_>>>()?;
    (!cols.is_empty()).then_some(cols)
}

fn parse_index_column_name(column: &IndexColumn) -> Option<String> {
    match &column.column.expr {
        Expr::Identifier(ident) => Some(ident.value.to_ascii_lowercase()),
        Expr::CompoundIdentifier(idents) => {
            idents.last().map(|ident| ident.value.to_ascii_lowercase())
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_table_schema_and_unique_constraints() {
        let sql = "CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT UNIQUE, tenant_id \
                   INTEGER, name TEXT, UNIQUE(tenant_id, name));";
        let schema = parse_table_schema(sql);
        assert_eq!(schema.columns, vec!["id", "email", "tenant_id", "name"]);
        assert_eq!(schema.unique_indexes, vec![vec!["email"], vec!["tenant_id", "name"]]);
        assert!(!schema.without_rowid);
    }

    #[test]
    fn parses_table_schema_without_rowid() {
        let sql = "CREATE TABLE t (a INTEGER, b TEXT, PRIMARY KEY(a, b)) WITHOUT ROWID;";
        let schema = parse_table_schema(sql);
        assert_eq!(schema.columns, vec!["a", "b"]);
        assert_eq!(schema.unique_indexes, vec![vec!["a", "b"]]);
        assert!(schema.without_rowid);
    }

    #[test]
    fn parses_index_columns() {
        let sql = "CREATE INDEX idx_users_lookup ON users(email, \"tenant_id\");";
        let cols = parse_index_columns(sql).expect("parse index columns");
        assert_eq!(cols, vec!["email", "tenant_id"]);
    }

    #[test]
    fn rejects_partial_indexes() {
        let sql = "CREATE UNIQUE INDEX idx_users_email ON users(email) WHERE email IS NOT NULL;";
        assert!(parse_index_columns(sql).is_none());
    }

    #[test]
    fn detects_unique_indexes() {
        assert!(parse_index_is_unique("CREATE UNIQUE INDEX idx_users_email ON users(email);"));
        assert!(!parse_index_is_unique("CREATE INDEX idx_users_email ON users(email);"));
    }
}
