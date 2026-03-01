mod util;

use miniql::table::QueryError;
use miniql::{Db, Error as TableError, ScanScratch};
use rusqlite::{Connection, params};
use util::{collect_miniql_rows, collect_sqlite_rows, execute_sqlite_query};

#[derive(Clone, Copy)]
struct NegativeCase {
    name: &'static str,
    sql: &'static str,
    expected: QueryError,
    sqlite_accepts: bool,
}

#[derive(Clone, Copy)]
struct NegativeMatrix<'a> {
    name: &'static str,
    cases: &'a [NegativeCase],
}

#[derive(Clone, Copy)]
struct DivergenceCase {
    name: &'static str,
    sql: &'static str,
}

#[test]
fn negative_sql_parity_cases() {
    let file = util::make_db(|conn| {
        conn.execute_batch(
            "CREATE TABLE users (id INTEGER, name TEXT, city TEXT);
             CREATE TABLE orders (id INTEGER, user_id INTEGER, amount INTEGER, note TEXT);
             CREATE TABLE items (grp TEXT, val INTEGER);",
        )
        .expect("create negative parity tables");

        let users =
            [(1_i64, "alice", Some("austin")), (2, "bob", Some("boston")), (3, "cara", None)];
        for (id, name, city) in users {
            conn.execute(
                "INSERT INTO users (id, name, city) VALUES (?1, ?2, ?3)",
                params![id, name, city],
            )
            .expect("insert users");
        }

        let orders = [
            (10_i64, Some(1_i64), 100_i64, "o1"),
            (11, Some(1), 40, "o2"),
            (12, Some(2), 75, "o3"),
            (13, None, 10, "orphan_null"),
        ];
        for (id, user_id, amount, note) in orders {
            conn.execute(
                "INSERT INTO orders (id, user_id, amount, note) VALUES (?1, ?2, ?3, ?4)",
                params![id, user_id, amount, note],
            )
            .expect("insert orders");
        }

        let items = [("a", 100_i64), ("b", 75), ("c", 10)];
        for (grp, val) in items {
            conn.execute("INSERT INTO items (grp, val) VALUES (?1, ?2)", params![grp, val])
                .expect("insert items");
        }
    });

    let db = Db::open(file.path()).expect("open miniql db");
    let conn = Connection::open(file.path()).expect("open sqlite db");

    let core_cases = [
        NegativeCase {
            name: "parse_error",
            sql: "SELEC name FROM users",
            expected: QueryError::SqlParse,
            sqlite_accepts: false,
        },
        NegativeCase {
            name: "unsupported_union",
            sql: "SELECT name FROM users UNION SELECT name FROM users",
            expected: QueryError::SqlUnsupported,
            sqlite_accepts: true,
        },
        NegativeCase {
            name: "unsupported_cte",
            sql: "WITH c AS (SELECT name FROM users) SELECT name FROM c",
            expected: QueryError::SqlUnsupported,
            sqlite_accepts: true,
        },
        NegativeCase {
            name: "unsupported_projection_expression",
            sql: "SELECT name || city FROM users",
            expected: QueryError::SqlUnsupported,
            sqlite_accepts: true,
        },
        NegativeCase {
            name: "unsupported_join_non_equality_on",
            sql: "SELECT u.name, o.amount FROM users AS u JOIN orders AS o ON u.id > o.user_id",
            expected: QueryError::SqlUnsupported,
            sqlite_accepts: true,
        },
        NegativeCase {
            name: "unsupported_multi_join",
            sql: "SELECT u.name FROM users AS u JOIN orders AS o ON u.id = o.user_id JOIN items \
                  AS i ON i.val = o.amount",
            expected: QueryError::SqlUnsupported,
            sqlite_accepts: true,
        },
        NegativeCase {
            name: "unknown_column",
            sql: "SELECT missing FROM users",
            expected: QueryError::SqlUnknownColumn,
            sqlite_accepts: false,
        },
        NegativeCase {
            name: "ambiguous_column",
            sql: "SELECT id FROM users AS u JOIN orders AS o ON u.id = o.user_id",
            expected: QueryError::SqlAmbiguousColumn,
            sqlite_accepts: false,
        },
        NegativeCase {
            name: "invalid_limit_negative",
            sql: "SELECT name FROM users LIMIT -1",
            expected: QueryError::SqlInvalidLimitOffset,
            sqlite_accepts: true,
        },
        NegativeCase {
            name: "invalid_order_ordinal_out_of_range",
            sql: "SELECT name FROM users ORDER BY 2",
            expected: QueryError::SqlInvalidLimitOffset,
            sqlite_accepts: false,
        },
    ];

    let edge_semantics_cases = [
        NegativeCase {
            name: "unsupported_like_escape",
            sql: "SELECT name FROM users WHERE name LIKE 'a!%' ESCAPE '!'",
            expected: QueryError::SqlUnsupported,
            sqlite_accepts: true,
        },
        NegativeCase {
            name: "unsupported_order_by_collate_nocase",
            sql: "SELECT name FROM users ORDER BY name COLLATE NOCASE",
            expected: QueryError::SqlUnsupported,
            sqlite_accepts: true,
        },
        NegativeCase {
            name: "unsupported_where_collate_nocase",
            sql: "SELECT name FROM users WHERE name COLLATE NOCASE = 'ALICE'",
            expected: QueryError::SqlUnsupported,
            sqlite_accepts: true,
        },
        NegativeCase {
            name: "unsupported_order_by_nulls_last",
            sql: "SELECT name FROM users ORDER BY city NULLS LAST",
            expected: QueryError::SqlUnsupported,
            sqlite_accepts: true,
        },
        NegativeCase {
            name: "unsupported_order_by_nulls_first",
            sql: "SELECT name FROM users ORDER BY city NULLS FIRST",
            expected: QueryError::SqlUnsupported,
            sqlite_accepts: true,
        },
    ];

    let matrices = [
        NegativeMatrix { name: "core_negative", cases: &core_cases },
        NegativeMatrix { name: "edge_semantics_negative", cases: &edge_semantics_cases },
    ];

    for matrix in matrices {
        for case in matrix.cases {
            let mut scratch = ScanScratch::with_capacity(16, 0);
            let err = db
                .query(case.sql, &mut scratch, |_| Ok(()))
                .expect_err("miniql should reject negative SQL case");
            assert!(
                matches!(err, TableError::Query(query_err) if query_err == case.expected),
                "matrix '{}' case '{}' expected miniql error {:?}, got {:?}\nsql: {}",
                matrix.name,
                case.name,
                case.expected,
                err,
                case.sql
            );

            let sqlite_accepts = execute_sqlite_query(&conn, case.sql).is_ok();
            assert_eq!(
                sqlite_accepts, case.sqlite_accepts,
                "matrix '{}' case '{}' sqlite acceptance mismatch (expected {}, got {})\nsql: {}",
                matrix.name, case.name, case.sqlite_accepts, sqlite_accepts, case.sql
            );
        }
    }
}

#[test]
fn negative_sql_parity_edge_numeric_text_divergence_matrix() {
    let file = util::make_db(|conn| {
        conn.execute_batch(
            "CREATE TABLE users (id INTEGER, name TEXT, city TEXT);
             INSERT INTO users (id, name, city) VALUES
               (1, 'alice', 'austin'),
               (2, 'bob', 'boston'),
               (3, 'cara', NULL);",
        )
        .expect("create divergence parity table");
    });

    let db = Db::open(file.path()).expect("open miniql db");
    let conn = Connection::open(file.path()).expect("open sqlite db");

    let matrix = [
        DivergenceCase {
            name: "integer_column_gt_text_literal",
            sql: "SELECT id FROM users WHERE id > '1' ORDER BY id ASC",
        },
        DivergenceCase {
            name: "integer_column_ne_text_literal",
            sql: "SELECT id FROM users WHERE id != '2' ORDER BY id ASC",
        },
        DivergenceCase {
            name: "integer_column_between_text_bounds",
            sql: "SELECT id FROM users WHERE id BETWEEN '1' AND '2' ORDER BY id ASC",
        },
        DivergenceCase {
            name: "integer_column_in_text_list",
            sql: "SELECT id FROM users WHERE id IN ('1', '3') ORDER BY id ASC",
        },
    ];

    for case in matrix {
        let miniql_rows =
            collect_miniql_rows(&db, case.sql, 8).expect("execute miniql divergence sql");
        let sqlite_rows =
            collect_sqlite_rows(&conn, case.sql).expect("execute sqlite divergence sql");
        assert_ne!(
            miniql_rows, sqlite_rows,
            "edge divergence case unexpectedly matched sqlite; move to parity matrix if behavior \
             is now aligned\ncase: {}\nsql: {}\nminiql: {miniql_rows:?}\nsqlite: {sqlite_rows:?}",
            case.name, case.sql
        );
    }
}
