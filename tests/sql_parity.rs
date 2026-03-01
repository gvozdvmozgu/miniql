mod util;

use miniql::Db;
use rusqlite::{Connection, params};
use util::{collect_miniql_rows, collect_sqlite_rows};

#[derive(Debug, Clone, Copy)]
struct ParityCase {
    name: &'static str,
    sql: &'static str,
}

#[derive(Debug)]
struct ParityMatrix<'a> {
    name: &'static str,
    cases: &'a [ParityCase],
}

#[test]
fn supported_sql_matches_sqlite_rows_and_order() {
    let file = util::make_db(|conn| {
        conn.execute_batch(
            "CREATE TABLE users (id INTEGER, name TEXT, city TEXT);
             CREATE TABLE orders (user_id INTEGER, amount INTEGER, note TEXT);
             CREATE TABLE items (grp TEXT, val INTEGER);
             CREATE TABLE places (city TEXT);
             CREATE TABLE metrics (grp TEXT, score REAL, weight INTEGER);
             CREATE TABLE edge_values (id INTEGER, v, label TEXT);",
        )
        .expect("create parity tables");

        let users = [
            (1_i64, "alice", Some("austin")),
            (2, "bob", Some("boston")),
            (3, "cara", None),
            (4, "dina", Some("denver")),
        ];
        for (id, name, city) in users {
            conn.execute(
                "INSERT INTO users (id, name, city) VALUES (?1, ?2, ?3)",
                params![id, name, city],
            )
            .expect("insert users");
        }

        let orders = [
            (Some(1_i64), 100_i64, "o1"),
            (Some(1), 40, "o2"),
            (Some(2), 75, "o3"),
            (None, 10, "orphan_null"),
            (Some(9), 5, "orphan_missing"),
        ];
        for (user_id, amount, note) in orders {
            conn.execute(
                "INSERT INTO orders (user_id, amount, note) VALUES (?1, ?2, ?3)",
                params![user_id, amount, note],
            )
            .expect("insert orders");
        }

        let items = [
            (Some("a"), Some(1_i64)),
            (Some("a"), Some(2)),
            (Some("b"), Some(3)),
            (Some("c"), Some(2)),
            (Some("c"), Some(2)),
            (None, Some(5)),
            (None, None),
        ];
        for (grp, val) in items {
            conn.execute("INSERT INTO items (grp, val) VALUES (?1, ?2)", params![grp, val])
                .expect("insert items");
        }

        let places = [Some("chicago"), Some("austin"), Some("boston"), Some("austin"), None];
        for city in places {
            conn.execute("INSERT INTO places (city) VALUES (?1)", params![city])
                .expect("insert places");
        }

        let metrics = [("x", 1.5_f64, 1_i64), ("x", 2.25, 2), ("y", 9.5, 3), ("z", -0.0, 4)];
        for (grp, score, weight) in metrics {
            conn.execute(
                "INSERT INTO metrics (grp, score, weight) VALUES (?1, ?2, ?3)",
                params![grp, score, weight],
            )
            .expect("insert metrics");
        }

        conn.execute_batch(
            "INSERT INTO edge_values (id, v, label) VALUES
               (1, NULL, NULL),
               (2, -1, 'beta'),
               (3, 2, 'Alpha'),
               (4, 2.0, 'alpha'),
               (5, '2', 'ALPHA'),
               (6, '10', 'zeta'),
               (7, 'aardvark', 'aardvark'),
               (8, X'6162', 'blob');",
        )
        .expect("insert edge values");
    });

    let db = Db::open(file.path()).expect("open miniql db");
    let conn = Connection::open(file.path()).expect("open sqlite db");

    let core_cases = [
        ParityCase {
            name: "scan_where_order_limit",
            sql: "SELECT name, city FROM users WHERE id >= 2 ORDER BY id DESC LIMIT 2",
        },
        ParityCase {
            name: "scan_wildcard_order",
            sql: "SELECT * FROM users ORDER BY id ASC LIMIT 3",
        },
        ParityCase {
            name: "scan_qualified_wildcard",
            sql: "SELECT u.* FROM users AS u ORDER BY u.id DESC LIMIT 2",
        },
        ParityCase {
            name: "scan_order_by_alias",
            sql: "SELECT name AS n, city AS c FROM users WHERE city IS NOT NULL ORDER BY n DESC \
                  LIMIT 2",
        },
        ParityCase {
            name: "scan_limit_offset_comma_syntax",
            sql: "SELECT name FROM users ORDER BY id ASC LIMIT 1, 2",
        },
        ParityCase {
            name: "scan_not_nested_where",
            sql: "SELECT name FROM users WHERE NOT (city IS NULL) ORDER BY id ASC",
        },
        ParityCase {
            name: "distinct_alias_limit_offset",
            sql: "SELECT DISTINCT city AS c FROM places ORDER BY c ASC LIMIT 3 OFFSET 1",
        },
        ParityCase {
            name: "aggregate_group_by_having_order",
            sql: "SELECT grp, SUM(val) AS total, COUNT(*) AS n FROM items GROUP BY grp HAVING \
                  COUNT(*) >= 1 ORDER BY total DESC, grp ASC",
        },
        ParityCase {
            name: "aggregate_hidden_order_expr",
            sql: "SELECT grp FROM items GROUP BY grp ORDER BY SUM(val) DESC, grp ASC LIMIT 3",
        },
        ParityCase {
            name: "aggregate_no_group_functions",
            sql: "SELECT COUNT(*), COUNT(val), SUM(val), AVG(val), MIN(val), MAX(val) FROM items",
        },
        ParityCase {
            name: "aggregate_filtered_no_group",
            sql: "SELECT COUNT(*), SUM(val) FROM items WHERE grp = 'a'",
        },
        ParityCase {
            name: "aggregate_having_non_projected_group_key",
            sql: "SELECT SUM(val) FROM items GROUP BY grp HAVING grp = 'a' OR grp IS NULL ORDER \
                  BY 1 DESC",
        },
        ParityCase {
            name: "aggregate_real_avg",
            sql: "SELECT grp, AVG(score), MAX(weight) FROM metrics GROUP BY grp ORDER BY grp ASC",
        },
        ParityCase {
            name: "inner_join_where_order",
            sql: "SELECT u.name, o.amount FROM users AS u INNER JOIN orders AS o ON u.id = \
                  o.user_id WHERE o.amount >= 40 ORDER BY u.name ASC, o.amount DESC",
        },
        ParityCase {
            name: "left_join_with_null_predicate",
            sql: "SELECT u.name, o.amount FROM users AS u LEFT JOIN orders AS o ON u.id = \
                  o.user_id WHERE o.amount IS NULL OR o.amount > 50 ORDER BY u.name ASC, o.amount \
                  ASC",
        },
        ParityCase {
            name: "join_alias_projection_order_alias",
            sql: "SELECT u.name AS uname, o.amount AS amt FROM users AS u INNER JOIN orders AS o \
                  ON u.id = o.user_id ORDER BY uname ASC, amt DESC",
        },
        ParityCase {
            name: "join_projection_with_wildcard",
            sql: "SELECT u.*, o.amount FROM users AS u INNER JOIN orders AS o ON u.id = o.user_id \
                  ORDER BY u.id ASC, o.amount ASC LIMIT 3",
        },
        ParityCase {
            name: "join_distinct_order_ordinal",
            sql: "SELECT DISTINCT u.name FROM users AS u LEFT JOIN orders AS o ON u.id = \
                  o.user_id ORDER BY 1 ASC",
        },
    ];

    let edge_null_order_cases = [
        ParityCase {
            name: "null_order_value_asc",
            sql: "SELECT id FROM edge_values ORDER BY v ASC, id ASC",
        },
        ParityCase {
            name: "null_order_value_desc",
            sql: "SELECT id FROM edge_values ORDER BY v DESC, id ASC",
        },
        ParityCase {
            name: "null_order_text_asc",
            sql: "SELECT id FROM edge_values ORDER BY label ASC, id ASC",
        },
        ParityCase {
            name: "null_order_text_desc",
            sql: "SELECT id FROM edge_values ORDER BY label DESC, id ASC",
        },
    ];

    let edge_numeric_text_cases = [
        ParityCase {
            name: "scan_where_in_list",
            sql: "SELECT name FROM users WHERE id IN (1, 3, 4) ORDER BY id ASC",
        },
        ParityCase {
            name: "scan_where_between",
            sql: "SELECT name FROM users WHERE id BETWEEN 2 AND 3 ORDER BY id ASC",
        },
        ParityCase {
            name: "numeric_text_eq_with_mixed_storage",
            sql: "SELECT id FROM edge_values WHERE v = 2 ORDER BY id ASC",
        },
        ParityCase {
            name: "numeric_text_between_with_mixed_storage",
            sql: "SELECT id FROM edge_values WHERE v BETWEEN 0 AND 3 ORDER BY id ASC",
        },
        ParityCase {
            name: "numeric_text_in_with_mixed_storage",
            sql: "SELECT id FROM edge_values WHERE v IN (-1, 2) ORDER BY id ASC",
        },
        ParityCase {
            name: "numeric_text_lt_with_mixed_storage",
            sql: "SELECT id FROM edge_values WHERE v < 3 ORDER BY id ASC",
        },
    ];

    let edge_collation_like_cases = [
        ParityCase {
            name: "scan_where_like",
            sql: "SELECT name FROM users WHERE name LIKE 'a%' OR name LIKE '_ob' ORDER BY id ASC",
        },
        ParityCase {
            name: "join_where_in_between_like",
            sql: "SELECT u.name, o.note FROM users AS u LEFT JOIN orders AS o ON u.id = o.user_id \
                  WHERE u.id IN (1, 2, 4) AND o.amount BETWEEN 40 AND 100 AND o.note LIKE 'o_' \
                  ORDER BY u.id ASC, o.amount ASC",
        },
        ParityCase {
            name: "like_ascii_case_insensitive_prefix",
            sql: "SELECT id FROM edge_values WHERE label LIKE 'a%' ORDER BY id ASC",
        },
        ParityCase {
            name: "like_single_char_wildcard",
            sql: "SELECT id FROM edge_values WHERE label LIKE '_lpha' ORDER BY id ASC",
        },
        ParityCase {
            name: "binary_text_order_case_sensitive",
            sql: "SELECT label FROM edge_values WHERE label IS NOT NULL ORDER BY label ASC, id ASC",
        },
    ];

    let matrices = [
        ParityMatrix { name: "core_supported", cases: &core_cases },
        ParityMatrix { name: "edge_null_ordering", cases: &edge_null_order_cases },
        ParityMatrix { name: "edge_numeric_text_comparison", cases: &edge_numeric_text_cases },
        ParityMatrix { name: "edge_collation_like", cases: &edge_collation_like_cases },
    ];

    for matrix in matrices {
        for case in matrix.cases {
            let miniql_rows = collect_miniql_rows(&db, case.sql, 16).expect("execute miniql query");
            let sqlite_rows = collect_sqlite_rows(&conn, case.sql).expect("execute sqlite query");
            assert_eq!(
                miniql_rows, sqlite_rows,
                "query parity mismatch\nmatrix: {}\ncase: {}\nquery: {}\nminiql: \
                 {miniql_rows:?}\nsqlite: {sqlite_rows:?}",
                matrix.name, case.name, case.sql,
            );
        }
    }
}
