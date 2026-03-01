mod util;

use miniql::table::QueryError;
use miniql::{Db, Error as TableError, Row, ScanScratch, ValueRef};
use rusqlite::params;

fn text_col(row: &Row<'_>, idx: usize, label: &str) -> String {
    match row.get(idx) {
        Some(ValueRef::Text(bytes)) => String::from_utf8_lossy(bytes).to_string(),
        other => panic!("unexpected {label} value: {other:?}"),
    }
}

fn i64_col(row: &Row<'_>, idx: usize, label: &str) -> i64 {
    match row.get(idx) {
        Some(ValueRef::Integer(value)) => value,
        other => panic!("unexpected {label} value: {other:?}"),
    }
}

fn opt_i64_col(row: &Row<'_>, idx: usize, label: &str) -> Option<i64> {
    match row.get(idx) {
        Some(ValueRef::Integer(value)) => Some(value),
        Some(ValueRef::Null) | None => None,
        other => panic!("unexpected {label} value: {other:?}"),
    }
}

#[test]
fn executes_select_with_where_order_limit_offset() {
    let file = util::make_db(|conn| {
        conn.execute_batch("CREATE TABLE people (name TEXT, age INTEGER, city TEXT);").unwrap();
        let rows = [
            ("alice", 30, "austin"),
            ("bob", 25, "boston"),
            ("carol", 25, "chicago"),
            ("dave", 40, "denver"),
            ("eve", 35, "elpaso"),
        ];
        for (name, age, city) in rows {
            conn.execute(
                "INSERT INTO people (name, age, city) VALUES (?1, ?2, ?3)",
                params![name, age, city],
            )
            .unwrap();
        }
    });

    let db = Db::open(file.path()).expect("open db");
    let mut scratch = ScanScratch::with_capacity(3, 0);
    let mut seen = Vec::new();

    db.query(
        "SELECT name, age FROM people WHERE age >= 25 ORDER BY age DESC, name ASC LIMIT 2 OFFSET 1",
        &mut scratch,
        |row| {
            let name = text_col(&row, 0, "name");
            let age = i64_col(&row, 1, "age");
            seen.push((name, age));
            Ok(())
        },
    )
    .expect("execute sql");

    assert_eq!(seen, vec![("eve".to_string(), 35), ("alice".to_string(), 30)]);
}

#[test]
fn executes_prepared_scan_query_multiple_times() {
    let file = util::make_db(|conn| {
        conn.execute_batch("CREATE TABLE people (name TEXT, age INTEGER);").unwrap();
        let rows = [("alice", 30_i64), ("bob", 25), ("carol", 25), ("dave", 40), ("eve", 35)];
        for (name, age) in rows {
            conn.execute("INSERT INTO people (name, age) VALUES (?1, ?2)", params![name, age])
                .unwrap();
        }
    });

    let db = Db::open(file.path()).expect("open db");
    let mut stmt = db
        .prepare(
            "SELECT name, age FROM people WHERE age >= 25 ORDER BY age DESC, name ASC LIMIT 2 \
             OFFSET 1",
        )
        .expect("prepare sql");
    let mut scratch = ScanScratch::with_capacity(3, 0);

    let mut first = Vec::new();
    stmt.for_each(&mut scratch, |row| {
        let name = text_col(&row, 0, "name");
        let age = i64_col(&row, 1, "age");
        first.push((name, age));
        Ok(())
    })
    .expect("execute prepared sql first pass");

    let mut second = Vec::new();
    stmt.for_each(&mut scratch, |row| {
        let name = text_col(&row, 0, "name");
        let age = i64_col(&row, 1, "age");
        second.push((name, age));
        Ok(())
    })
    .expect("execute prepared sql second pass");

    assert_eq!(first, vec![("eve".to_string(), 35), ("alice".to_string(), 30)]);
    assert_eq!(second, first);
}

#[test]
fn executes_prepared_join_query_multiple_times() {
    let file = util::make_db(|conn| {
        conn.execute_batch(
            "CREATE TABLE users (id INTEGER, name TEXT);
             CREATE TABLE orders (user_id INTEGER, amount INTEGER);",
        )
        .unwrap();
        let users = [(1_i64, "alice"), (2, "bob"), (3, "cara"), (4, "dina")];
        for (id, name) in users {
            conn.execute("INSERT INTO users (id, name) VALUES (?1, ?2)", params![id, name])
                .unwrap();
        }
        let orders = [(1_i64, 100_i64), (1, 40), (2, 75)];
        for (user_id, amount) in orders {
            conn.execute(
                "INSERT INTO orders (user_id, amount) VALUES (?1, ?2)",
                params![user_id, amount],
            )
            .unwrap();
        }
    });

    let db = Db::open(file.path()).expect("open db");
    let mut stmt = db
        .prepare(
            "SELECT u.name, o.amount
             FROM users AS u
             LEFT JOIN orders AS o ON u.id = o.user_id
             WHERE o.amount IS NULL OR o.amount > 50
             ORDER BY u.name ASC, o.amount ASC",
        )
        .expect("prepare join sql");
    let mut scratch = ScanScratch::with_capacity(4, 0);

    let mut first = Vec::new();
    stmt.for_each(&mut scratch, |row| {
        let name = text_col(&row, 0, "name");
        let amount = opt_i64_col(&row, 1, "amount");
        first.push((name, amount));
        Ok(())
    })
    .expect("execute prepared join first pass");

    let mut second = Vec::new();
    stmt.for_each(&mut scratch, |row| {
        let name = text_col(&row, 0, "name");
        let amount = opt_i64_col(&row, 1, "amount");
        second.push((name, amount));
        Ok(())
    })
    .expect("execute prepared join second pass");

    assert_eq!(
        first,
        vec![
            ("alice".to_string(), Some(100)),
            ("bob".to_string(), Some(75)),
            ("cara".to_string(), None),
            ("dina".to_string(), None)
        ]
    );
    assert_eq!(second, first);
}

#[test]
fn prepare_rejects_non_select_statement() {
    let file = util::make_db(|conn| {
        conn.execute_batch("CREATE TABLE t (id INTEGER);").unwrap();
    });
    let db = Db::open(file.path()).expect("open db");
    let err = match db.prepare("UPDATE t SET id = 1") {
        Ok(_) => panic!("expected unsupported SQL"),
        Err(err) => err,
    };
    assert!(matches!(err, TableError::Query(QueryError::SqlUnsupported)));
}

#[test]
fn executes_where_in_between_and_like() {
    let file = util::make_db(|conn| {
        conn.execute_batch("CREATE TABLE people (name TEXT, age INTEGER, city TEXT);").unwrap();
        let rows = [
            ("alice", 30_i64, Some("austin")),
            ("anna", 27, Some("austin")),
            ("bob", 25, Some("boston")),
            ("carol", 25, Some("chicago")),
            ("dave", 40, Some("denver")),
            ("amelia", 29, None),
        ];
        for (name, age, city) in rows {
            conn.execute(
                "INSERT INTO people (name, age, city) VALUES (?1, ?2, ?3)",
                params![name, age, city],
            )
            .unwrap();
        }
    });

    let db = Db::open(file.path()).expect("open db");
    let mut scratch = ScanScratch::with_capacity(3, 0);
    let mut seen = Vec::new();

    db.query(
        "SELECT name
         FROM people
         WHERE city IN ('austin', 'boston')
           AND age BETWEEN 25 AND 30
           AND name LIKE 'a%'
         ORDER BY name ASC",
        &mut scratch,
        |row| {
            let name = text_col(&row, 0, "name");
            seen.push(name);
            Ok(())
        },
    )
    .expect("execute where in/between/like sql");

    assert_eq!(seen, vec!["alice".to_string(), "anna".to_string()]);
}

#[test]
fn executes_where_not_in_not_between_and_not_like() {
    let file = util::make_db(|conn| {
        conn.execute_batch("CREATE TABLE vals (v INTEGER, s TEXT);").unwrap();
        let rows = [(Some(1_i64), "alpha"), (Some(2), "beta"), (Some(3), "able"), (None, "gamma")];
        for (v, s) in rows {
            conn.execute("INSERT INTO vals (v, s) VALUES (?1, ?2)", params![v, s]).unwrap();
        }
    });

    let db = Db::open(file.path()).expect("open db");
    let mut scratch = ScanScratch::with_capacity(2, 0);
    let mut seen = Vec::new();

    db.query(
        "SELECT v, s
         FROM vals
         WHERE v NOT IN (1, 3)
           AND v NOT BETWEEN 5 AND 10
           AND s NOT LIKE 'a%'
         ORDER BY v ASC",
        &mut scratch,
        |row| {
            let v = i64_col(&row, 0, "v");
            let s = text_col(&row, 1, "s");
            seen.push((v, s));
            Ok(())
        },
    )
    .expect("execute where not in/not between/not like sql");

    assert_eq!(seen, vec![(2_i64, "beta".to_string())]);
}

#[test]
fn executes_group_by_aggregate_query() {
    let file = util::make_db(|conn| {
        conn.execute_batch("CREATE TABLE items (grp TEXT, val INTEGER);").unwrap();
        let rows = [("a", 1_i64), ("a", 2), ("b", 3)];
        for (grp, val) in rows {
            conn.execute("INSERT INTO items (grp, val) VALUES (?1, ?2)", params![grp, val])
                .unwrap();
        }
    });

    let db = Db::open(file.path()).expect("open db");
    let mut scratch = ScanScratch::with_capacity(2, 0);
    let mut seen = Vec::new();

    db.query("SELECT grp, SUM(val), COUNT(*) FROM items GROUP BY grp", &mut scratch, |row| {
        let grp = text_col(&row, 0, "grp");
        let sum = i64_col(&row, 1, "sum");
        let count = i64_col(&row, 2, "count");
        seen.push((grp, sum, count));
        Ok(())
    })
    .expect("execute aggregate sql");

    seen.sort();
    assert_eq!(seen, vec![("a".to_string(), 3, 2), ("b".to_string(), 3, 1)]);
}

#[test]
fn executes_select_distinct_with_order_limit_offset() {
    let file = util::make_db(|conn| {
        conn.execute_batch("CREATE TABLE places (city TEXT);").unwrap();
        let rows = ["chicago", "austin", "boston", "austin", "boston"];
        for city in rows {
            conn.execute("INSERT INTO places (city) VALUES (?1)", params![city]).unwrap();
        }
    });

    let db = Db::open(file.path()).expect("open db");
    let mut scratch = ScanScratch::with_capacity(1, 0);
    let mut seen = Vec::new();

    db.query(
        "SELECT DISTINCT city AS c FROM places ORDER BY c ASC LIMIT 2 OFFSET 1",
        &mut scratch,
        |row| {
            let city = text_col(&row, 0, "city");
            seen.push(city);
            Ok(())
        },
    )
    .expect("execute distinct sql");

    assert_eq!(seen, vec!["boston".to_string(), "chicago".to_string()]);
}

#[test]
fn executes_group_by_having_with_alias_and_hidden_aggregate() {
    let file = util::make_db(|conn| {
        conn.execute_batch("CREATE TABLE items (grp TEXT, val INTEGER);").unwrap();
        let rows = [("a", 1_i64), ("a", 2), ("b", 3), ("c", 2), ("c", 2)];
        for (grp, val) in rows {
            conn.execute("INSERT INTO items (grp, val) VALUES (?1, ?2)", params![grp, val])
                .unwrap();
        }
    });

    let db = Db::open(file.path()).expect("open db");
    let mut scratch = ScanScratch::with_capacity(2, 0);
    let mut seen = Vec::new();

    db.query(
        "SELECT grp, SUM(val) AS total FROM items GROUP BY grp HAVING COUNT(*) >= 2 AND total > 2",
        &mut scratch,
        |row| {
            let grp = text_col(&row, 0, "grp");
            let total = i64_col(&row, 1, "total");
            seen.push((grp, total));
            Ok(())
        },
    )
    .expect("execute having sql");

    seen.sort();
    assert_eq!(seen, vec![("a".to_string(), 3), ("c".to_string(), 4)]);
}

#[test]
fn executes_group_by_having_on_non_projected_group_key() {
    let file = util::make_db(|conn| {
        conn.execute_batch("CREATE TABLE items (grp TEXT, val INTEGER);").unwrap();
        let rows = [("a", 1_i64), ("a", 2), ("b", 3)];
        for (grp, val) in rows {
            conn.execute("INSERT INTO items (grp, val) VALUES (?1, ?2)", params![grp, val])
                .unwrap();
        }
    });

    let db = Db::open(file.path()).expect("open db");
    let mut scratch = ScanScratch::with_capacity(2, 0);
    let mut seen = Vec::new();

    db.query("SELECT SUM(val) FROM items GROUP BY grp HAVING grp = 'a'", &mut scratch, |row| {
        let total = i64_col(&row, 0, "total");
        seen.push(total);
        Ok(())
    })
    .expect("execute having query with hidden group key");

    assert_eq!(seen, vec![3]);
}

#[test]
fn executes_group_by_order_by_alias_limit_offset() {
    let file = util::make_db(|conn| {
        conn.execute_batch("CREATE TABLE items (grp TEXT, val INTEGER);").unwrap();
        let rows = [("a", 2_i64), ("a", 1), ("b", 3), ("c", 2), ("c", 2)];
        for (grp, val) in rows {
            conn.execute("INSERT INTO items (grp, val) VALUES (?1, ?2)", params![grp, val])
                .unwrap();
        }
    });

    let db = Db::open(file.path()).expect("open db");
    let mut scratch = ScanScratch::with_capacity(2, 0);
    let mut seen = Vec::new();

    db.query(
        "SELECT grp, SUM(val) AS total FROM items GROUP BY grp ORDER BY total DESC, grp ASC LIMIT \
         2 OFFSET 1",
        &mut scratch,
        |row| {
            let grp = text_col(&row, 0, "grp");
            let total = i64_col(&row, 1, "total");
            seen.push((grp, total));
            Ok(())
        },
    )
    .expect("execute aggregate order by alias sql");

    assert_eq!(seen, vec![("a".to_string(), 3), ("b".to_string(), 3)]);
}

#[test]
fn executes_group_by_order_by_hidden_aggregate_expression() {
    let file = util::make_db(|conn| {
        conn.execute_batch("CREATE TABLE items (grp TEXT, val INTEGER);").unwrap();
        let rows = [("a", 1_i64), ("a", 2), ("b", 5), ("c", 2), ("c", 2)];
        for (grp, val) in rows {
            conn.execute("INSERT INTO items (grp, val) VALUES (?1, ?2)", params![grp, val])
                .unwrap();
        }
    });

    let db = Db::open(file.path()).expect("open db");
    let mut scratch = ScanScratch::with_capacity(2, 0);
    let mut seen = Vec::new();

    db.query(
        "SELECT grp FROM items GROUP BY grp ORDER BY SUM(val) DESC, grp ASC",
        &mut scratch,
        |row| {
            let grp = text_col(&row, 0, "grp");
            seen.push(grp);
            Ok(())
        },
    )
    .expect("execute aggregate order by hidden expression");

    assert_eq!(seen, vec!["b".to_string(), "c".to_string(), "a".to_string()]);
}

#[test]
fn executes_inner_join_with_projection_and_order_by() {
    let file = util::make_db(|conn| {
        conn.execute_batch(
            "CREATE TABLE users (id INTEGER, name TEXT);
             CREATE TABLE orders (user_id INTEGER, amount INTEGER);",
        )
        .unwrap();
        let users = [(1_i64, "alice"), (2, "bob"), (3, "cara")];
        for (id, name) in users {
            conn.execute("INSERT INTO users (id, name) VALUES (?1, ?2)", params![id, name])
                .unwrap();
        }
        let orders = [(1_i64, 100_i64), (1, 120), (2, 75), (9, 10)];
        for (user_id, amount) in orders {
            conn.execute(
                "INSERT INTO orders (user_id, amount) VALUES (?1, ?2)",
                params![user_id, amount],
            )
            .unwrap();
        }
    });

    let db = Db::open(file.path()).expect("open db");
    let mut scratch = ScanScratch::with_capacity(4, 0);
    let mut seen = Vec::new();

    db.query(
        "SELECT u.name, o.amount
         FROM users AS u
         INNER JOIN orders AS o ON u.id = o.user_id
         ORDER BY u.name ASC, o.amount DESC",
        &mut scratch,
        |row| {
            let name = text_col(&row, 0, "name");
            let amount = i64_col(&row, 1, "amount");
            seen.push((name, amount));
            Ok(())
        },
    )
    .expect("execute inner join");

    assert_eq!(
        seen,
        vec![("alice".to_string(), 120), ("alice".to_string(), 100), ("bob".to_string(), 75)]
    );
}

#[test]
fn executes_left_join_with_where_limit_offset() {
    let file = util::make_db(|conn| {
        conn.execute_batch(
            "CREATE TABLE users (id INTEGER, name TEXT);
             CREATE TABLE orders (user_id INTEGER, amount INTEGER);",
        )
        .unwrap();
        let users = [(1_i64, "alice"), (2, "bob"), (3, "cara"), (4, "dina")];
        for (id, name) in users {
            conn.execute("INSERT INTO users (id, name) VALUES (?1, ?2)", params![id, name])
                .unwrap();
        }
        let orders = [(1_i64, 100_i64), (1, 40), (2, 75)];
        for (user_id, amount) in orders {
            conn.execute(
                "INSERT INTO orders (user_id, amount) VALUES (?1, ?2)",
                params![user_id, amount],
            )
            .unwrap();
        }
    });

    let db = Db::open(file.path()).expect("open db");
    let mut scratch = ScanScratch::with_capacity(4, 0);
    let mut seen = Vec::new();

    db.query(
        "SELECT u.name, o.amount
         FROM users AS u
         LEFT JOIN orders AS o ON u.id = o.user_id
         WHERE o.amount >= 40
         ORDER BY o.amount DESC, u.name ASC
         LIMIT 2 OFFSET 1",
        &mut scratch,
        |row| {
            let name = text_col(&row, 0, "name");
            let amount = i64_col(&row, 1, "amount");
            seen.push((name, amount));
            Ok(())
        },
    )
    .expect("execute left join");

    assert_eq!(seen, vec![("bob".to_string(), 75), ("alice".to_string(), 40)]);
}

#[test]
fn executes_left_join_distinct_projection_with_order_by_ordinal() {
    let file = util::make_db(|conn| {
        conn.execute_batch(
            "CREATE TABLE users (id INTEGER, name TEXT);
             CREATE TABLE orders (user_id INTEGER, amount INTEGER);",
        )
        .unwrap();
        let users = [(1_i64, "alice"), (2, "bob"), (3, "cara"), (4, "dina")];
        for (id, name) in users {
            conn.execute("INSERT INTO users (id, name) VALUES (?1, ?2)", params![id, name])
                .unwrap();
        }
        let orders = [(1_i64, 100_i64), (1, 40), (2, 75), (9, 10)];
        for (user_id, amount) in orders {
            conn.execute(
                "INSERT INTO orders (user_id, amount) VALUES (?1, ?2)",
                params![user_id, amount],
            )
            .unwrap();
        }
    });

    let db = Db::open(file.path()).expect("open db");
    let mut scratch = ScanScratch::with_capacity(4, 0);
    let mut seen = Vec::new();

    db.query(
        "SELECT DISTINCT u.name
         FROM users AS u
         LEFT JOIN orders AS o ON u.id = o.user_id
         ORDER BY 1 ASC",
        &mut scratch,
        |row| {
            seen.push(text_col(&row, 0, "name"));
            Ok(())
        },
    )
    .expect("execute distinct left join");

    assert_eq!(
        seen,
        vec!["alice".to_string(), "bob".to_string(), "cara".to_string(), "dina".to_string()]
    );
}

#[test]
fn limit_zero_short_circuits_scan_and_join_paths() {
    let file = util::make_db(|conn| {
        conn.execute_batch(
            "CREATE TABLE users (id INTEGER, name TEXT);
             CREATE TABLE orders (user_id INTEGER, amount INTEGER);
             INSERT INTO users (id, name) VALUES (1, 'alice'), (2, 'bob');
             INSERT INTO orders (user_id, amount) VALUES (1, 100), (1, 40), (2, 75);",
        )
        .unwrap();
    });

    let db = Db::open(file.path()).expect("open db");
    let mut scratch = ScanScratch::with_capacity(4, 0);

    let mut scan_rows = 0usize;
    db.query("SELECT name FROM users ORDER BY name ASC LIMIT 0 OFFSET 1000", &mut scratch, |_| {
        scan_rows += 1;
        Ok(())
    })
    .expect("execute limit 0 scan");
    assert_eq!(scan_rows, 0);

    let mut join_rows = 0usize;
    db.query(
        "SELECT u.name, o.amount
         FROM users AS u
         LEFT JOIN orders AS o ON u.id = o.user_id
         ORDER BY u.name ASC, o.amount DESC
         LIMIT 0 OFFSET 1000",
        &mut scratch,
        |_| {
            join_rows += 1;
            Ok(())
        },
    )
    .expect("execute limit 0 join");
    assert_eq!(join_rows, 0);
}

#[test]
fn executes_join_where_with_in_between_and_like() {
    let file = util::make_db(|conn| {
        conn.execute_batch(
            "CREATE TABLE users (id INTEGER, name TEXT);
             CREATE TABLE orders (user_id INTEGER, amount INTEGER, note TEXT);",
        )
        .unwrap();
        let users = [(1_i64, "alice"), (2, "bob"), (3, "cara"), (4, "dina")];
        for (id, name) in users {
            conn.execute("INSERT INTO users (id, name) VALUES (?1, ?2)", params![id, name])
                .unwrap();
        }
        let orders =
            [(1_i64, 20_i64, "x0"), (1, 40, "o1"), (2, 75, "o2"), (4, 100, "p4"), (4, 100, "o4")];
        for (user_id, amount, note) in orders {
            conn.execute(
                "INSERT INTO orders (user_id, amount, note) VALUES (?1, ?2, ?3)",
                params![user_id, amount, note],
            )
            .unwrap();
        }
    });

    let db = Db::open(file.path()).expect("open db");
    let mut scratch = ScanScratch::with_capacity(4, 0);
    let mut seen = Vec::new();

    db.query(
        "SELECT u.name, o.note
         FROM users AS u
         LEFT JOIN orders AS o ON u.id = o.user_id
         WHERE u.id IN (1, 2, 4)
           AND o.amount BETWEEN 40 AND 100
           AND o.note LIKE 'o_'
         ORDER BY u.id ASC, o.amount ASC",
        &mut scratch,
        |row| {
            let name = text_col(&row, 0, "name");
            let note = text_col(&row, 1, "note");
            seen.push((name, note));
            Ok(())
        },
    )
    .expect("execute join where in/between/like sql");

    assert_eq!(
        seen,
        vec![
            ("alice".to_string(), "o1".to_string()),
            ("bob".to_string(), "o2".to_string()),
            ("dina".to_string(), "o4".to_string()),
        ]
    );
}

#[test]
fn rejects_unsupported_sql_join() {
    let file = util::make_db(|conn| {
        conn.execute_batch("CREATE TABLE t (id INTEGER); INSERT INTO t (id) VALUES (1), (2);")
            .unwrap();
    });

    let db = Db::open(file.path()).expect("open db");
    let mut scratch = ScanScratch::with_capacity(1, 0);
    let err = db
        .query("SELECT * FROM t AS a RIGHT JOIN t AS b ON a.id = b.id", &mut scratch, |_| Ok(()))
        .expect_err("expected unsupported SQL");

    assert!(matches!(err, TableError::Query(QueryError::SqlUnsupported)));
}
