use codspeed_criterion_compat::{
    BenchmarkId, Criterion, black_box, criterion_group, criterion_main,
};
use miniql::{Db, Row, ScanScratch, ValueRef};
use rusqlite::types::ValueRef as SqliteValueRef;
use rusqlite::{Connection, Row as SqliteRow, params};
use tempfile::NamedTempFile;

const ROW_COUNTS: &[usize] = &[2_000, 20_000];

const SCAN_SQL: &str = "SELECT id, score FROM users WHERE score >= 100 ORDER BY score DESC, id \
                        ASC LIMIT 200 OFFSET 50";
const AGG_SQL: &str = "SELECT grp, SUM(val) AS total, COUNT(*) AS n FROM items GROUP BY grp \
                       HAVING COUNT(*) >= 2 ORDER BY total DESC, grp ASC LIMIT 128";
const AGG_HIDDEN_ORDER_SQL: &str =
    "SELECT grp FROM items GROUP BY grp ORDER BY SUM(val) DESC, grp ASC LIMIT 128";
const JOIN_SQL: &str = "SELECT u.id, o.amount FROM users AS u INNER JOIN orders AS o ON u.id = \
                        o.user_id WHERE o.amount >= 200 ORDER BY u.id ASC, o.amount DESC LIMIT \
                        2000";
const LEFT_JOIN_SQL: &str = "SELECT u.id, o.amount FROM users AS u LEFT JOIN orders AS o ON u.id \
                             = o.user_id WHERE o.amount IS NULL OR o.amount >= 200 ORDER BY u.id \
                             ASC, o.amount DESC LIMIT 2000";
const DISTINCT_SQL: &str = "SELECT DISTINCT city FROM places ORDER BY city ASC";
const DISTINCT_HIGH_CARD_SQL: &str = "SELECT DISTINCT name FROM users ORDER BY name ASC";

fn create_sql_bench_db(row_count: usize) -> NamedTempFile {
    let file = NamedTempFile::new().expect("create temp file");
    let conn = Connection::open(file.path()).expect("open connection");

    conn.execute_batch(
        "
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        CREATE TABLE users(id INTEGER, name TEXT, city TEXT, score INTEGER);
        CREATE TABLE orders(user_id INTEGER, amount INTEGER, note TEXT);
        CREATE INDEX orders_user_id_idx ON orders(user_id);
        CREATE TABLE items(grp TEXT, val INTEGER);
        CREATE TABLE places(city TEXT);
        ",
    )
    .expect("create schema");

    conn.execute_batch("BEGIN").expect("begin txn");

    {
        let mut users = conn
            .prepare("INSERT INTO users(id, name, city, score) VALUES (?1, ?2, ?3, ?4)")
            .expect("prepare users insert");
        for i in 0..row_count {
            let id = (i as i64) + 1;
            let name = format!("user_{i:05}");
            let city = format!("city_{:03}", i % 128);
            let score = ((i * 31) % 1_000) as i64;
            users.execute(params![id, name, city, score]).expect("insert user");
        }
    }

    {
        let matched_users = (row_count * 3 / 4).max(1);
        let mut orders = conn
            .prepare("INSERT INTO orders(user_id, amount, note) VALUES (?1, ?2, ?3)")
            .expect("prepare orders insert");
        for i in 0..(row_count * 2) {
            let user_id = if i % 11 == 0 {
                row_count as i64 + (i % 32) as i64 + 1
            } else {
                ((i % matched_users) as i64) + 1
            };
            let amount = ((i * 17) % 500) as i64;
            let note = format!("order_{i:05}");
            orders.execute(params![user_id, amount, note]).expect("insert order");
        }
    }

    {
        let groups = (row_count / 20).max(1);
        let mut items = conn
            .prepare("INSERT INTO items(grp, val) VALUES (?1, ?2)")
            .expect("prepare items insert");
        for i in 0..(row_count * 2) {
            let grp = format!("g{:04}", i % groups);
            let val = if i % 9 == 0 { None } else { Some((i % 1_000) as i64) };
            items.execute(params![grp, val]).expect("insert item");
        }
    }

    {
        let mut places =
            conn.prepare("INSERT INTO places(city) VALUES (?1)").expect("prepare places insert");
        for i in 0..(row_count * 4) {
            let city = if i % 19 == 0 { None } else { Some(format!("city_{:03}", i % 128)) };
            places.execute(params![city]).expect("insert place");
        }
    }

    conn.execute_batch("COMMIT").expect("commit txn");
    conn.close().expect("close connection");
    file
}

fn fold_miniql_value(value: Option<ValueRef<'_>>) -> i64 {
    match value {
        Some(ValueRef::Null) | None => 0,
        Some(ValueRef::Integer(v)) => v,
        Some(ValueRef::Real(v)) => normalized_real_bits(v) as i64,
        Some(ValueRef::Text(bytes)) | Some(ValueRef::Blob(bytes)) => bytes.len() as i64,
    }
}

fn fold_miniql_row(row: Row<'_>) -> i64 {
    let mut checksum = 0i64;
    for idx in 0..row.len() {
        checksum = checksum.wrapping_mul(31).wrapping_add(fold_miniql_value(row.get(idx)));
    }
    checksum
}

fn fold_sqlite_value(value: SqliteValueRef<'_>) -> i64 {
    match value {
        SqliteValueRef::Null => 0,
        SqliteValueRef::Integer(v) => v,
        SqliteValueRef::Real(v) => normalized_real_bits(v) as i64,
        SqliteValueRef::Text(bytes) | SqliteValueRef::Blob(bytes) => bytes.len() as i64,
    }
}

fn fold_sqlite_row(row: &SqliteRow<'_>, col_count: usize) -> i64 {
    let mut checksum = 0i64;
    for idx in 0..col_count {
        let value = row.get_ref(idx).expect("read sqlite value");
        checksum = checksum.wrapping_mul(31).wrapping_add(fold_sqlite_value(value));
    }
    checksum
}

fn normalized_real_bits(value: f64) -> u64 {
    if value == 0.0 {
        0.0f64.to_bits()
    } else if value.is_nan() {
        0x7ff8_0000_0000_0000
    } else {
        value.to_bits()
    }
}

fn bench_sql_scan(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("sql_scan");
    for &row_count in ROW_COUNTS {
        let db_file = create_sql_bench_db(row_count);
        let db_path = db_file.path().to_path_buf();

        bench_group.bench_with_input(BenchmarkId::new("miniql", row_count), &row_count, |b, _| {
            let db = Db::open(&db_path).expect("open db");
            let mut scratch = ScanScratch::with_capacity(8, 0);
            b.iter(|| {
                let mut rows = 0usize;
                let mut checksum = 0i64;
                db.query(SCAN_SQL, &mut scratch, |row| {
                    checksum = checksum.wrapping_add(fold_miniql_row(row));
                    rows += 1;
                    Ok(())
                })
                .expect("run miniql scan sql");
                black_box((rows, checksum))
            });
        });

        bench_group.bench_with_input(
            BenchmarkId::new("rusqlite", row_count),
            &row_count,
            |b, _| {
                let conn = Connection::open(&db_path).expect("open db");
                let mut stmt = conn.prepare(SCAN_SQL).expect("prepare query");
                let col_count = stmt.column_count();
                b.iter(|| {
                    let mut rows_iter = stmt.query([]).expect("run query");
                    let mut rows = 0usize;
                    let mut checksum = 0i64;
                    while let Some(row) = rows_iter.next().expect("next row") {
                        checksum = checksum.wrapping_add(fold_sqlite_row(row, col_count));
                        rows += 1;
                    }
                    black_box((rows, checksum))
                });
            },
        );
    }
}

fn bench_sql_aggregate_order(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("sql_aggregate_order");
    let variants = [("projected", AGG_SQL), ("hidden_order_expr", AGG_HIDDEN_ORDER_SQL)];
    for &row_count in ROW_COUNTS {
        let db_file = create_sql_bench_db(row_count);
        let db_path = db_file.path().to_path_buf();

        for (variant, sql) in variants {
            bench_group.bench_with_input(
                BenchmarkId::new(format!("miniql_{variant}"), row_count),
                &row_count,
                |b, _| {
                    let db = Db::open(&db_path).expect("open db");
                    let mut scratch = ScanScratch::with_capacity(4, 0);
                    b.iter(|| {
                        let mut rows = 0usize;
                        let mut checksum = 0i64;
                        db.query(sql, &mut scratch, |row| {
                            checksum = checksum.wrapping_add(fold_miniql_row(row));
                            rows += 1;
                            Ok(())
                        })
                        .expect("run miniql aggregate sql");
                        black_box((rows, checksum))
                    });
                },
            );

            bench_group.bench_with_input(
                BenchmarkId::new(format!("rusqlite_{variant}"), row_count),
                &row_count,
                |b, _| {
                    let conn = Connection::open(&db_path).expect("open db");
                    let mut stmt = conn.prepare(sql).expect("prepare query");
                    let col_count = stmt.column_count();
                    b.iter(|| {
                        let mut rows_iter = stmt.query([]).expect("run query");
                        let mut rows = 0usize;
                        let mut checksum = 0i64;
                        while let Some(row) = rows_iter.next().expect("next row") {
                            checksum = checksum.wrapping_add(fold_sqlite_row(row, col_count));
                            rows += 1;
                        }
                        black_box((rows, checksum))
                    });
                },
            );
        }
    }
}

fn bench_sql_join(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("sql_join");
    let variants = [("inner", JOIN_SQL), ("left", LEFT_JOIN_SQL)];
    for &row_count in ROW_COUNTS {
        let db_file = create_sql_bench_db(row_count);
        let db_path = db_file.path().to_path_buf();

        for (variant, sql) in variants {
            bench_group.bench_with_input(
                BenchmarkId::new(format!("miniql_{variant}"), row_count),
                &row_count,
                |b, _| {
                    let db = Db::open(&db_path).expect("open db");
                    let mut scratch = ScanScratch::with_capacity(8, 0);
                    b.iter(|| {
                        let mut rows = 0usize;
                        let mut checksum = 0i64;
                        db.query(sql, &mut scratch, |row| {
                            checksum = checksum.wrapping_add(fold_miniql_row(row));
                            rows += 1;
                            Ok(())
                        })
                        .expect("run miniql join sql");
                        black_box((rows, checksum))
                    });
                },
            );

            bench_group.bench_with_input(
                BenchmarkId::new(format!("rusqlite_{variant}"), row_count),
                &row_count,
                |b, _| {
                    let conn = Connection::open(&db_path).expect("open db");
                    let mut stmt = conn.prepare(sql).expect("prepare query");
                    let col_count = stmt.column_count();
                    b.iter(|| {
                        let mut rows_iter = stmt.query([]).expect("run query");
                        let mut rows = 0usize;
                        let mut checksum = 0i64;
                        while let Some(row) = rows_iter.next().expect("next row") {
                            checksum = checksum.wrapping_add(fold_sqlite_row(row, col_count));
                            rows += 1;
                        }
                        black_box((rows, checksum))
                    });
                },
            );
        }
    }
}

fn bench_sql_distinct(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("sql_distinct");
    let variants = [("low_card", DISTINCT_SQL), ("high_card", DISTINCT_HIGH_CARD_SQL)];
    for &row_count in ROW_COUNTS {
        let db_file = create_sql_bench_db(row_count);
        let db_path = db_file.path().to_path_buf();

        for (variant, sql) in variants {
            bench_group.bench_with_input(
                BenchmarkId::new(format!("miniql_{variant}"), row_count),
                &row_count,
                |b, _| {
                    let db = Db::open(&db_path).expect("open db");
                    let mut scratch = ScanScratch::with_capacity(2, 0);
                    b.iter(|| {
                        let mut rows = 0usize;
                        let mut checksum = 0i64;
                        db.query(sql, &mut scratch, |row| {
                            checksum = checksum.wrapping_add(fold_miniql_row(row));
                            rows += 1;
                            Ok(())
                        })
                        .expect("run miniql distinct sql");
                        black_box((rows, checksum))
                    });
                },
            );

            bench_group.bench_with_input(
                BenchmarkId::new(format!("rusqlite_{variant}"), row_count),
                &row_count,
                |b, _| {
                    let conn = Connection::open(&db_path).expect("open db");
                    let mut stmt = conn.prepare(sql).expect("prepare query");
                    let col_count = stmt.column_count();
                    b.iter(|| {
                        let mut rows_iter = stmt.query([]).expect("run query");
                        let mut rows = 0usize;
                        let mut checksum = 0i64;
                        while let Some(row) = rows_iter.next().expect("next row") {
                            checksum = checksum.wrapping_add(fold_sqlite_row(row, col_count));
                            rows += 1;
                        }
                        black_box((rows, checksum))
                    });
                },
            );
        }
    }
}

criterion_group!(
    benches,
    bench_sql_scan,
    bench_sql_aggregate_order,
    bench_sql_join,
    bench_sql_distinct
);
criterion_main!(benches);
