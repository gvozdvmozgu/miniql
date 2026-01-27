use codspeed_criterion_compat::{
    BenchmarkId, Criterion, black_box, criterion_group, criterion_main,
};
use miniql::{Db, ScanScratch, avg, col, count, count_star, group, lit_i64, sum};
use rusqlite::{Connection, params};
use tempfile::NamedTempFile;

const ROW_COUNTS: &[usize] = &[1_000, 10_000];

fn create_group_db(row_count: usize) -> NamedTempFile {
    let file = NamedTempFile::new().expect("create temp file");
    let conn = Connection::open(file.path()).expect("open connection");

    conn.execute_batch(
        "
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        CREATE TABLE items(grp INTEGER, val INTEGER);
        ",
    )
    .expect("create schema");

    let groups = (row_count / 10).max(1);
    conn.execute_batch("BEGIN").unwrap();
    {
        let mut stmt =
            conn.prepare("INSERT INTO items(grp, val) VALUES (?1, ?2)").expect("prepare insert");
        for i in 0..row_count {
            let grp = if i % 25 == 0 { None } else { Some((i % groups) as i64) };
            let val = if i % 10 == 0 { None } else { Some((i % 1000) as i64) };
            stmt.execute(params![grp, val]).expect("insert item");
        }
    }
    conn.execute_batch("COMMIT").unwrap();

    conn.close().expect("close connection");
    file
}

fn bench_grouped_aggregate(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("aggregate_grouped");
    let threshold = 1_000i64;

    for &row_count in ROW_COUNTS {
        let db_file = create_group_db(row_count);
        let db_path = db_file.path();

        bench_group.bench_with_input(BenchmarkId::new("miniql", row_count), &row_count, |b, _| {
            let db = Db::open(db_path).expect("open db");
            let items = db.table("items").expect("items table");
            let mut scratch = ScanScratch::with_capacity(2, 0);
            let mut agg = items
                .scan()
                .aggregate([group(col(0)), sum(col(1)), count(col(1))])
                .group_by([col(0)])
                .having(col(1).gt(lit_i64(threshold)))
                .compile()
                .expect("compile aggregate");

            b.iter(|| {
                let mut seen = 0usize;
                agg.for_each(&mut scratch, |row| {
                    let _ = row.get(0);
                    seen += 1;
                    Ok(())
                })
                .expect("aggregate");
                black_box(seen)
            });
        });

        bench_group.bench_with_input(
            BenchmarkId::new("rusqlite", row_count),
            &row_count,
            |b, _| {
                let conn = Connection::open(db_path).expect("open db");
                let mut stmt = conn
                    .prepare(
                        "SELECT grp, SUM(val), COUNT(val) FROM items GROUP BY grp HAVING SUM(val) \
                         > ?1",
                    )
                    .expect("prepare");

                b.iter(|| {
                    let mut rows = stmt.query(params![threshold]).expect("query");
                    let mut seen = 0usize;
                    while let Some(row) = rows.next().expect("next") {
                        let _grp: Option<i64> = row.get(0).expect("grp");
                        let _sum: Option<i64> = row.get(1).expect("sum");
                        let _count: i64 = row.get(2).expect("count");
                        seen += 1;
                    }
                    black_box(seen)
                });
            },
        );
    }
}

fn bench_single_group_aggregate(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("aggregate_single_group");

    for &row_count in ROW_COUNTS {
        let db_file = create_group_db(row_count);
        let db_path = db_file.path();

        bench_group.bench_with_input(BenchmarkId::new("miniql", row_count), &row_count, |b, _| {
            let db = Db::open(db_path).expect("open db");
            let items = db.table("items").expect("items table");
            let mut scratch = ScanScratch::with_capacity(2, 0);
            let mut agg = items
                .scan()
                .aggregate([sum(col(1)), avg(col(1)), count_star()])
                .compile()
                .expect("compile aggregate");

            b.iter(|| {
                let mut seen = 0usize;
                agg.for_each(&mut scratch, |row| {
                    let _sum = row.get(0);
                    seen += 1;
                    Ok(())
                })
                .expect("aggregate");
                black_box(seen)
            });
        });

        bench_group.bench_with_input(
            BenchmarkId::new("rusqlite", row_count),
            &row_count,
            |b, _| {
                let conn = Connection::open(db_path).expect("open db");
                let mut stmt = conn
                    .prepare("SELECT SUM(val), AVG(val), COUNT(*) FROM items")
                    .expect("prepare");

                b.iter(|| {
                    let mut rows = stmt.query([]).expect("query");
                    let mut seen = 0usize;
                    while let Some(row) = rows.next().expect("next") {
                        let _sum: Option<i64> = row.get(0).expect("sum");
                        let _avg: Option<f64> = row.get(1).expect("avg");
                        let _count: i64 = row.get(2).expect("count");
                        seen += 1;
                    }
                    black_box(seen)
                });
            },
        );
    }
}

criterion_group!(benches, bench_grouped_aggregate, bench_single_group_aggregate);
criterion_main!(benches);
