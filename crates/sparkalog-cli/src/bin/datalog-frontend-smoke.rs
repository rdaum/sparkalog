use std::fmt::Write;
use std::time::Instant;

use sparkalog_datalog::{Database, DelimitedOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rows = std::env::args()
        .nth(1)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(100_000);
    let mut facts = String::with_capacity(rows.saturating_mul(16));
    for value in 0..rows {
        writeln!(
            &mut facts,
            "{},{}",
            value as u32,
            value.wrapping_add(1) as u32
        )?;
    }
    let mut database = Database::new();
    database.load_program(
        ".decl edge(from:number, to:number) .input edge copy(x, y) :- edge(x, y). .output copy",
    )?;

    let ingest_start = Instant::now();
    database.load_delimited("edge", &facts, DelimitedOptions::default())?;
    let ingest = ingest_start.elapsed();
    let execute_start = Instant::now();
    let summary = database.run()?;
    let execute = execute_start.elapsed();
    let output_rows = database.query("copy")?.rows.len();

    assert_eq!(output_rows, rows);
    println!(
        "rows={rows} ingest_ms={:.3} execute_ms={:.3} backend={:?}",
        ingest.as_secs_f64() * 1_000.0,
        execute.as_secs_f64() * 1_000.0,
        summary.backend
    );
    Ok(())
}
