use std::path::Path;

use sparkalog_datalog::{Database, InternedValue};

fn main() {
    if let Err(error) = run() {
        eprintln!("sparkalog: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let command = arguments.next().unwrap_or_else(|| "help".into());
    match command.as_str() {
        "run" | "check" => {
            let path = arguments
                .next()
                .ok_or_else(|| format!("usage: sparkalog {command} FILE"))?;
            if arguments.next().is_some() {
                return Err(format!("usage: sparkalog {command} FILE").into());
            }
            run_file(&command, Path::new(&path))
        }
        "help" | "--help" | "-h" => {
            println!("usage: sparkalog <run|check> FILE");
            Ok(())
        }
        _ => Err(format!("unknown command {command}; expected run or check").into()),
    }
}

fn run_file(command: &str, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let source = std::fs::read_to_string(path)?;
    let mut database = Database::new();
    database.load_program(&source)?;
    if command == "check" {
        println!("{}: valid", path.display());
        return Ok(());
    }
    let summary = database.run()?;
    for output in database.outputs()? {
        for tuple in output.rows {
            let values = tuple
                .iter()
                .map(format_value)
                .collect::<Vec<_>>()
                .join(", ");
            println!("{}({values}).", output.predicate);
        }
    }
    let iterations = summary.sccs.iter().map(|scc| scc.iterations).sum::<usize>();
    eprintln!(
        "completed {} components in {iterations} fixpoint rounds",
        summary.sccs.len()
    );
    Ok(())
}

fn format_value(value: &InternedValue) -> String {
    match value {
        InternedValue::U32(value) => value.to_string(),
        InternedValue::String(value) => format!("{value:?}"),
        InternedValue::Symbol(value) => format!("'{value}"),
    }
}
