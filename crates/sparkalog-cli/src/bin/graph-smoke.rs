use sparkalog_execution::{
    CudaStream, FilterPlacementPolicy, InputProvenance, filter_auto, filter_cpu_parallel,
    filter_cpu_serial, filter_cuda,
};
use sparkalog_relational::U32Predicate;
use sparkalog_storage::{LoadError, OperatorWorkspace, load_binary_u32};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("reference/gdlog/data/com-dblp/edge.facts"));
    run(&path).inspect_err(|error| {
        if matches!(
            error.downcast_ref::<LoadError>(),
            Some(LoadError::GitLfsPointer(_))
        ) {
            eprintln!("materialize the default graph with scripts/fetch-gdlog-data.sh");
        }
    })
}

fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let load_start = Instant::now();
    let relation = load_binary_u32(path)?;
    let load_time = load_start.elapsed();
    let source = relation
        .column(0)
        .expect("binary relation has a source column");
    let destination = relation
        .column(1)
        .expect("binary relation has a destination column");
    let max_node = source
        .as_slice()
        .iter()
        .chain(destination.as_slice())
        .copied()
        .max()
        .unwrap_or(0);
    let cutoff = max_node / 10 + 1;
    let predicate = U32Predicate::Lt(cutoff);
    let stream = CudaStream::new()?;

    let mut serial = OperatorWorkspace::new()?;
    serial.reserve_rows(relation.len())?;
    filter_cpu_serial(source, predicate, &mut serial)?;
    let start = Instant::now();
    filter_cpu_serial(source, predicate, &mut serial)?;
    let serial_time = start.elapsed();

    let mut parallel = OperatorWorkspace::new()?;
    parallel.reserve_rows(relation.len())?;
    filter_cpu_parallel(source, predicate, &mut parallel)?;
    let start = Instant::now();
    filter_cpu_parallel(source, predicate, &mut parallel)?;
    let parallel_time = start.elapsed();

    let mut gpu = OperatorWorkspace::new()?;
    gpu.reserve_rows(relation.len())?;
    filter_cuda(source, predicate, &mut gpu, &stream)?.wait()?;
    let start = Instant::now();
    filter_cuda(source, predicate, &mut gpu, &stream)?.wait()?;
    let gpu_time = start.elapsed();

    assert_eq!(
        parallel.selection().as_slice(),
        serial.selection().as_slice()
    );
    assert_eq!(gpu.selection().as_slice(), serial.selection().as_slice());

    let mut automatic = OperatorWorkspace::new()?;
    let selected_placement = filter_auto(
        source,
        predicate,
        InputProvenance::Cpu,
        &mut automatic,
        Some(&stream),
        FilterPlacementPolicy::MEASURED_GB10,
    )?;
    assert_eq!(
        automatic.selection().as_slice(),
        serial.selection().as_slice()
    );

    println!("graph={}", path.display());
    println!(
        "rows={} arity={} max_node={} load_ms={:.3}",
        relation.len(),
        relation.arity(),
        max_node,
        load_time.as_secs_f64() * 1_000.0
    );
    println!(
        "filter=source<{cutoff} selected={} serial_us={:.3} parallel_us={:.3} gpu_us={:.3}",
        serial.selection().len(),
        serial_time.as_secs_f64() * 1_000_000.0,
        parallel_time.as_secs_f64() * 1_000_000.0,
        gpu_time.as_secs_f64() * 1_000_000.0,
    );
    println!("automatic_placement={selected_placement:?}");
    Ok(())
}
