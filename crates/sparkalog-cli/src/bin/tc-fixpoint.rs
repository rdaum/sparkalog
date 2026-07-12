use sparkalog_execution::{
    AntiJoinPlacementPolicy, CudaStream, DistinctPlacementPolicy, InputProvenance,
    JoinPlacementPolicy, UnionPlacementPolicy,
};
use sparkalog_recursion::{
    IterationPolicies, RecursiveExecutor, RelationStore, transitive_closure_scc,
};
use sparkalog_relational::RelationId;
use sparkalog_storage::Relation;
use std::time::Instant;

#[derive(Clone, Copy)]
enum Backend {
    Auto,
    Cpu,
    Gpu,
}

struct Config {
    vertices: usize,
    max_iterations: usize,
    backend: Backend,
}

fn parse_config() -> Result<Config, String> {
    let mut vertices = 128;
    let mut max_iterations = 1_024;
    let mut backend = Backend::Auto;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--vertices" => {
                let value = arguments.next().ok_or("--vertices requires a count")?;
                vertices = value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid vertex count: {value}"))?;
                if !(2..=4_096).contains(&vertices) {
                    return Err("vertex count must be between 2 and 4,096".to_owned());
                }
            }
            "--max-iterations" => {
                let value = arguments
                    .next()
                    .ok_or("--max-iterations requires a count")?;
                max_iterations = value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid iteration count: {value}"))?;
            }
            "--backend" => {
                backend = match arguments.next().as_deref() {
                    Some("auto") => Backend::Auto,
                    Some("cpu") => Backend::Cpu,
                    Some("gpu") => Backend::Gpu,
                    Some(value) => return Err(format!("invalid backend: {value}")),
                    None => return Err("--backend requires auto, cpu, or gpu".to_owned()),
                };
            }
            _ => return Err(format!("unknown argument: {argument}")),
        }
    }
    Ok(Config {
        vertices,
        max_iterations,
        backend,
    })
}

fn cpu_policies() -> IterationPolicies {
    IterationPolicies {
        join: JoinPlacementPolicy {
            gpu_min_delta_rows: usize::MAX,
            gpu_unavailable_parallel_min_rows: usize::MAX,
        },
        distinct: DistinctPlacementPolicy {
            cpu_produced_gpu_min_rows: usize::MAX,
            gpu_produced_gpu_min_rows: usize::MAX,
            gpu_unavailable_parallel_min_rows: usize::MAX,
        },
        anti_join: AntiJoinPlacementPolicy {
            cpu_produced_gpu_min_rows: usize::MAX,
            gpu_produced_gpu_min_rows: usize::MAX,
            cpu_produced_parallel_min_rows: usize::MAX,
            gpu_produced_parallel_min_rows: usize::MAX,
        },
        union: UnionPlacementPolicy {
            cpu_produced_gpu_min_rows: usize::MAX,
            gpu_produced_gpu_min_rows: usize::MAX,
            gpu_unavailable_parallel_min_rows: usize::MAX,
        },
    }
}

fn gpu_policies() -> IterationPolicies {
    IterationPolicies {
        join: JoinPlacementPolicy {
            gpu_min_delta_rows: 0,
            gpu_unavailable_parallel_min_rows: usize::MAX,
        },
        distinct: DistinctPlacementPolicy {
            cpu_produced_gpu_min_rows: 0,
            gpu_produced_gpu_min_rows: 0,
            gpu_unavailable_parallel_min_rows: usize::MAX,
        },
        anti_join: AntiJoinPlacementPolicy {
            cpu_produced_gpu_min_rows: 0,
            gpu_produced_gpu_min_rows: 0,
            cpu_produced_parallel_min_rows: usize::MAX,
            gpu_produced_parallel_min_rows: usize::MAX,
        },
        union: UnionPlacementPolicy {
            cpu_produced_gpu_min_rows: 0,
            gpu_produced_gpu_min_rows: 0,
            gpu_unavailable_parallel_min_rows: usize::MAX,
        },
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_config().map_err(|message| format!("tc-fixpoint: {message}"))?;
    let edges = config.vertices - 1;
    let mut edge = Relation::new(2, edges)?;
    for row in 0..edges {
        edge.column_mut(0).unwrap().as_mut_slice()[row] = row as u32;
        edge.column_mut(1).unwrap().as_mut_slice()[row] = row as u32 + 1;
    }
    let stream = CudaStream::new()?;
    let (policies, stream) = match config.backend {
        Backend::Auto => (IterationPolicies::default(), Some(&stream)),
        Backend::Cpu => (cpu_policies(), None),
        Backend::Gpu => (gpu_policies(), Some(&stream)),
    };
    let edge_id = RelationId(0);
    let path_id = RelationId(1);
    let mut store = RelationStore::new();
    store.insert_static(edge_id, edge.view(), InputProvenance::Cpu)?;
    store.insert_recursive(path_id, edge.view(), edge.view(), InputProvenance::Cpu)?;
    let mut executor =
        RecursiveExecutor::compile(transitive_closure_scc(path_id, edge_id), &store)?;
    let start = Instant::now();
    let summary = executor.run(&mut store, stream, policies, config.max_iterations)?;
    let elapsed = start.elapsed();
    let expected = config.vertices * (config.vertices - 1) / 2;
    assert_eq!(store.full(path_id)?.len(), expected);
    assert!(store.delta(path_id)?.is_empty());
    println!(
        "vertices={} closure_rows={} iterations={} elapsed_ms={:.3} last_placements={:?}",
        config.vertices,
        store.full(path_id)?.len(),
        summary.iterations,
        elapsed.as_secs_f64() * 1_000.0,
        summary.last_placements,
    );
    Ok(())
}
