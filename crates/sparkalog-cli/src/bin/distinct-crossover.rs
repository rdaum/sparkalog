use sparkalog_execution::{
    CudaStream, distinct_cpu_parallel, distinct_cpu_serial, distinct_cuda, join_cpu_parallel,
    join_cuda,
};
use sparkalog_relational::{BinaryDistinct, BinaryEqualityJoin, JoinInput, JoinProjection};
use sparkalog_storage::{
    DistinctWorkspace, JoinWorkspace, RelationView, U32RangeIndex, load_binary_u32,
};
use std::fmt::Write as _;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
enum Producer {
    Cpu,
    Gpu,
}

impl Producer {
    fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }
}

#[derive(Clone, Copy)]
enum Backend {
    CpuSerial,
    CpuParallel,
    Gpu,
}

impl Backend {
    fn name(self) -> &'static str {
        match self {
            Self::CpuSerial => "cpu_serial",
            Self::CpuParallel => "cpu_parallel",
            Self::Gpu => "gpu",
        }
    }
}

struct Config {
    graph: PathBuf,
    output: Option<PathBuf>,
    quick: bool,
    sample_override: Option<usize>,
}

fn parse_config() -> Result<Config, String> {
    let mut graph = PathBuf::from("reference/gdlog/data/com-dblp/edge.facts");
    let mut output = None;
    let mut quick = false;
    let mut sample_override = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--graph" => {
                graph = PathBuf::from(arguments.next().ok_or("--graph requires a path")?);
            }
            "--output" => {
                output = Some(PathBuf::from(
                    arguments.next().ok_or("--output requires a path")?,
                ));
            }
            "--quick" => quick = true,
            "--samples" => {
                let value = arguments.next().ok_or("--samples requires a count")?;
                let count = value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid sample count: {value}"))?;
                if count == 0 {
                    return Err("sample count must be greater than zero".to_owned());
                }
                sample_override = Some(count);
            }
            _ => return Err(format!("unknown argument: {argument}")),
        }
    }
    Ok(Config {
        graph,
        output,
        quick,
        sample_override,
    })
}

fn join_plan() -> BinaryEqualityJoin {
    BinaryEqualityJoin {
        left_key: 1,
        right_key: 0,
        output: [
            JoinProjection {
                input: JoinInput::Left,
                column: 0,
            },
            JoinProjection {
                input: JoinInput::Right,
                column: 1,
            },
        ],
    }
}

fn distinct_plan() -> BinaryDistinct {
    BinaryDistinct { columns: [0, 1] }
}

fn produce_candidates(
    producer: Producer,
    delta: RelationView<'_>,
    edge: RelationView<'_>,
    index: &U32RangeIndex,
    workspace: &mut JoinWorkspace,
    stream: &CudaStream,
) -> Result<(), Box<dyn std::error::Error>> {
    match producer {
        Producer::Cpu => {
            join_cpu_parallel(delta, edge, index, join_plan(), workspace)?;
        }
        Producer::Gpu => join_cuda(delta, edge, index, join_plan(), workspace, stream)?,
    }
    Ok(())
}

fn run_distinct(
    backend: Backend,
    input: RelationView<'_>,
    workspace: &mut DistinctWorkspace,
    stream: &CudaStream,
) -> Result<(), Box<dyn std::error::Error>> {
    match backend {
        Backend::CpuSerial => distinct_cpu_serial(input, distinct_plan(), workspace)?,
        Backend::CpuParallel => distinct_cpu_parallel(input, distinct_plan(), workspace)?,
        Backend::Gpu => distinct_cuda(input, distinct_plan(), workspace, stream)?.wait()?,
    }
    Ok(())
}

fn sample_count(delta_rows: usize) -> usize {
    if delta_rows <= 8_192 {
        10
    } else if delta_rows <= 131_072 {
        5
    } else {
        3
    }
}

fn percentile(sorted: &[Duration], numerator: usize, denominator: usize) -> u128 {
    let index = (sorted.len() - 1) * numerator / denominator;
    sorted[index].as_nanos()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_config().map_err(|message| format!("distinct-crossover: {message}"))?;
    let load_start = Instant::now();
    let edge = load_binary_u32(&config.graph)?;
    let load_elapsed = load_start.elapsed();
    let index_start = Instant::now();
    let index = U32RangeIndex::build(edge.column(0).expect("binary edge source column"))?;
    let index_elapsed = index_start.elapsed();
    let stream = CudaStream::new()?;
    let producers = [Producer::Cpu, Producer::Gpu];
    let backends = [Backend::CpuSerial, Backend::CpuParallel, Backend::Gpu];
    let delta_sizes = if config.quick {
        vec![32, 8_192, 131_072]
    } else {
        vec![
            32,
            128,
            512,
            2_048,
            8_192,
            32_768,
            131_072,
            524_288,
            edge.len(),
        ]
    };
    let mut csv = String::from(
        "producer,backend,delta_rows,input_rows,unique_rows,samples,p10_ns,median_ns,p90_ns\n",
    );

    eprintln!(
        "loaded {} edges in {:.3} ms; built {}-key range index in {:.3} ms",
        edge.len(),
        load_elapsed.as_secs_f64() * 1_000.0,
        index.unique_keys(),
        index_elapsed.as_secs_f64() * 1_000.0
    );

    for delta_rows in delta_sizes {
        let delta = edge
            .view()
            .prefix(delta_rows)
            .expect("delta size is within the edge relation");
        let mut candidates = JoinWorkspace::new(2)?;
        produce_candidates(
            Producer::Cpu,
            delta,
            edge.view(),
            &index,
            &mut candidates,
            &stream,
        )?;
        let input_rows = candidates.output().len();
        let mut reference = DistinctWorkspace::new()?;
        distinct_cpu_serial(candidates.output().view(), distinct_plan(), &mut reference)?;
        let unique_rows = reference.output().len();

        for producer in producers {
            for backend in backends {
                let mut workspace = DistinctWorkspace::new()?;
                produce_candidates(
                    producer,
                    delta,
                    edge.view(),
                    &index,
                    &mut candidates,
                    &stream,
                )?;
                run_distinct(backend, candidates.output().view(), &mut workspace, &stream)?;
                assert_eq!(
                    workspace.output().view().column_slice(0),
                    reference.output().view().column_slice(0)
                );
                assert_eq!(
                    workspace.output().view().column_slice(1),
                    reference.output().view().column_slice(1)
                );

                let samples = config
                    .sample_override
                    .unwrap_or_else(|| sample_count(delta_rows));
                let mut elapsed = Vec::with_capacity(samples);
                for _ in 0..samples {
                    produce_candidates(
                        producer,
                        delta,
                        edge.view(),
                        &index,
                        &mut candidates,
                        &stream,
                    )?;
                    let start = Instant::now();
                    run_distinct(backend, candidates.output().view(), &mut workspace, &stream)?;
                    elapsed.push(start.elapsed());
                    assert_eq!(workspace.output().len(), unique_rows);
                    black_box(workspace.output().view().column_slice(0));
                    black_box(workspace.output().view().column_slice(1));
                }
                elapsed.sort_unstable();
                let p10 = percentile(&elapsed, 1, 10);
                let median = percentile(&elapsed, 1, 2);
                let p90 = percentile(&elapsed, 9, 10);
                writeln!(
                    csv,
                    "{},{},{delta_rows},{input_rows},{unique_rows},{samples},{p10},{median},{p90}",
                    producer.name(),
                    backend.name()
                )?;
                eprintln!(
                    "producer={} backend={} delta_rows={} input_rows={} unique_rows={} median_ms={:.3}",
                    producer.name(),
                    backend.name(),
                    delta_rows,
                    input_rows,
                    unique_rows,
                    median as f64 / 1_000_000.0
                );
            }
        }
    }

    if let Some(output) = config.output {
        std::fs::write(&output, &csv)?;
        eprintln!("wrote {}", output.display());
    } else {
        print!("{csv}");
    }
    Ok(())
}
