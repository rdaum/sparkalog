use sparkalog_execution::{
    CudaStream, anti_join_cpu_parallel, anti_join_cuda, distinct_cpu_parallel, distinct_cuda,
    join_cpu_parallel, join_cuda, union_cpu_parallel, union_cpu_serial, union_cuda,
};
use sparkalog_relational::{
    BinaryDistinct, BinaryEqualityJoin, JoinInput, JoinProjection, SortedBinaryAntiJoin,
    SortedBinaryUnion,
};
use sparkalog_storage::{
    AntiJoinWorkspace, DistinctWorkspace, JoinWorkspace, RelationView, U32RangeIndex,
    UnionWorkspace, load_binary_u32,
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

fn anti_join_plan() -> SortedBinaryAntiJoin {
    SortedBinaryAntiJoin {
        left: [0, 1],
        right: [0, 1],
    }
}

fn union_plan() -> SortedBinaryUnion {
    SortedBinaryUnion {
        left: [0, 1],
        right: [0, 1],
    }
}

struct Inputs {
    candidates: JoinWorkspace,
    distinct_candidates: DistinctWorkspace,
    full: DistinctWorkspace,
    newt: AntiJoinWorkspace,
}

impl Inputs {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            candidates: JoinWorkspace::new(2)?,
            distinct_candidates: DistinctWorkspace::new()?,
            full: DistinctWorkspace::new()?,
            newt: AntiJoinWorkspace::new()?,
        })
    }
}

fn prepare_inputs(
    producer: Producer,
    delta: RelationView<'_>,
    edge: RelationView<'_>,
    index: &U32RangeIndex,
    inputs: &mut Inputs,
    stream: &CudaStream,
) -> Result<(), Box<dyn std::error::Error>> {
    match producer {
        Producer::Cpu => {
            join_cpu_parallel(delta, edge, index, join_plan(), &mut inputs.candidates)?;
            distinct_cpu_parallel(
                inputs.candidates.output().view(),
                distinct_plan(),
                &mut inputs.distinct_candidates,
            )?;
            distinct_cpu_parallel(edge, distinct_plan(), &mut inputs.full)?;
            anti_join_cpu_parallel(
                inputs.distinct_candidates.output().view(),
                inputs.full.output().view(),
                anti_join_plan(),
                &mut inputs.newt,
            )?;
        }
        Producer::Gpu => {
            join_cuda(
                delta,
                edge,
                index,
                join_plan(),
                &mut inputs.candidates,
                stream,
            )?;
            distinct_cuda(
                inputs.candidates.output().view(),
                distinct_plan(),
                &mut inputs.distinct_candidates,
                stream,
            )?
            .wait()?;
            distinct_cuda(edge, distinct_plan(), &mut inputs.full, stream)?.wait()?;
            anti_join_cuda(
                inputs.distinct_candidates.output().view(),
                inputs.full.output().view(),
                anti_join_plan(),
                &mut inputs.newt,
                stream,
            )?
            .wait()?;
        }
    }
    Ok(())
}

fn run_union(
    backend: Backend,
    left: RelationView<'_>,
    right: RelationView<'_>,
    workspace: &mut UnionWorkspace,
    stream: &CudaStream,
) -> Result<(), Box<dyn std::error::Error>> {
    match backend {
        Backend::CpuSerial => union_cpu_serial(left, right, union_plan(), workspace)?,
        Backend::CpuParallel => union_cpu_parallel(left, right, union_plan(), workspace)?,
        Backend::Gpu => union_cuda(left, right, union_plan(), workspace, stream)?.wait()?,
    }
    Ok(())
}

fn sample_count(delta_rows: usize) -> usize {
    if delta_rows <= 8_192 {
        7
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
    let config = parse_config().map_err(|message| format!("union-crossover: {message}"))?;
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
        "producer,backend,delta_rows,left_rows,right_rows,output_rows,samples,p10_ns,median_ns,p90_ns\n",
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
        let mut inputs = Inputs::new()?;
        prepare_inputs(
            Producer::Cpu,
            delta,
            edge.view(),
            &index,
            &mut inputs,
            &stream,
        )?;
        let left_rows = inputs.full.output().len();
        let right_rows = inputs.newt.output().len();
        let mut reference = UnionWorkspace::new()?;
        union_cpu_serial(
            inputs.full.output().view(),
            inputs.newt.output().view(),
            union_plan(),
            &mut reference,
        )?;
        let output_rows = reference.output().len();

        for producer in producers {
            for backend in backends {
                let mut workspace = UnionWorkspace::new()?;
                prepare_inputs(producer, delta, edge.view(), &index, &mut inputs, &stream)?;
                run_union(
                    backend,
                    inputs.full.output().view(),
                    inputs.newt.output().view(),
                    &mut workspace,
                    &stream,
                )?;
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
                    prepare_inputs(producer, delta, edge.view(), &index, &mut inputs, &stream)?;
                    let start = Instant::now();
                    run_union(
                        backend,
                        inputs.full.output().view(),
                        inputs.newt.output().view(),
                        &mut workspace,
                        &stream,
                    )?;
                    elapsed.push(start.elapsed());
                    assert_eq!(workspace.output().len(), output_rows);
                    black_box(workspace.output().view().column_slice(0));
                    black_box(workspace.output().view().column_slice(1));
                }
                elapsed.sort_unstable();
                let p10 = percentile(&elapsed, 1, 10);
                let median = percentile(&elapsed, 1, 2);
                let p90 = percentile(&elapsed, 9, 10);
                writeln!(
                    csv,
                    "{},{},{delta_rows},{left_rows},{right_rows},{output_rows},{samples},{p10},{median},{p90}",
                    producer.name(),
                    backend.name()
                )?;
                eprintln!(
                    "producer={} backend={} delta_rows={} left_rows={} right_rows={} output_rows={} median_ms={:.3}",
                    producer.name(),
                    backend.name(),
                    delta_rows,
                    left_rows,
                    right_rows,
                    output_rows,
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
