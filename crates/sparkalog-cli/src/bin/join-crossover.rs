use sparkalog_execution::{
    CudaStream, join_cpu_bitmap_parallel, join_cpu_bitmap_serial, join_cpu_parallel,
    join_cpu_serial, join_cuda,
};
use sparkalog_relational::{BinaryEqualityJoin, JoinInput, JoinProjection};
use sparkalog_storage::{
    JoinWorkspace, RelationView, U32BitmapIndex, U32RangeIndex, load_binary_u32,
};
use std::fmt::Write as _;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
enum Backend {
    CpuRangeSerial,
    CpuRangeParallel,
    CpuBitmapSerial,
    CpuBitmapParallel,
    Gpu,
}

impl Backend {
    fn name(self) -> &'static str {
        match self {
            Self::CpuRangeSerial => "cpu_range_serial",
            Self::CpuRangeParallel => "cpu_range_parallel",
            Self::CpuBitmapSerial => "cpu_bitmap_serial",
            Self::CpuBitmapParallel => "cpu_bitmap_parallel",
            Self::Gpu => "gpu",
        }
    }
}

struct Config {
    graph: PathBuf,
    output: Option<PathBuf>,
    quick: bool,
}

fn parse_config() -> Result<Config, String> {
    let mut graph = PathBuf::from("reference/gdlog/data/com-dblp/edge.facts");
    let mut output = None;
    let mut quick = false;
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
            _ => return Err(format!("unknown argument: {argument}")),
        }
    }
    Ok(Config {
        graph,
        output,
        quick,
    })
}

fn plan() -> BinaryEqualityJoin {
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

fn run_join(
    backend: Backend,
    delta: RelationView<'_>,
    edge: RelationView<'_>,
    range_index: &U32RangeIndex,
    bitmap_index: &U32BitmapIndex,
    workspace: &mut JoinWorkspace,
    stream: &CudaStream,
) -> Result<(), Box<dyn std::error::Error>> {
    match backend {
        Backend::CpuRangeSerial => join_cpu_serial(delta, edge, range_index, plan(), workspace)?,
        Backend::CpuRangeParallel => {
            join_cpu_parallel(delta, edge, range_index, plan(), workspace)?
        }
        Backend::CpuBitmapSerial => {
            join_cpu_bitmap_serial(delta, edge, bitmap_index, plan(), workspace)?
        }
        Backend::CpuBitmapParallel => {
            join_cpu_bitmap_parallel(delta, edge, bitmap_index, plan(), workspace)?
        }
        Backend::Gpu => join_cuda(delta, edge, range_index, plan(), workspace, stream)?,
    }
    Ok(())
}

fn sample_count(delta_rows: usize) -> usize {
    if delta_rows <= 8_192 {
        15
    } else if delta_rows <= 131_072 {
        7
    } else {
        3
    }
}

fn percentile(sorted: &[Duration], numerator: usize, denominator: usize) -> u128 {
    let index = (sorted.len() - 1) * numerator / denominator;
    sorted[index].as_nanos()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_config().map_err(|message| format!("join-crossover: {message}"))?;
    let load_start = Instant::now();
    let edge = load_binary_u32(&config.graph)?;
    let load_elapsed = load_start.elapsed();
    let range_index_start = Instant::now();
    let range_index = U32RangeIndex::build(edge.column(0).expect("binary edge source column"))?;
    let range_index_elapsed = range_index_start.elapsed();
    let bitmap_index_start = Instant::now();
    let bitmap_index = U32BitmapIndex::build(edge.column(0).expect("binary edge source column"))?;
    let bitmap_index_elapsed = bitmap_index_start.elapsed();
    let stream = CudaStream::new()?;
    let backends = [
        Backend::CpuRangeSerial,
        Backend::CpuRangeParallel,
        Backend::CpuBitmapSerial,
        Backend::CpuBitmapParallel,
        Backend::Gpu,
    ];
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
    let mut csv = String::from("backend,delta_rows,output_rows,samples,p10_ns,median_ns,p90_ns\n");

    eprintln!(
        "loaded {} edges in {:.3} ms; built {}-key range index in {:.3} ms; bitmap index in {:.3} ms",
        edge.len(),
        load_elapsed.as_secs_f64() * 1_000.0,
        range_index.unique_keys(),
        range_index_elapsed.as_secs_f64() * 1_000.0,
        bitmap_index_elapsed.as_secs_f64() * 1_000.0
    );

    for delta_rows in delta_sizes {
        let delta = edge
            .view()
            .prefix(delta_rows)
            .expect("delta size is within the edge relation");
        let mut reference = JoinWorkspace::new(2)?;
        run_join(
            Backend::CpuRangeSerial,
            delta,
            edge.view(),
            &range_index,
            &bitmap_index,
            &mut reference,
            &stream,
        )?;
        let expected_rows = reference.output().len();

        for backend in backends {
            let mut workspace = JoinWorkspace::new(2)?;
            run_join(
                backend,
                delta,
                edge.view(),
                &range_index,
                &bitmap_index,
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

            let samples = sample_count(delta_rows);
            let mut elapsed = Vec::with_capacity(samples);
            for _ in 0..samples {
                let start = Instant::now();
                run_join(
                    backend,
                    delta,
                    edge.view(),
                    &range_index,
                    &bitmap_index,
                    &mut workspace,
                    &stream,
                )?;
                elapsed.push(start.elapsed());
                assert_eq!(workspace.output().len(), expected_rows);
                black_box(workspace.output().view().column_slice(0));
                black_box(workspace.output().view().column_slice(1));
            }
            elapsed.sort_unstable();
            let p10 = percentile(&elapsed, 1, 10);
            let median = percentile(&elapsed, 1, 2);
            let p90 = percentile(&elapsed, 9, 10);
            writeln!(
                csv,
                "{},{delta_rows},{expected_rows},{samples},{p10},{median},{p90}",
                backend.name()
            )?;
            eprintln!(
                "backend={} delta_rows={} output_rows={} median_ms={:.3}",
                backend.name(),
                delta_rows,
                expected_rows,
                median as f64 / 1_000_000.0
            );
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
