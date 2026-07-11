use sparkalog_execution::{
    CudaStream, fill_mod_u32, filter_cpu_parallel, filter_cpu_serial, filter_cuda,
};
use sparkalog_relational::U32Predicate;
use sparkalog_storage::{Column, OperatorWorkspace};
use std::fmt::Write as _;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

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
    rows: Vec<usize>,
    output: Option<PathBuf>,
    sample_override: Option<usize>,
}

fn parse_config() -> Result<Config, String> {
    let mut quick = false;
    let mut output = None;
    let mut sample_override = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--quick" => quick = true,
            "--output" => {
                output = Some(PathBuf::from(
                    arguments.next().ok_or("--output requires a path")?,
                ));
            }
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

    let rows = if quick {
        vec![32, 4_096, 131_072]
    } else {
        vec![
            32, 128, 512, 2_048, 8_192, 32_768, 131_072, 524_288, 2_097_152, 8_388_608,
        ]
    };
    Ok(Config {
        rows,
        output,
        sample_override,
    })
}

fn prepare_input(
    column: &mut Column,
    producer: Producer,
    stream: &CudaStream,
) -> Result<(), Box<dyn std::error::Error>> {
    match producer {
        Producer::Cpu => {
            for (row, value) in column.as_mut_slice().iter_mut().enumerate() {
                *value = (row % 100) as u32;
            }
        }
        Producer::Gpu => fill_mod_u32(column, 100, stream)?.wait()?,
    }
    Ok(())
}

fn run_filter(
    column: &Column,
    predicate: U32Predicate,
    backend: Backend,
    workspace: &mut OperatorWorkspace,
    stream: &CudaStream,
) -> Result<(), Box<dyn std::error::Error>> {
    match backend {
        Backend::CpuSerial => filter_cpu_serial(column, predicate, workspace)?,
        Backend::CpuParallel => filter_cpu_parallel(column, predicate, workspace)?,
        Backend::Gpu => filter_cuda(column, predicate, workspace, stream)?.wait()?,
    }
    Ok(())
}

fn expected_selected(rows: usize, selectivity: usize) -> usize {
    (rows / 100) * selectivity + (rows % 100).min(selectivity)
}

fn sample_count(rows: usize) -> usize {
    if rows <= 8_192 {
        25
    } else if rows <= 524_288 {
        10
    } else {
        5
    }
}

fn percentile(sorted: &[u128], numerator: usize, denominator: usize) -> u128 {
    let index = (sorted.len() - 1) * numerator / denominator;
    sorted[index]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_config().map_err(|message| format!("filter-crossover: {message}"))?;
    let stream = CudaStream::new()?;
    let producers = [Producer::Cpu, Producer::Gpu];
    let backends = [Backend::CpuSerial, Backend::CpuParallel, Backend::Gpu];
    let selectivities = [1_usize, 10, 50, 90];
    let mut csv = String::from(
        "producer,backend,rows,selectivity_pct,selected,samples,p10_ns,median_ns,p90_ns\n",
    );

    for rows in config.rows {
        let mut column = Column::new_filled(rows, 0)?;
        for producer in producers {
            for backend in backends {
                let mut workspace = OperatorWorkspace::new()?;
                workspace.reserve_rows(rows)?;
                for selectivity in selectivities {
                    let predicate = U32Predicate::Lt(selectivity as u32);
                    let expected = expected_selected(rows, selectivity);

                    for _ in 0..2 {
                        prepare_input(&mut column, producer, &stream)?;
                        run_filter(&column, predicate, backend, &mut workspace, &stream)?;
                        assert_eq!(workspace.selection().len(), expected);
                    }

                    let samples = config.sample_override.unwrap_or_else(|| sample_count(rows));
                    let mut elapsed = Vec::with_capacity(samples);
                    for _ in 0..samples {
                        prepare_input(&mut column, producer, &stream)?;
                        let start = Instant::now();
                        run_filter(&column, predicate, backend, &mut workspace, &stream)?;
                        elapsed.push(start.elapsed().as_nanos());
                        assert_eq!(workspace.selection().len(), expected);
                        black_box(workspace.selection().as_slice());
                    }
                    elapsed.sort_unstable();
                    let p10 = percentile(&elapsed, 1, 10);
                    let median = percentile(&elapsed, 1, 2);
                    let p90 = percentile(&elapsed, 9, 10);
                    writeln!(
                        csv,
                        "{},{},{rows},{selectivity},{expected},{samples},{p10},{median},{p90}",
                        producer.name(),
                        backend.name(),
                    )?;
                    eprintln!(
                        "producer={} backend={} rows={} selectivity={} median_ns={}",
                        producer.name(),
                        backend.name(),
                        rows,
                        selectivity,
                        median
                    );
                }
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
