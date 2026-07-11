# Filter crossover benchmark

`filter-crossover.csv` is the full filter/compaction matrix recorded on
2026-07-11 on the local NVIDIA GB10 (compute capability 12.1) with CUDA 13.0
and Rust 1.95.0.

Reproduce it with:

```sh
cargo run --release --bin filter-crossover -- \
  --output benchmarks/filter-crossover.csv
```

For each sample, the selected input producer fills `row_id % 100` into the
canonical managed column immediately before the timed interval. GPU production
is synchronized without a CPU read. The timed interval covers filter dispatch,
execution, compaction, required synchronization, and CPU-visible result length.
The producer itself is outside the interval. Output cardinality is checked
after every sample.

The matrix contains 240 cases:

- 10 cardinalities from 32 through 8,388,608 rows;
- 1%, 10%, 50%, and 90% selectivity;
- CPU- and GPU-produced inputs;
- serial Rust, Rayon-parallel Rust, and CUDA+CUB execution.

## Findings

- Serial Rust won all 40 cases through 8,192 rows.
- At 32,768 rows CUDA won seven of eight cases. Serial Rust retained a narrow
  win for the CPU-produced, 1%-selectivity case; the CPU-produced 10% case was
  effectively tied at 15,584 ns for serial Rust versus 15,536 ns for CUDA.
- CUDA won every case at 131,072 rows and above.
- Parallel Rust did not win a case. Its persistent Rayon pool removed the
  roughly one-millisecond thread-creation cost of the initial scoped-thread
  implementation, but the GPU crossover still occurs before parallel CPU
  compaction overtakes serial Rust consistently.

The first placement policy should therefore use serial Rust for small filters
and CUDA for large filters. Parallel Rust remains a useful GPU-unavailable
fallback, but the data does not support inserting it between those tiers on
this machine.
