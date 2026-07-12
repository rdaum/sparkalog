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

## Stored policy

`FilterPlacementPolicy::MEASURED_GB10` records conservative thresholds which
held across the tested selectivities:

| Input provenance | GPU threshold |
| --- | ---: |
| CPU-produced | 131,072 rows |
| GPU-produced | 32,768 rows |

When CUDA is unavailable, the policy switches from serial to parallel Rust at
8,388,608 rows. With CUDA available, the measured GPU threshold is reached
first and parallel Rust is not selected.

# Indexed DBLP join crossover

`join-crossover.csv` records path expansion over the 1,049,866-edge SNAP DBLP
graph. Each case joins an increasing prefix of `edge` as
`delta(x,y) ⋈ edge(y,z) → candidate(x,z)`. Before timing, every backend is
checked against the serial range-index output, including tuple order.

The native comparison includes serial and Rayon-parallel execution over both
a contiguous sorted posting-list index and immutable `hi_sparse_bitset`
postings. The bitmap cardinality is stored beside each posting, so both index
types have an O(1) count lookup before output emission. CUDA uses the managed
range index and a count, exclusive-scan, emit pipeline.

On the recorded run the 189,114-key range index built in 16.5 ms and the
bitmap index in 70.5 ms. The range index won every serial and parallel native
case. Bitmap iteration was 1.6x slower at 512 delta rows, 2.3x slower at
131,072 rows, and 3.3x slower for the full graph in serial execution. This
operator performs key lookup plus complete posting enumeration, not a set-set
intersection, so contiguous row IDs are the better representation here.

Serial range-indexed Rust won through 512 delta rows. At 2,048 rows CUDA took
0.065 ms, versus 0.079 ms for parallel range-indexed Rust and 0.114 ms for
serial Rust. CUDA remained fastest for every larger delta, reaching 1.477 ms
for 7,064,738 emitted tuples. `JoinPlacementPolicy::MEASURED_GB10_DBLP`
therefore switches to CUDA at 2,048 delta rows, or to parallel Rust at the same
point when CUDA is unavailable. The threshold is specific to this graph's
fanout and skew; later planning should incorporate estimated output rows.

# DBLP candidate distinct crossover

`distinct-crossover.csv` records sort-and-unique over the candidate relations
produced by the DBLP expansion join. Each timed sample regenerates its input
outside the interval with either parallel Rust or CUDA, preserving CPU/GPU
producer provenance. The timed interval includes tuple packing, sorting,
unique compaction, unpacking to canonical columns, synchronization, and the
CPU-visible result length. Exact sorted output is checked against serial Rust
before samples are recorded.

All implementations use a temporary packed `u64` key for each `(u32, u32)`
tuple. Serial Rust uses `sort_unstable`, parallel Rust uses Rayon parallel
sort, and CUDA uses CUB radix sort followed by CUB unique. The packed buffers,
CUB scratch, unique count, and two output columns are reusable CUDA-managed
allocations.

Producer provenance did not materially change the crossover on this run.
Serial Rust won at 7,424 candidate rows: 0.059 ms for CPU-produced input and
0.050 ms for GPU-produced input, versus 0.094 and 0.090 ms for CUDA. At 31,060
rows CUDA won at 0.108 and 0.096 ms, versus 0.265 and 0.254 ms for serial Rust.
CUDA remained fastest above that point. For the full 7,064,738-row candidate
relation it took 7.829 ms, versus 30.096 ms for parallel Rust and 90.571 ms
for serial Rust on GPU-produced input.

`DistinctPlacementPolicy::MEASURED_GB10_DBLP` therefore selects serial Rust
below 32,768 candidate rows and CUDA from 32,768 rows for either producer.
When CUDA is unavailable, it switches from serial to parallel Rust at 131,072
rows. These thresholds cover the measured DBLP duplicate rates; tuple width
and key distribution remain relevant to future generalized distinct plans.
