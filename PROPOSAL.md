# Sparkalog proposal

## Thesis

Sparkalog is an experiment in a Datalog-style relational engine for the DGX
Spark. Its first question is deliberately narrower than "can a GPU run
Datalog?": can a CPU scheduler and CUDA kernels operate on one canonical
relation representation without the usual CPU-RAM-to-VRAM replica and copy
cycle?

DGX Spark's CPU and integrated GPU share a dynamic, 128 GB LPDDR5x memory
space. NVIDIA documents this as eliminating redundant CPU-RAM/VRAM copies for
CPU/GPU collaboration, although the 273 GB/s memory fabric is shared by the
CPU, GPU, and the rest of the system. [DGX Spark memory model][spark-memory]

That changes the intended execution boundary from:

```text
CPU builds relation -> pack/copy to device -> GPU operator -> copy result back
```

to:

```text
CPU builds and publishes canonical columns -> GPU operator -> CPU schedules the next step
```

The claim is not that memory access is free. The GPU still consumes shared
memory bandwidth, and CPU work can contend with it. The claim is that a
relation need not have a separate device-resident twin merely to be processed
by CUDA.

## Why a Datalog engine

Semi-naive evaluation repeatedly produces and consumes delta relations. A
discrete-GPU design must decide which deltas, indexes, and intermediate
relations to copy, when to retain a device copy, and when to return data to the
CPU. GDlog is a useful reference point: it is a CUDA implementation of sparse
transitive closure and its documented full benchmark assumes an A100 with at
least 40 GB of GPU memory. [GDlog][gdlog]

Sparkalog should test whether shared memory makes a hybrid plan simpler:

- CPU owns parsing, planning, rule scheduling, and small or irregular work.
- GPU owns large, regular scans, joins, filtering, expansion, sorting, and
  deduplication.
- Both work over the same published column buffers.
- The scheduler chooses an operator placement; it does not first solve a data
  placement problem.

## Representation and ownership

The initial relation shape is structure-of-arrays, not object graphs:

```text
Relation2 { left: [u32; n], right: [u32; n] }
Delta2    { left: [u32; m], right: [u32; m] }
```

Contiguous columns give NEON code sequential vector loads and CUDA code
coalesced loads. The same layout will not always be ideal for both processors;
GPU joins may eventually justify a derived blocked or partitioned view. That
view must be earned by a measured hot path, not built into the first API.

The first allocation boundary is explicit. `ManagedBuffer<T>` uses
`cudaMallocManaged`, so CPU code writes a CUDA-managed allocation and a kernel
reads that same allocation. It does **not** claim that arbitrary `Vec<T>` or
`malloc` memory is universally valid as a CUDA kernel pointer. Once this path
is measured on the installed Spark driver, we can decide whether a broader
system-allocation API is safe and worthwhile.

## Synchronization invariant

Shared memory does not permit unsynchronised concurrent mutation.

1. The CPU fills a relation or delta buffer.
2. It publishes the completed range to an explicit CUDA stream.
3. The GPU reads or writes that range.
4. The CPU waits for the stream before reusing the range.

The first skeleton makes this visible: every operation accepts `CudaStream`,
and host observation follows `stream.synchronize()`. Future producer/consumer
queues need equivalent release/acquire and CUDA-event boundaries, especially
on the platform's relaxed ARM memory model.

## Initial vertical slices

### 0. Shared-buffer proof

Implemented now. The CPU fills one million `i32` values in a managed
allocation, a CUDA `sm_121` kernel increments them, and the CPU verifies the
same allocation after stream synchronization. There is no `cudaMemcpy` in this
path.

### 1. Column filter and compaction

Add a predicate over one `u32` column and compact selected row indexes into a
second shared buffer. Compare CPU NEON, GPU, and end-to-end hybrid time.

### 2. Delta expansion

Represent `edge(src, dst)` and `delta(src, dst)` as columns. Implement the
regular expansion step of transitive closure, then sort/deduplicate the
candidate delta. Keep the fixpoint scheduler on the CPU.

The indexed expansion join is implemented with serial Rust, Rayon, sparse
bitmap, and CUDA count-scan-emit paths. On DBLP, contiguous range postings beat
`hi_sparse_bitset` postings for this lookup-and-enumerate operation, while CUDA
becomes fastest at 2,048 delta rows. Binary candidate sorting and
deduplication are also implemented: a temporary packed `u64` gives serial
Rust, Rayon parallel sort, and CUDA radix-sort/unique implementations the same
lexicographic key before results return to canonical `u32` columns. Integration
with the recursive `FULL`/`DELTA`/`NEWT` lifecycle remains the next part of this
slice.

### 3. Placement policy

Measure a size- and shape-based threshold for CPU versus GPU operators. This
must compare whole pipeline time, including producer work and synchronization,
not only kernel duration.

The first measured filter policy is now recorded in
`FilterPlacementPolicy::MEASURED_GB10`. It selects CUDA from 131,072 rows for
CPU-produced input and from 32,768 rows for GPU-produced input. Parallel Rust
is currently only a CUDA-unavailable fallback from 8,388,608 rows; the filter
measurements did not show a parallel-CPU region before CUDA became faster.
This first isolated-operator matrix controls producer provenance but excludes
the producer's own duration; later pipeline placement measurements must include
producer work as stated above.

`JoinPlacementPolicy::MEASURED_GB10_DBLP` records the first expansion-join
threshold: serial Rust below 2,048 delta rows, then CUDA when available or
parallel Rust otherwise. Unlike filter placement, this threshold is only a
starting point because join fanout and key skew can change output cardinality
substantially for the same input row count.

`DistinctPlacementPolicy::MEASURED_GB10_DBLP` selects serial Rust below 32,768
candidate rows and CUDA from that point for either CPU- or GPU-produced input.
Without CUDA it selects parallel Rust from 131,072 rows. The full DBLP join
candidate relation shrank from 7,064,738 to 4,908,681 tuples in 7.829 ms on
CUDA, compared with 30.096 ms for parallel Rust.

## Engine structure

Sparkalog is organized as a relational engine with Datalog as a frontend:

```text
Datalog source
    -> validation, dependency analysis, and stratification
    -> semi-naive recursive relational plans
    -> backend-neutral relational operators
    -> native Rust or CUDA execution over canonical shared columns
```

The workspace reflects those boundaries:

- `sparkalog-storage`: managed columns, relation layouts, and derived indexes;
- `sparkalog-relational`: scans, filters, projections, joins, anti-joins,
  unions, distinct, sorting, reduction, and persistence;
- `sparkalog-execution`: native Rust and CUDA implementations, reusable
  workspaces, synchronization, and placement;
- `sparkalog-recursion`: strata, strongly connected components, and the
  `FULL`/`DELTA`/`NEWT` lifecycle;
- `sparkalog-datalog`: syntax, rule safety, dependency analysis, stratified
  negation, and lowering into recursive relational plans.

The relational layer does not choose a processor. A physical planner selects
serial Rust, parallel Rust, or CUDA for an operator using measured pipeline
costs. Canonical inputs and outputs keep the same representation whichever
implementation is selected.

Evaluation is bottom-up and semi-naive. Stratified negation lowers to an
anti-join against a completed relation in an earlier stratum. Negative cycles
are outside the initial semantics.

## Measurements that decide the design

- CPU producer -> GPU consumer latency against an explicit-copy baseline.
- The same test while the CPU prepares the next delta, to expose shared-fabric
  contention.
- sustained read, write, and read-modify-write bandwidth for CPU and GPU;
- NEON versus CUDA performance for filters and relation scans;
- GPU join/dedup throughput by cardinality, skew, and delta size;
- synchronization and kernel-launch cost for small deltas.

The result could be "keep a common canonical layout but use copied packed
GPU-side indexes for selected joins." That is still a useful result. The
project should not assume zero-copy wins until the full operator pipeline does.

## Skeleton compilation

The repository follows the narrow CUDA build pattern proven in `spark-infer`:
`sparkalog-execution/build.rs` invokes CUDA 13's `nvcc`, targets `sm_121`,
archives the native object, and links `cudart`. The native boundary currently
contains one small kernel in `sparkalog-execution/native/column_ops.cu`;
`sparkalog-storage` owns allocation and lifetime, while
`sparkalog-execution` owns streams and kernel submission.

```sh
cargo run --release
```

Set `CUDA_HOME` (or `CUDA_PATH`) when CUDA is not installed at
`/usr/local/cuda-13.0`.

[spark-memory]: https://docs.nvidia.com/dgx/dgx-spark-porting-guide/overview.html
[gdlog]: https://github.com/harp-lab/gdlog
