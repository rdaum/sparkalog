# Sparkalog

A DGX Spark experiment for running CPU and CUDA relational operators over one
canonical shared-memory relation representation.

Start with [the proposal](PROPOSAL.md). The current executable is a narrow
proof of the allocation and CUDA build boundary:

```sh
cargo run --release
```

It fills CUDA-managed memory from Rust on the CPU, increments the same memory
from an `sm_121` CUDA kernel, then verifies the result on the CPU. It is not a
Datalog engine yet.

## Workspace

The workspace follows the intended engine layers:

```text
sparkalog-datalog
    ↓ lowers safe, stratified rules
sparkalog-recursion
    ↓ orchestrates strata and semi-naive fixpoints
sparkalog-relational
    ↓ describes backend-neutral relational operations
sparkalog-storage ← sparkalog-execution
    canonical         native Rust/CUDA execution,
    shared columns    synchronization, and placement
```

- `sparkalog-storage` owns canonical CUDA-managed columns and relation shapes.
- `sparkalog-relational` defines the backend-neutral relational vocabulary.
- `sparkalog-execution` owns physical placement, CUDA streams, and kernels.
- `sparkalog-recursion` owns `FULL`/`DELTA`/`NEWT` fixpoint orchestration.
- `sparkalog-datalog` owns parsing, validation, stratification, and lowering.
- `sparkalog` is the executable integration boundary.

The upper layers are currently boundaries rather than an implemented Datalog
frontend. The shared-buffer proof now exercises storage and execution without
coupling either one to Datalog syntax.

## Filter crossover benchmark

The first paired relational operator has serial Rust, parallel Rust, and CUDA
implementations. Run the full CPU/GPU-provenance crossover matrix with:

```sh
cargo run --release --bin filter-crossover -- \
  --output benchmarks/filter-crossover.csv
```

The producer runs immediately before each timed sample but outside its timing
window. The measured interval begins when the filter is invoked and ends when
its compact row-ID selection is available to the CPU. Every sample validates
the selected cardinality. The full matrix covers 32 through 8,388,608 rows and
1%, 10%, 50%, and 90% selectivity. Use `--quick` for a short smoke run.
