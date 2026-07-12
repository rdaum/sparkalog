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

The Datalog frontend is now executable. It parses source with spans and
recovery, interns predicates and typed values into stable `u32` IDs, validates
arity and rule safety, rejects negative recursion, schedules strata/SCCs, and
lowers rules into backend-neutral relational clause plans. The general native
Rust path supports arbitrary arity, constants, join chains, multiple recursive
atoms, mutual recursion, and stratified negation. Eligible binary recursive
programs use the measured native Rust/CUDA operator pipeline over canonical
managed columns.

## Datalog quick start

```prolog
edge('a, 'b).
edge('b, 'c).

path(x, y) :- edge(x, y).
path(x, z) :- path(x, y), edge(y, z).

.output path
```

```sh
cargo run --bin sparkalog -- check examples/transitive-closure.dl
cargo run --bin sparkalog -- explain examples/transitive-closure.dl
cargo run --bin sparkalog -- run examples/transitive-closure.dl
```

Bare terms are rule-local variables. Prefix symbols with an apostrophe, quote
strings, and write `u32` numbers in decimal. Safe negation uses `!atom(...)`.
The parser also accepts practical Souffle-style `.decl`, `.input`, and
`.output` directives; declarations are syntax-compatible hints while resolved
predicate arity is inferred and checked from facts and rules.

The library-facing `Database` API supports loading and checking a program,
stable fact insertion, repeatable execution, decoded queries, parallel CSV/TSV
ingestion, explain output, and catalog persistence through
`ProgramCatalog::write_to`/`read_from`. Join order is selected from relation
cardinality estimates while preferring connected atoms, and dead bindings are
released after their last planned use.

Run the repeatable frontend ingestion/execution smoke benchmark with an
optional row count:

```sh
cargo run --release --bin datalog-frontend-smoke -- 100000
```

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

## Real GDlog graph data

The reference checkout stores its datasets as Git LFS pointers. Materialize
the default one-million-edge `com-dblp` graph without installing Git LFS:

```sh
scripts/fetch-gdlog-data.sh
```

GitHub currently reports that LFS is disabled for the GDlog repository, so the
fetcher falls back to SNAP's original `com-dblp` archive and verifies its
1,049,866-edge cardinality and decompressed SHA-256 digest. If GDlog LFS becomes
available again, the same script verifies its declared byte length and SHA-256
digest. Additional dataset directory names may be passed when their LFS objects
are available.

Load `com-dblp` directly into two canonical managed `u32` columns and compare
all filter backends with:

```sh
cargo run --release --bin graph-smoke
```

The loader reads the TSV twice: once to determine relation cardinality and once
to populate managed columns. It therefore does not build a duplicate heap copy
of the graph.

Run the real indexed expansion join across progressively larger DBLP deltas:

```sh
cargo run --release --bin join-crossover -- \
  --output benchmarks/join-crossover.csv
```

This builds both sparse range and `hi_sparse_bitset` bitmap indexes over
`edge.source`, then evaluates `delta(x,y) ⋈ edge(y,z) → candidate(x,z)` with
serial and parallel Rust for both index representations, plus CUDA
count-scan-emit over the managed range index. Every backend must produce
identical ordered output columns before its timings are recorded. Use
`--quick` for a three-cardinality smoke run.

Sort and deduplicate those real join candidates with all three execution
backends using:

```sh
cargo run --release --bin distinct-crossover -- \
  --output benchmarks/distinct-crossover.csv
```

The operator temporarily packs each binary tuple into a managed `u64`, giving
both Rust sorting and CUDA radix sorting the same lexicographic key. Results
are unpacked into canonical managed `u32` columns. The benchmark regenerates
each input immediately before timing with either the parallel Rust or CUDA
join, and validates identical sorted output from serial Rust, parallel Rust,
and CUDA. Use `--quick` for a short smoke run.

Subtract the sorted `FULL` edge relation from those distinct candidates with:

```sh
cargo run --release --bin anti-join-crossover -- \
  --output benchmarks/anti-join-crossover.csv
```

This evaluates the real first-step operation
`NEWT = distinct(delta ⋈ edge) − distinct(edge)`. Serial Rust uses a sorted
merge, parallel Rust uses merge-count and merge-emit passes over left chunks,
and CUDA uses parallel membership marking plus CUB stable compaction. Inputs
are regenerated immediately before every timed sample with either the native
or CUDA join/distinct pipeline, and all outputs are checked exactly.

Merge `NEWT` into the sorted `FULL` relation with:

```sh
cargo run --release --bin union-crossover -- \
  --output benchmarks/union-crossover.csv
```

Serial Rust performs a sorted merge, parallel Rust uses merge-path partitions,
and CUDA uses CUB device merge followed by unique compaction. Recursive rule
runtimes swap their anti-join and union outputs into relation-store state, so
rounds rotate canonical `FULL`, `DELTA`, and `NEWT` buffers without copying.

Run the terminating driver over a generated chain graph with automatic,
CPU-only, or GPU-only placement:

```sh
cargo run --release --bin tc-fixpoint -- --vertices 128 --backend auto
```

The executor caches right-side indexes by relation version, evaluates every
rule in an SCC before applying any updates, and stops when every `DELTA` is
empty. `--max-iterations` provides an explicit safety bound. The harness
verifies the chain's exact
`vertices * (vertices - 1) / 2` transitive closure before reporting timings.
