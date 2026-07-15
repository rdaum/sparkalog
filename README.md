# Sparkalog

A GPU-accelerated Datalog engine written in Rust, designed around the unified
memory of systems like the NVIDIA DGX Spark. The engine parses and validates a
Souffle-style Datalog program, stratifies it, lowers it to backend-neutral
relational plans, and evaluates recursive fixpoints over CUDA-managed columns
with measured serial Rust / parallel Rust / CUDA placement policies.

This project is mostly AI-written. The code is the source of truth; this README
describes what is actually in the repository.

## Workspace

The workspace is split into engine layers that depend downwards:

```text
sparkalog-datalog
    ↓ parses, validates, stratifies, and lowers rules
sparkalog-recursion
    ↓ orchestrates strata and semi-naive fixpoints
sparkalog-relational
    ↓ backend-neutral relational algebra vocabulary
sparkalog-execution ──→ sparkalog-storage
    CUDA streams,         canonical CUDA-managed columns
    kernels, placement    and relation shapes
```

- `sparkalog-storage` owns canonical CUDA-managed `u32` columns, relation
  buffers, workspaces, and range/bitmap indexes. Its `load_binary_u32` loader
  reads a whitespace-separated two-column `u32` relation directly into managed
  columns in two passes, so it never builds a duplicate heap copy.
- `sparkalog-relational` re-exports storage types and defines the
  backend-neutral relational algebra vocabulary: `RelationVersion`
  (`Full`/`Delta`/`Newt`), `BinaryEqualityJoin`, `BinaryDistinct`,
  `SortedBinaryAntiJoin`, `SortedBinaryUnion`, and the general
  `RelationalClausePlan` / `GeneralProgramPlan` plan types.
- `sparkalog-execution` owns physical placement, CUDA streams, kernels (CUB
  filter, radix sort, merge, unique, compaction), and the
  `*PlacementPolicy::MEASURED_GB10*` thresholds derived from the crossover
  benchmarks. It links the CUDA kernels built from `native/` via `build.rs`.
- `sparkalog-recursion` owns the semi-naive fixpoint `RecursiveExecutor`, the
  `RelationStore` of `FULL`/`DELTA`/`NEWT` buffers, and `IterationPolicies`.
- `sparkalog-datalog` owns parsing with spans and recovery, interning of
  predicates and typed values into stable `u32` IDs, arity and rule-safety
  validation, stratification (rejecting negative recursion), SCC scheduling,
  and lowering into general clause plans and the binary recursive operator
  pipeline.
- `sparkalog` (the `sparkalog-cli` crate) is the executable integration
  boundary: the `sparkalog` binary plus the crossover and smoke binaries.

The Datalog frontend is executable end-to-end. A program is parsed with
spans and lexer/parser recovery, predicates and typed values are interned
into stable IDs, arity conflicts and unsafe negation are rejected, negative
dependencies inside a recursive SCC are rejected, strata/SCCs are scheduled,
and rules are lowered into backend-neutral relational clause plans.

At run time, `Database::run` first tries to obtain a `CudaStream`. If a binary
recursive lowering exists and a stream is available, it runs the measured
native Rust/CUDA operator pipeline (`ExecutionBackend::BinaryHybrid`). If no
binary lowering exists, or CUDA is unavailable, it falls back to the general
native Rust path (`ExecutionBackend::GeneralCpu`), which supports arbitrary
arity, constants, join chains, multiple recursive atoms, mutual recursion, and
stratified negation but executes on the CPU only. The binary path is only
produced for eligible binary recursive programs (see `lower_binary`); all other
programs use the general path.

## Datalog quick start

```prolog
edge('a, 'b).
edge('b, 'c).
edge('c, 'd).

path(x, y) :- edge(x, y).
path(x, z) :- path(x, y), edge(y, z).

.output path
```

```sh
cargo run --bin sparkalog -- check examples/transitive-closure.dl
cargo run --bin sparkalog -- explain examples/transitive-closure.dl
cargo run --bin sparkalog -- run examples/transitive-closure.dl
```

`check` validates and exits, `explain` prints the preferred backend and the
lowered plan, and `run` executes and prints every `.output` relation followed
by an SCC/iteration summary on stderr. See `examples/stratified-negation.dl`
for a stratified-negation program.

### Syntax

- Bare identifiers are rule-local variables.
- Prefix symbols with an apostrophe (`'a`), quote strings (`"a b"`), and write
  `u32` numbers in decimal.
- Safe negation uses `!atom(...)`; every variable in a negated atom must be
  bound by a positive body atom, and the head must be bound by a positive body
  atom.
- The parser accepts Souffle-style `.decl`, `.input`, and `.output` directives.
  `.decl name(attr:type, ...)` records arity for a predicate. `.input name`
  marks an already-declared predicate as EDB (it is an error to also derive it
  with a rule); it does **not** auto-load any file — facts are inserted through
  the `Database` API. `.output name` selects a defined predicate for emission.
  A predicate's arity is set by the first `.decl` or fact/rule that mentions
  it, and conflicting arities are rejected.

### Library API

The library-facing `Database` API (`sparkalog_datalog::Database`) supports:

- `load_program` parsing, resolving, and lowering a source string.
- `insert` stable fact insertion by predicate name.
- `load_delimited` parallel CSV/TSV ingestion via `parse_delimited_parallel`.
- `run` / `run_with_stream` repeatable execution, returning a `RunSummary`
  with the selected `ExecutionBackend` and per-SCC iteration counts.
- `query` and `outputs` decoding interned `u32` IDs back into `InternedValue`s.
- `explain` printing the preferred backend and lowered plan.
- `ProgramCatalog::write_to` / `read_from` for catalog persistence.

Join order in the general path is selected by `optimize_general` from relation
cardinality estimates while preferring atoms that share an already-bound
binding, and `live_bindings` records the bindings still needed after each
positive atom so dead bindings can be dropped mid-join.

Run the repeatable frontend ingestion/execution smoke benchmark with an
optional row count (default 100,000):

```sh
cargo run --release --bin datalog-frontend-smoke -- 100000
```

This loads a synthetic `edge` relation via `load_delimited`, runs
`copy(x, y) :- edge(x, y)`, and asserts the output row count matches the input.

## Crossover benchmarks

Each paired relational operator has serial Rust, parallel Rust (Rayon), and
CUDA+CUB implementations with a measured placement policy. Every backend's
output is checked exactly against serial Rust before any timing is recorded.
The producer fills the input immediately before each timed sample but outside
the timed interval; the interval covers operator dispatch, execution,
compaction, required synchronization, and the CPU-visible result length. Use
`--quick` for a short smoke run and `--output FILE.csv` to record results.

### Filter

```sh
cargo run --release --bin filter-crossover -- \
  --output benchmarks/filter-crossover.csv
```

The full matrix covers 32 through 8,388,608 rows and 1%, 10%, 50%, and 90%
selectivity, for both CPU- and GPU-produced inputs. See
`benchmarks/README.md` for the recorded findings and the stored
`FilterPlacementPolicy::MEASURED_GB10` thresholds.

## Real GDlog graph data

The reference checkout of GDlog stores its datasets as Git LFS pointers.
Materialize the default one-million-edge `com-dblp` graph without installing
Git LFS:

```sh
scripts/fetch-gdlog-data.sh
```

GitHub currently reports that LFS is disabled for the GDlog repository, so the
fetcher falls back to SNAP's original `com-dblp` archive and verifies its
1,049,866-edge cardinality and decompressed SHA-256 digest. If GDlog LFS
becomes available again, the same script verifies its declared byte length and
SHA-256 digest. Additional dataset directory names may be passed when their LFS
objects are available; use `--all` to materialize every LFS pointer under
`reference/gdlog/data`.

Load `com-dblp` directly into two canonical managed `u32` columns and compare
the filter backends with:

```sh
cargo run --release --bin graph-smoke
```

`graph-smoke` defaults to `reference/gdlog/data/com-dblp/edge.facts` and uses
`load_binary_u32`, which reads the file twice (once to count rows, once to
populate managed columns) and therefore does not build a duplicate heap copy.

### Indexed join, distinct, anti-join, union

```sh
cargo run --release --bin join-crossover -- \
  --output benchmarks/join-crossover.csv
```

`join-crossover` builds both a contiguous sorted range index and an
`hi_sparse_bitset` bitmap index over `edge.source`, then evaluates
`delta(x,y) ⋈ edge(y,z) → candidate(x,z)` with serial and parallel Rust for both
index representations, plus CUDA count-scan-emit over the managed range index.

```sh
cargo run --release --bin distinct-crossover -- \
  --output benchmarks/distinct-crossover.csv
```

`distinct-crossover` sorts and deduplicates those candidates. Each binary
tuple is temporarily packed into a managed `u64` so both Rust sorting and CUDA
radix sorting share the same lexicographic key, then unpacked into canonical
managed `u32` columns. Inputs are regenerated immediately before each timed
sample with either parallel Rust or CUDA.

```sh
cargo run --release --bin anti-join-crossover -- \
  --output benchmarks/anti-join-crossover.csv
```

`anti-join-crossover` evaluates the real first semi-naive subtraction step
`NEWT = distinct(delta ⋈ edge) − distinct(edge)`. Serial Rust uses a sorted
merge, parallel Rust uses merge-count and merge-emit passes over left chunks,
and CUDA uses parallel binary search of `FULL` plus CUB flagged selection and
compaction.

```sh
cargo run --release --bin union-crossover -- \
  --output benchmarks/union-crossover.csv
```

`union-crossover` merges `NEWT` into the sorted `FULL` relation. Serial Rust
performs a sorted merge, parallel Rust uses merge-path partitions, and CUDA
uses CUB device-wide merge followed by unique compaction. Recursive rule
runtimes swap their anti-join and union outputs into relation-store state, so
rounds rotate canonical `FULL`, `DELTA`, and `NEWT` buffers without copying.

The recorded timings and the stored `*PlacementPolicy::MEASURED_GB10_DBLP`
thresholds for each operator are summarized in `benchmarks/README.md`.

## Transitive closure fixpoint

Run the terminating driver over a generated chain graph with automatic,
CPU-only, or GPU-only placement:

```sh
cargo run --release --bin tc-fixpoint -- --vertices 128 --backend auto
```

`tc-fixpoint` compiles `transitive_closure_scc(path, edge)`, evaluates it with
`RecursiveExecutor`, and verifies the chain's exact
`vertices * (vertices - 1) / 2` closure before reporting timings. The executor
caches the right-side join index by relation, column, and version; evaluates
every rule in an SCC before applying any updates; and stops when every `DELTA`
is empty. `--max-iterations` (default 1024) provides an explicit safety bound.
`--backend auto|cpu|gpu` selects placement, and `--vertices` accepts 2–4096.