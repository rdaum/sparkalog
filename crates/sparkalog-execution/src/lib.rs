//! Physical execution, synchronization, and CPU/GPU placement.

mod placement;

pub use placement::{
    AntiJoinPlacementContext, AntiJoinPlacementPolicy, DistinctPlacementContext,
    DistinctPlacementPolicy, FilterPlacementContext, FilterPlacementPolicy, InputProvenance,
    JoinPlacementContext, JoinPlacementPolicy, Placement, UnionPlacementContext,
    UnionPlacementPolicy,
};

use rayon::prelude::*;
use sparkalog_relational::{
    BinaryDistinct, BinaryEqualityJoin, JoinInput, JoinProjection, SortedBinaryAntiJoin,
    SortedBinaryUnion, U32Predicate,
};
use sparkalog_storage::{
    AntiJoinWorkspace, Column, DistinctWorkspace, JoinWorkspace, ManagedBuffer, OperatorWorkspace,
    RelationBuffer, RelationView, U32BitmapIndex, U32RangeIndex, UnionWorkspace,
};
use std::ffi::c_void;
use std::fmt;
use std::ptr::NonNull;

const CUDA_SUCCESS: i32 = 0;
const CUDA_STREAM_NON_BLOCKING: u32 = 1;

unsafe extern "C" {
    fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: u32) -> i32;
    fn cudaStreamDestroy(stream: *mut c_void) -> i32;
    fn sparkalog_add_one_i32(data: *mut i32, len: usize, stream: *mut c_void) -> i32;
    fn sparkalog_stream_synchronize(stream: *mut c_void) -> i32;
    fn sparkalog_filter_u32_temporary_bytes(
        flags: *const u32,
        output: *mut u32,
        count: *mut u32,
        len: usize,
        temporary_bytes: *mut usize,
        stream: *mut c_void,
    ) -> i32;
    fn sparkalog_filter_u32(
        input: *const u32,
        len: usize,
        operation: u32,
        operand: u32,
        flags: *mut u32,
        output: *mut u32,
        count: *mut u32,
        temporary: *mut c_void,
        temporary_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    fn sparkalog_fill_mod_u32(
        output: *mut u32,
        len: usize,
        modulus: u32,
        stream: *mut c_void,
    ) -> i32;
    fn sparkalog_join_u32_temporary_bytes(
        counts: *const u64,
        offsets: *mut u64,
        left_rows: usize,
        temporary_bytes: *mut usize,
        stream: *mut c_void,
    ) -> i32;
    fn sparkalog_join_u32_count(
        left_keys: *const u32,
        left_rows: usize,
        index_keys: *const u32,
        index_starts: *const u32,
        unique_keys: usize,
        counts: *mut u64,
        offsets: *mut u64,
        total: *mut u64,
        temporary: *mut c_void,
        temporary_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    fn sparkalog_join_u32_emit(
        left_keys: *const u32,
        left_rows: usize,
        index_keys: *const u32,
        index_starts: *const u32,
        index_rows: *const u32,
        unique_keys: usize,
        projection0: *const u32,
        projection0_side: u32,
        projection1: *const u32,
        projection1_side: u32,
        offsets: *const u64,
        output0: *mut u32,
        output1: *mut u32,
        stream: *mut c_void,
    ) -> i32;
    fn sparkalog_distinct_u32_temporary_bytes(
        packed: *mut u64,
        scratch: *mut u64,
        unique_rows: *mut u64,
        rows: usize,
        temporary_bytes: *mut usize,
        stream: *mut c_void,
    ) -> i32;
    fn sparkalog_distinct_u32(
        first: *const u32,
        second: *const u32,
        rows: usize,
        packed: *mut u64,
        scratch: *mut u64,
        unique_rows: *mut u64,
        output_first: *mut u32,
        output_second: *mut u32,
        temporary: *mut c_void,
        temporary_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    fn sparkalog_anti_join_u32_temporary_bytes(
        flags: *const u32,
        selected: *mut u32,
        selected_rows: *mut u32,
        left_rows: usize,
        temporary_bytes: *mut usize,
        stream: *mut c_void,
    ) -> i32;
    fn sparkalog_anti_join_u32(
        left_first: *const u32,
        left_second: *const u32,
        left_rows: usize,
        right_first: *const u32,
        right_second: *const u32,
        right_rows: usize,
        flags: *mut u32,
        selected: *mut u32,
        selected_rows: *mut u32,
        output_first: *mut u32,
        output_second: *mut u32,
        temporary: *mut c_void,
        temporary_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    fn sparkalog_union_u32_temporary_bytes(
        left: *const u64,
        left_rows: usize,
        right: *const u64,
        right_rows: usize,
        merged: *mut u64,
        unique: *mut u64,
        unique_rows: *mut u64,
        temporary_bytes: *mut usize,
        stream: *mut c_void,
    ) -> i32;
    fn sparkalog_union_u32(
        left_first: *const u32,
        left_second: *const u32,
        left_rows: usize,
        right_first: *const u32,
        right_second: *const u32,
        right_rows: usize,
        left: *mut u64,
        right: *mut u64,
        merged: *mut u64,
        unique: *mut u64,
        unique_rows: *mut u64,
        output_first: *mut u32,
        output_second: *mut u32,
        temporary: *mut c_void,
        temporary_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Cuda {
        operation: &'static str,
        code: i32,
    },
    Storage(sparkalog_storage::Error),
    TooManyRows(usize),
    MissingColumn {
        input: &'static str,
        column: u32,
    },
    IndexSourceMismatch {
        relation_rows: usize,
        index_rows: usize,
    },
    OutputTooLarge(u64),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda { operation, code } => {
                write!(formatter, "{operation} failed with CUDA error {code}")
            }
            Self::Storage(error) => error.fmt(formatter),
            Self::TooManyRows(rows) => {
                write!(
                    formatter,
                    "{rows} rows cannot be represented by u32 row IDs"
                )
            }
            Self::MissingColumn { input, column } => {
                write!(formatter, "{input} relation has no column {column}")
            }
            Self::IndexSourceMismatch {
                relation_rows,
                index_rows,
            } => write!(
                formatter,
                "right relation has {relation_rows} rows but its index has {index_rows}"
            ),
            Self::OutputTooLarge(rows) => {
                write!(
                    formatter,
                    "join output of {rows} rows exceeds addressable memory"
                )
            }
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

impl From<sparkalog_storage::Error> for Error {
    fn from(error: sparkalog_storage::Error) -> Self {
        Self::Storage(error)
    }
}

fn cuda_result(operation: &'static str, code: i32) -> Result<()> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Error::Cuda { operation, code })
    }
}

pub struct CudaStream {
    raw: NonNull<c_void>,
}

impl CudaStream {
    pub fn new() -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: CUDA writes a stream handle to `raw` on success.
        unsafe {
            cuda_result(
                "cudaStreamCreateWithFlags",
                cudaStreamCreateWithFlags(&mut raw, CUDA_STREAM_NON_BLOCKING),
            )?;
        }
        let raw = NonNull::new(raw).expect("successful CUDA stream creation returned null");
        Ok(Self { raw })
    }

    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: `raw` remains valid until this stream is dropped.
        unsafe {
            cuda_result(
                "cudaStreamSynchronize",
                sparkalog_stream_synchronize(self.raw.as_ptr()),
            )
        }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        // SAFETY: `raw` was created by CUDA and is owned by this wrapper.
        unsafe {
            let _ = cudaStreamDestroy(self.raw.as_ptr());
        }
    }
}

/// A CUDA submission which retains exclusive access to its managed allocation
/// until the stream reaches the operation.
#[must_use = "dropping a pending CUDA operation waits for its stream"]
pub struct Pending<'a> {
    _column: &'a mut ManagedBuffer<i32>,
    stream: &'a CudaStream,
    completed: bool,
}

impl Pending<'_> {
    pub fn wait(mut self) -> Result<()> {
        self.stream.synchronize()?;
        self.completed = true;
        Ok(())
    }
}

impl Drop for Pending<'_> {
    fn drop(&mut self) {
        if !self.completed {
            let _ = self.stream.synchronize();
        }
    }
}

/// Enqueue the narrow CUDA operation retained as the workspace smoke test.
pub fn add_one_i32<'a>(
    column: &'a mut ManagedBuffer<i32>,
    stream: &'a CudaStream,
) -> Result<Pending<'a>> {
    // SAFETY: `column` is CUDA-managed and the mutable borrow prevents other
    // host access for the duration of submission. The caller synchronizes
    // before accessing it again.
    unsafe {
        cuda_result(
            "sparkalog_add_one_i32",
            sparkalog_add_one_i32(column.as_mut_ptr(), column.len(), stream.raw.as_ptr()),
        )?;
    }
    Ok(Pending {
        _column: column,
        stream,
        completed: false,
    })
}

fn check_row_count(rows: usize) -> Result<()> {
    if rows > u32::MAX as usize {
        Err(Error::TooManyRows(rows))
    } else {
        Ok(())
    }
}

fn distinct_inputs<'a>(
    input: RelationView<'a>,
    plan: BinaryDistinct,
) -> Result<(&'a [u32], &'a [u32])> {
    check_row_count(input.len())?;
    Ok((
        relation_column(input, "input", plan.columns[0])?,
        relation_column(input, "input", plan.columns[1])?,
    ))
}

fn pack_tuple(first: u32, second: u32) -> u64 {
    (u64::from(first) << 32) | u64::from(second)
}

fn deduplicate_sorted(packed: &mut [u64]) -> usize {
    let mut unique = 0;
    for read in 0..packed.len() {
        if unique == 0 || packed[read] != packed[unique - 1] {
            packed[unique] = packed[read];
            unique += 1;
        }
    }
    unique
}

fn unpack_distinct_output(
    workspace: &mut DistinctWorkspace,
    unique: usize,
    parallel: bool,
) -> Result<()> {
    let (output, packed) = workspace.cpu_output_parts();
    unpack_packed_output(output, packed.as_slice(), unique, parallel)
}

fn unpack_packed_output(
    output: &mut RelationBuffer,
    packed: &[u64],
    unique: usize,
    parallel: bool,
) -> Result<()> {
    let packed = &packed[..unique];
    let (first, second) = output.columns_mut().split_at_mut(1);
    let output0 = &mut first[0].as_mut_slice()[..unique];
    let output1 = &mut second[0].as_mut_slice()[..unique];
    if parallel {
        output0
            .par_iter_mut()
            .zip(output1.par_iter_mut())
            .zip(packed.par_iter())
            .for_each(|((first, second), &tuple)| {
                *first = (tuple >> 32) as u32;
                *second = tuple as u32;
            });
    } else {
        for ((first, second), &tuple) in output0.iter_mut().zip(output1).zip(packed) {
            *first = (tuple >> 32) as u32;
            *second = tuple as u32;
        }
    }
    output.set_len(unique)?;
    Ok(())
}

/// Sort and deduplicate binary tuples on the calling thread.
pub fn distinct_cpu_serial(
    input: RelationView<'_>,
    plan: BinaryDistinct,
    workspace: &mut DistinctWorkspace,
) -> Result<()> {
    let (first, second) = distinct_inputs(input, plan)?;
    workspace.reserve_rows(input.len())?;
    workspace.output_mut().clear();
    let packed = &mut workspace.packed_mut().as_mut_slice()[..input.len()];
    for ((output, &first), &second) in packed.iter_mut().zip(first).zip(second) {
        *output = pack_tuple(first, second);
    }
    packed.sort_unstable();
    let unique = deduplicate_sorted(packed);
    unpack_distinct_output(workspace, unique, false)
}

/// Sort and deduplicate binary tuples using the persistent Rayon pool.
pub fn distinct_cpu_parallel(
    input: RelationView<'_>,
    plan: BinaryDistinct,
    workspace: &mut DistinctWorkspace,
) -> Result<()> {
    let (first, second) = distinct_inputs(input, plan)?;
    if input.is_empty() || rayon::current_num_threads() == 1 {
        return distinct_cpu_serial(input, plan, workspace);
    }
    workspace.reserve_rows(input.len())?;
    workspace.output_mut().clear();
    let packed = &mut workspace.packed_mut().as_mut_slice()[..input.len()];
    packed.par_iter_mut().enumerate().for_each(|(row, output)| {
        *output = pack_tuple(first[row], second[row]);
    });
    packed.par_sort_unstable();
    let unique = deduplicate_sorted(packed);
    unpack_distinct_output(workspace, unique, true)
}

#[must_use = "dropping a pending CUDA distinct waits for its stream"]
pub struct PendingDistinct<'a> {
    _input: RelationView<'a>,
    workspace: &'a mut DistinctWorkspace,
    stream: &'a CudaStream,
    completed: bool,
}

impl PendingDistinct<'_> {
    pub fn wait(mut self) -> Result<()> {
        self.stream.synchronize()?;
        let unique_u64 = self.workspace.count().as_slice()[0];
        let unique = usize::try_from(unique_u64).map_err(|_| Error::OutputTooLarge(unique_u64))?;
        self.workspace.output_mut().set_len(unique)?;
        self.completed = true;
        Ok(())
    }
}

impl Drop for PendingDistinct<'_> {
    fn drop(&mut self) {
        if !self.completed {
            let _ = self.stream.synchronize();
        }
    }
}

/// Sort and deduplicate binary tuples with CUDA radix sort and unique
/// compaction over reusable managed buffers.
pub fn distinct_cuda<'a>(
    input: RelationView<'a>,
    plan: BinaryDistinct,
    workspace: &'a mut DistinctWorkspace,
    stream: &'a CudaStream,
) -> Result<PendingDistinct<'a>> {
    let (first, second) = distinct_inputs(input, plan)?;
    workspace.reserve_rows(input.len())?;
    workspace.output_mut().clear();
    workspace.count_mut().as_mut_slice()[0] = 0;

    let mut temporary_bytes = 0;
    {
        let (_, packed, scratch, count, _) = workspace.cuda_parts();
        // SAFETY: the managed buffers remain live and a null temporary pointer
        // asks CUB only for its maximum sort/unique workspace requirement.
        unsafe {
            cuda_result(
                "sparkalog_distinct_u32_temporary_bytes",
                sparkalog_distinct_u32_temporary_bytes(
                    packed.as_mut_ptr(),
                    scratch.as_mut_ptr(),
                    count.as_mut_ptr(),
                    input.len(),
                    &mut temporary_bytes,
                    stream.raw.as_ptr(),
                ),
            )?;
        }
    }
    workspace.reserve_temporary_bytes(temporary_bytes)?;

    let launch_result = {
        let (output, packed, scratch, count, temporary) = workspace.cuda_parts();
        let (output0, output1) = output.columns_mut().split_at_mut(1);
        // SAFETY: all pointers refer to live managed allocations. The pending
        // result retains exclusive workspace access until the stream completes.
        unsafe {
            cuda_result(
                "sparkalog_distinct_u32",
                sparkalog_distinct_u32(
                    first.as_ptr(),
                    second.as_ptr(),
                    input.len(),
                    packed.as_mut_ptr(),
                    scratch.as_mut_ptr(),
                    count.as_mut_ptr(),
                    output0[0].as_mut_ptr(),
                    output1[0].as_mut_ptr(),
                    temporary.as_mut_ptr().cast(),
                    temporary_bytes,
                    stream.raw.as_ptr(),
                ),
            )
        }
    };
    if let Err(error) = launch_result {
        let _ = stream.synchronize();
        return Err(error);
    }
    Ok(PendingDistinct {
        _input: input,
        workspace,
        stream,
        completed: false,
    })
}

/// Execute binary distinct using the backend selected by a measured policy.
pub fn distinct_auto(
    input: RelationView<'_>,
    plan: BinaryDistinct,
    input_provenance: InputProvenance,
    workspace: &mut DistinctWorkspace,
    stream: Option<&CudaStream>,
    policy: DistinctPlacementPolicy,
) -> Result<Placement> {
    let placement = policy.place(DistinctPlacementContext {
        rows: input.len(),
        input_provenance,
        gpu_available: stream.is_some(),
    });
    match placement {
        Placement::CpuSerial => distinct_cpu_serial(input, plan, workspace)?,
        Placement::CpuParallel => distinct_cpu_parallel(input, plan, workspace)?,
        Placement::Gpu => {
            let stream = stream.expect("GPU placement requires an available CUDA stream");
            distinct_cuda(input, plan, workspace, stream)?.wait()?;
        }
    }
    Ok(placement)
}

type BinaryColumns<'a> = (&'a [u32], &'a [u32]);

fn anti_join_inputs<'a>(
    left: RelationView<'a>,
    right: RelationView<'a>,
    plan: SortedBinaryAntiJoin,
) -> Result<(BinaryColumns<'a>, BinaryColumns<'a>)> {
    check_row_count(left.len())?;
    check_row_count(right.len())?;
    Ok((
        (
            relation_column(left, "left", plan.left[0])?,
            relation_column(left, "left", plan.left[1])?,
        ),
        (
            relation_column(right, "right", plan.right[0])?,
            relation_column(right, "right", plan.right[1])?,
        ),
    ))
}

fn compare_pair(
    first: u32,
    second: u32,
    other_first: u32,
    other_second: u32,
) -> std::cmp::Ordering {
    (first, second).cmp(&(other_first, other_second))
}

fn lower_bound_pair(columns: BinaryColumns<'_>, first: u32, second: u32) -> usize {
    let (right_first, right_second) = columns;
    let mut low = 0;
    let mut high = right_first.len();
    while low < high {
        let middle = low + (high - low) / 2;
        match compare_pair(right_first[middle], right_second[middle], first, second) {
            std::cmp::Ordering::Less => low = middle + 1,
            std::cmp::Ordering::Greater | std::cmp::Ordering::Equal => high = middle,
        }
    }
    low
}

fn anti_join_chunk_count(
    left: BinaryColumns<'_>,
    right: BinaryColumns<'_>,
    row_start: usize,
    row_end: usize,
) -> u64 {
    if row_start == row_end {
        return 0;
    }
    let (left_first, left_second) = left;
    let (right_first, right_second) = right;
    let mut right_row = lower_bound_pair(right, left_first[row_start], left_second[row_start]);
    let mut selected = 0_u64;
    for left_row in row_start..row_end {
        while right_row < right_first.len()
            && compare_pair(
                right_first[right_row],
                right_second[right_row],
                left_first[left_row],
                left_second[left_row],
            ) == std::cmp::Ordering::Less
        {
            right_row += 1;
        }
        let matched = right_row < right_first.len()
            && right_first[right_row] == left_first[left_row]
            && right_second[right_row] == left_second[left_row];
        selected += u64::from(!matched);
    }
    selected
}

fn anti_join_chunk_emit(
    left: BinaryColumns<'_>,
    right: BinaryColumns<'_>,
    row_start: usize,
    row_end: usize,
    output_first: &mut [u32],
    output_second: &mut [u32],
) {
    if row_start == row_end {
        return;
    }
    let (left_first, left_second) = left;
    let (right_first, right_second) = right;
    let mut right_row = lower_bound_pair(right, left_first[row_start], left_second[row_start]);
    let mut selected = 0;
    for left_row in row_start..row_end {
        while right_row < right_first.len()
            && compare_pair(
                right_first[right_row],
                right_second[right_row],
                left_first[left_row],
                left_second[left_row],
            ) == std::cmp::Ordering::Less
        {
            right_row += 1;
        }
        let matched = right_row < right_first.len()
            && right_first[right_row] == left_first[left_row]
            && right_second[right_row] == left_second[left_row];
        if !matched {
            output_first[selected] = left_first[left_row];
            output_second[selected] = left_second[left_row];
            selected += 1;
        }
    }
}

/// Subtract one sorted binary relation from another with a linear merge.
pub fn anti_join_cpu_serial(
    left: RelationView<'_>,
    right: RelationView<'_>,
    plan: SortedBinaryAntiJoin,
    workspace: &mut AntiJoinWorkspace,
) -> Result<()> {
    let ((left_first, left_second), (right_first, right_second)) =
        anti_join_inputs(left, right, plan)?;
    workspace.reserve_rows(left.len())?;
    workspace.output_mut().clear();
    let output = workspace.output_mut();
    let (first, second) = output.columns_mut().split_at_mut(1);
    let output_first = first[0].as_mut_slice();
    let output_second = second[0].as_mut_slice();
    let mut right_row = 0;
    let mut selected = 0;
    for left_row in 0..left.len() {
        while right_row < right.len()
            && compare_pair(
                right_first[right_row],
                right_second[right_row],
                left_first[left_row],
                left_second[left_row],
            ) == std::cmp::Ordering::Less
        {
            right_row += 1;
        }
        let matched = right_row < right.len()
            && right_first[right_row] == left_first[left_row]
            && right_second[right_row] == left_second[left_row];
        if !matched {
            output_first[selected] = left_first[left_row];
            output_second[selected] = left_second[left_row];
            selected += 1;
        }
    }
    output.set_len(selected)?;
    Ok(())
}

/// Subtract sorted binary relations with parallel merge-count and stable
/// merge-emit passes over disjoint left chunks.
pub fn anti_join_cpu_parallel(
    left: RelationView<'_>,
    right: RelationView<'_>,
    plan: SortedBinaryAntiJoin,
    workspace: &mut AntiJoinWorkspace,
) -> Result<()> {
    let (left_columns, right_columns) = anti_join_inputs(left, right, plan)?;
    if left.is_empty() || rayon::current_num_threads() == 1 {
        return anti_join_cpu_serial(left, right, plan, workspace);
    }
    let left_len = left.len();
    let workers = rayon::current_num_threads().min(left_len);
    let chunk_size = left_len.div_ceil(workers);
    let chunk_count = left_len.div_ceil(chunk_size);
    workspace.reserve_rows(left_len)?;
    workspace.reserve_chunks(chunk_count)?;
    workspace.output_mut().clear();

    let (output, chunk_offsets) = workspace.cpu_parallel_parts();
    let chunk_offsets = &mut chunk_offsets.as_mut_slice()[..chunk_count];
    chunk_offsets
        .par_iter_mut()
        .enumerate()
        .for_each(|(chunk, count)| {
            let row_start = chunk * chunk_size;
            let row_end = (row_start + chunk_size).min(left_len);
            *count = anti_join_chunk_count(left_columns, right_columns, row_start, row_end);
        });
    let mut total = 0_u64;
    for offset in chunk_offsets.iter_mut() {
        let count = *offset;
        *offset = total;
        total += count;
    }
    let total = usize::try_from(total).map_err(|_| Error::OutputTooLarge(total))?;
    let (first, second) = output.columns_mut().split_at_mut(1);
    let mut remaining_first = &mut first[0].as_mut_slice()[..total];
    let mut remaining_second = &mut second[0].as_mut_slice()[..total];
    let chunk_offsets = &*chunk_offsets;

    rayon::scope(|scope| {
        for chunk in 0..chunk_count {
            let row_start = chunk * chunk_size;
            let row_end = (row_start + chunk_size).min(left_len);
            let output_start = chunk_offsets[chunk] as usize;
            let output_end = if chunk + 1 == chunk_count {
                total
            } else {
                chunk_offsets[chunk + 1] as usize
            };
            let output_len = output_end - output_start;
            let (output_first, next_first) = remaining_first.split_at_mut(output_len);
            let (output_second, next_second) = remaining_second.split_at_mut(output_len);
            remaining_first = next_first;
            remaining_second = next_second;
            scope.spawn(move |_| {
                anti_join_chunk_emit(
                    left_columns,
                    right_columns,
                    row_start,
                    row_end,
                    output_first,
                    output_second,
                );
            });
        }
    });
    output.set_len(total)?;
    Ok(())
}

#[must_use = "dropping a pending CUDA anti-join waits for its stream"]
pub struct PendingAntiJoin<'a> {
    _left: RelationView<'a>,
    _right: RelationView<'a>,
    workspace: &'a mut AntiJoinWorkspace,
    stream: &'a CudaStream,
    completed: bool,
}

impl PendingAntiJoin<'_> {
    pub fn wait(mut self) -> Result<()> {
        self.stream.synchronize()?;
        let selected = self.workspace.count().as_slice()[0] as usize;
        self.workspace.output_mut().set_len(selected)?;
        self.completed = true;
        Ok(())
    }
}

impl Drop for PendingAntiJoin<'_> {
    fn drop(&mut self) {
        if !self.completed {
            let _ = self.stream.synchronize();
        }
    }
}

/// Subtract sorted binary relations with CUDA membership marking and stable
/// CUB compaction.
pub fn anti_join_cuda<'a>(
    left: RelationView<'a>,
    right: RelationView<'a>,
    plan: SortedBinaryAntiJoin,
    workspace: &'a mut AntiJoinWorkspace,
    stream: &'a CudaStream,
) -> Result<PendingAntiJoin<'a>> {
    let ((left_first, left_second), (right_first, right_second)) =
        anti_join_inputs(left, right, plan)?;
    workspace.reserve_rows(left.len())?;
    workspace.output_mut().clear();
    workspace.count_mut().as_mut_slice()[0] = 0;

    let mut temporary_bytes = 0;
    {
        let (_, selection, flags, count, _) = workspace.cuda_parts();
        // SAFETY: all pointers refer to live managed buffers. A null temporary
        // pointer asks CUB only for its compaction workspace requirement.
        unsafe {
            cuda_result(
                "sparkalog_anti_join_u32_temporary_bytes",
                sparkalog_anti_join_u32_temporary_bytes(
                    flags.as_ptr(),
                    selection.storage_mut().as_mut_ptr(),
                    count.as_mut_ptr(),
                    left.len(),
                    &mut temporary_bytes,
                    stream.raw.as_ptr(),
                ),
            )?;
        }
    }
    workspace.reserve_temporary_bytes(temporary_bytes)?;

    let launch_result = {
        let (output, selection, flags, count, temporary) = workspace.cuda_parts();
        let (output0, output1) = output.columns_mut().split_at_mut(1);
        // SAFETY: the pending result retains both inputs and exclusive output
        // workspace access until the stream completes.
        unsafe {
            cuda_result(
                "sparkalog_anti_join_u32",
                sparkalog_anti_join_u32(
                    left_first.as_ptr(),
                    left_second.as_ptr(),
                    left.len(),
                    right_first.as_ptr(),
                    right_second.as_ptr(),
                    right.len(),
                    flags.as_mut_ptr(),
                    selection.storage_mut().as_mut_ptr(),
                    count.as_mut_ptr(),
                    output0[0].as_mut_ptr(),
                    output1[0].as_mut_ptr(),
                    temporary.as_mut_ptr().cast(),
                    temporary_bytes,
                    stream.raw.as_ptr(),
                ),
            )
        }
    };
    if let Err(error) = launch_result {
        let _ = stream.synchronize();
        return Err(error);
    }
    Ok(PendingAntiJoin {
        _left: left,
        _right: right,
        workspace,
        stream,
        completed: false,
    })
}

/// Execute sorted binary anti-join using a measured placement policy.
pub fn anti_join_auto(
    left: RelationView<'_>,
    right: RelationView<'_>,
    plan: SortedBinaryAntiJoin,
    input_provenance: InputProvenance,
    workspace: &mut AntiJoinWorkspace,
    stream: Option<&CudaStream>,
    policy: AntiJoinPlacementPolicy,
) -> Result<Placement> {
    let placement = policy.place(AntiJoinPlacementContext {
        left_rows: left.len(),
        input_provenance,
        gpu_available: stream.is_some(),
    });
    match placement {
        Placement::CpuSerial => anti_join_cpu_serial(left, right, plan, workspace)?,
        Placement::CpuParallel => anti_join_cpu_parallel(left, right, plan, workspace)?,
        Placement::Gpu => {
            let stream = stream.expect("GPU placement requires an available CUDA stream");
            anti_join_cuda(left, right, plan, workspace, stream)?.wait()?;
        }
    }
    Ok(placement)
}

fn union_inputs<'a>(
    left: RelationView<'a>,
    right: RelationView<'a>,
    plan: SortedBinaryUnion,
) -> Result<(BinaryColumns<'a>, BinaryColumns<'a>)> {
    check_row_count(left.len())?;
    check_row_count(right.len())?;
    Ok((
        (
            relation_column(left, "left", plan.left[0])?,
            relation_column(left, "left", plan.left[1])?,
        ),
        (
            relation_column(right, "right", plan.right[0])?,
            relation_column(right, "right", plan.right[1])?,
        ),
    ))
}

/// Merge and deduplicate two sorted binary relations on the calling thread.
pub fn union_cpu_serial(
    left: RelationView<'_>,
    right: RelationView<'_>,
    plan: SortedBinaryUnion,
    workspace: &mut UnionWorkspace,
) -> Result<()> {
    let ((left_first, left_second), (right_first, right_second)) = union_inputs(left, right, plan)?;
    workspace.reserve_rows(left.len(), right.len())?;
    workspace.output_mut().clear();
    let output = workspace.output_mut();
    let (first, second) = output.columns_mut().split_at_mut(1);
    let output_first = first[0].as_mut_slice();
    let output_second = second[0].as_mut_slice();
    let mut left_row = 0;
    let mut right_row = 0;
    let mut written = 0;
    while left_row < left.len() || right_row < right.len() {
        let tuple = if right_row == right.len()
            || (left_row < left.len()
                && compare_pair(
                    left_first[left_row],
                    left_second[left_row],
                    right_first[right_row],
                    right_second[right_row],
                ) != std::cmp::Ordering::Greater)
        {
            let tuple = (left_first[left_row], left_second[left_row]);
            left_row += 1;
            tuple
        } else {
            let tuple = (right_first[right_row], right_second[right_row]);
            right_row += 1;
            tuple
        };
        if written == 0
            || output_first[written - 1] != tuple.0
            || output_second[written - 1] != tuple.1
        {
            output_first[written] = tuple.0;
            output_second[written] = tuple.1;
            written += 1;
        }
    }
    output.set_len(written)?;
    Ok(())
}

fn merge_partition(left: &[u64], right: &[u64], output_position: usize) -> usize {
    let mut low = output_position.saturating_sub(right.len());
    let mut high = output_position.min(left.len());
    while low < high {
        let left_position = low + (high - low) / 2;
        let right_position = output_position - left_position;
        if left_position < left.len()
            && right_position > 0
            && right[right_position - 1] > left[left_position]
        {
            low = left_position + 1;
        } else {
            high = left_position;
        }
    }
    low
}

fn merge_packed_range(left: &[u64], right: &[u64], output_start: usize, output: &mut [u64]) {
    let mut left_row = merge_partition(left, right, output_start);
    let mut right_row = output_start - left_row;
    for value in output {
        if right_row == right.len() || (left_row < left.len() && left[left_row] <= right[right_row])
        {
            *value = left[left_row];
            left_row += 1;
        } else {
            *value = right[right_row];
            right_row += 1;
        }
    }
}

/// Merge and deduplicate two sorted binary relations using parallel merge-path
/// partitions and stable canonical output.
pub fn union_cpu_parallel(
    left: RelationView<'_>,
    right: RelationView<'_>,
    plan: SortedBinaryUnion,
    workspace: &mut UnionWorkspace,
) -> Result<()> {
    let ((left_first, left_second), (right_first, right_second)) = union_inputs(left, right, plan)?;
    let total = left
        .len()
        .checked_add(right.len())
        .ok_or(Error::OutputTooLarge(u64::MAX))?;
    if total == 0 || rayon::current_num_threads() == 1 {
        return union_cpu_serial(left, right, plan, workspace);
    }
    workspace.reserve_rows(left.len(), right.len())?;
    workspace.output_mut().clear();
    let workers = rayon::current_num_threads().min(total);
    let chunk_size = total.div_ceil(workers);
    let (output, packed_left, packed_right, merged) = workspace.cpu_parts();
    let packed_left = &mut packed_left.as_mut_slice()[..left.len()];
    let packed_right = &mut packed_right.as_mut_slice()[..right.len()];
    rayon::join(
        || {
            packed_left
                .par_iter_mut()
                .enumerate()
                .for_each(|(row, output)| {
                    *output = pack_tuple(left_first[row], left_second[row]);
                });
        },
        || {
            packed_right
                .par_iter_mut()
                .enumerate()
                .for_each(|(row, output)| {
                    *output = pack_tuple(right_first[row], right_second[row]);
                });
        },
    );
    let packed_left = &*packed_left;
    let packed_right = &*packed_right;
    merged.as_mut_slice()[..total]
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk, output)| {
            merge_packed_range(packed_left, packed_right, chunk * chunk_size, output);
        });
    let merged = &mut merged.as_mut_slice()[..total];
    let unique = deduplicate_sorted(merged);
    unpack_packed_output(output, merged, unique, true)
}

#[must_use = "dropping a pending CUDA union waits for its stream"]
pub struct PendingUnion<'a> {
    _left: RelationView<'a>,
    _right: RelationView<'a>,
    workspace: &'a mut UnionWorkspace,
    stream: &'a CudaStream,
    completed: bool,
}

impl PendingUnion<'_> {
    pub fn wait(mut self) -> Result<()> {
        self.stream.synchronize()?;
        let unique_u64 = self.workspace.count().as_slice()[0];
        let unique = usize::try_from(unique_u64).map_err(|_| Error::OutputTooLarge(unique_u64))?;
        self.workspace.output_mut().set_len(unique)?;
        self.completed = true;
        Ok(())
    }
}

impl Drop for PendingUnion<'_> {
    fn drop(&mut self) {
        if !self.completed {
            let _ = self.stream.synchronize();
        }
    }
}

/// Merge and deduplicate sorted binary relations with CUDA device-wide merge
/// and unique compaction.
pub fn union_cuda<'a>(
    left: RelationView<'a>,
    right: RelationView<'a>,
    plan: SortedBinaryUnion,
    workspace: &'a mut UnionWorkspace,
    stream: &'a CudaStream,
) -> Result<PendingUnion<'a>> {
    let ((left_first, left_second), (right_first, right_second)) = union_inputs(left, right, plan)?;
    left.len()
        .checked_add(right.len())
        .ok_or(Error::OutputTooLarge(u64::MAX))?;
    workspace.reserve_rows(left.len(), right.len())?;
    workspace.output_mut().clear();
    workspace.count_mut().as_mut_slice()[0] = 0;

    let mut temporary_bytes = 0;
    {
        let parts = workspace.cuda_parts();
        // SAFETY: all pointers refer to live managed buffers. A null temporary
        // pointer asks CUB for the maximum merge/unique workspace requirement.
        unsafe {
            cuda_result(
                "sparkalog_union_u32_temporary_bytes",
                sparkalog_union_u32_temporary_bytes(
                    parts.left.as_ptr(),
                    left.len(),
                    parts.right.as_ptr(),
                    right.len(),
                    parts.merged.as_mut_ptr(),
                    parts.unique.as_mut_ptr(),
                    parts.count.as_mut_ptr(),
                    &mut temporary_bytes,
                    stream.raw.as_ptr(),
                ),
            )?;
        }
    }
    workspace.reserve_temporary_bytes(temporary_bytes)?;

    let launch_result = {
        let parts = workspace.cuda_parts();
        let (output0, output1) = parts.output.columns_mut().split_at_mut(1);
        // SAFETY: the pending result retains both inputs and exclusive output
        // workspace access until the stream completes.
        unsafe {
            cuda_result(
                "sparkalog_union_u32",
                sparkalog_union_u32(
                    left_first.as_ptr(),
                    left_second.as_ptr(),
                    left.len(),
                    right_first.as_ptr(),
                    right_second.as_ptr(),
                    right.len(),
                    parts.left.as_mut_ptr(),
                    parts.right.as_mut_ptr(),
                    parts.merged.as_mut_ptr(),
                    parts.unique.as_mut_ptr(),
                    parts.count.as_mut_ptr(),
                    output0[0].as_mut_ptr(),
                    output1[0].as_mut_ptr(),
                    parts.temporary.as_mut_ptr().cast(),
                    temporary_bytes,
                    stream.raw.as_ptr(),
                ),
            )
        }
    };
    if let Err(error) = launch_result {
        let _ = stream.synchronize();
        return Err(error);
    }
    Ok(PendingUnion {
        _left: left,
        _right: right,
        workspace,
        stream,
        completed: false,
    })
}

/// Execute sorted binary union using a measured placement policy.
pub fn union_auto(
    left: RelationView<'_>,
    right: RelationView<'_>,
    plan: SortedBinaryUnion,
    input_provenance: InputProvenance,
    workspace: &mut UnionWorkspace,
    stream: Option<&CudaStream>,
    policy: UnionPlacementPolicy,
) -> Result<Placement> {
    let placement = policy.place(UnionPlacementContext {
        left_rows: left.len(),
        right_rows: right.len(),
        input_provenance,
        gpu_available: stream.is_some(),
    });
    match placement {
        Placement::CpuSerial => union_cpu_serial(left, right, plan, workspace)?,
        Placement::CpuParallel => union_cpu_parallel(left, right, plan, workspace)?,
        Placement::Gpu => {
            let stream = stream.expect("GPU placement requires an available CUDA stream");
            union_cuda(left, right, plan, workspace, stream)?.wait()?;
        }
    }
    Ok(placement)
}

#[derive(Clone, Copy)]
struct IndexView<'a> {
    keys: &'a [u32],
    starts: &'a [u32],
    rows: &'a [u32],
}

impl<'a> IndexView<'a> {
    fn new(index: &'a U32RangeIndex) -> Self {
        Self {
            keys: index.keys().as_slice(),
            starts: index.starts().as_slice(),
            rows: index.rows().as_slice(),
        }
    }

    fn lookup(self, key: u32) -> &'a [u32] {
        let Ok(index) = self.keys.binary_search(&key) else {
            return &[];
        };
        &self.rows[self.starts[index] as usize..self.starts[index + 1] as usize]
    }
}

#[derive(Clone, Copy)]
enum ProjectionColumn<'a> {
    Left(&'a [u32]),
    Right(&'a [u32]),
}

impl ProjectionColumn<'_> {
    fn value(self, left_row: usize, right_row: usize) -> u32 {
        match self {
            Self::Left(column) => column[left_row],
            Self::Right(column) => column[right_row],
        }
    }

    fn as_ptr(self) -> *const u32 {
        match self {
            Self::Left(column) | Self::Right(column) => column.as_ptr(),
        }
    }

    fn side(self) -> u32 {
        match self {
            Self::Left(_) => 0,
            Self::Right(_) => 1,
        }
    }
}

fn relation_column<'a>(
    relation: RelationView<'a>,
    input: &'static str,
    column: u32,
) -> Result<&'a [u32]> {
    relation
        .column_slice(column as usize)
        .ok_or(Error::MissingColumn { input, column })
}

fn projection_column<'a>(
    left: RelationView<'a>,
    right: RelationView<'a>,
    projection: JoinProjection,
) -> Result<ProjectionColumn<'a>> {
    match projection.input {
        JoinInput::Left => {
            relation_column(left, "left", projection.column).map(ProjectionColumn::Left)
        }
        JoinInput::Right => {
            relation_column(right, "right", projection.column).map(ProjectionColumn::Right)
        }
    }
}

fn join_inputs<'a>(
    left: RelationView<'a>,
    right: RelationView<'a>,
    right_index: &'a U32RangeIndex,
    plan: BinaryEqualityJoin,
) -> Result<(&'a [u32], IndexView<'a>, [ProjectionColumn<'a>; 2])> {
    check_row_count(left.len())?;
    check_row_count(right.len())?;
    let left_key = relation_column(left, "left", plan.left_key)?;
    relation_column(right, "right", plan.right_key)?;
    if right.len() != right_index.source_rows() {
        return Err(Error::IndexSourceMismatch {
            relation_rows: right.len(),
            index_rows: right_index.source_rows(),
        });
    }
    Ok((
        left_key,
        IndexView::new(right_index),
        [
            projection_column(left, right, plan.output[0])?,
            projection_column(left, right, plan.output[1])?,
        ],
    ))
}

fn count_join_rows(
    left_key: &[u32],
    workspace: &mut JoinWorkspace,
    parallel: bool,
    match_count: impl Fn(u32) -> usize + Sync,
) -> Result<usize> {
    workspace.reserve_outer_rows(left_key.len())?;
    let (counts, offsets) = workspace.cpu_count_parts();
    let counts = &mut counts.as_mut_slice()[..left_key.len()];
    if parallel {
        left_key
            .par_iter()
            .zip(counts.par_iter_mut())
            .for_each(|(&key, count)| *count = match_count(key) as u64);
    } else {
        for (&key, count) in left_key.iter().zip(counts.iter_mut()) {
            *count = match_count(key) as u64;
        }
    }
    let offsets = &mut offsets.as_mut_slice()[..left_key.len()];
    let mut total = 0_u64;
    for (&count, offset) in counts.iter().zip(offsets) {
        *offset = total;
        total = total
            .checked_add(count)
            .ok_or(Error::OutputTooLarge(u64::MAX))?;
    }
    usize::try_from(total).map_err(|_| Error::OutputTooLarge(total))
}

/// Execute an indexed binary equality join on the calling thread.
pub fn join_cpu_serial(
    left: RelationView<'_>,
    right: RelationView<'_>,
    right_index: &U32RangeIndex,
    plan: BinaryEqualityJoin,
    workspace: &mut JoinWorkspace,
) -> Result<()> {
    let (left_key, index, projection) = join_inputs(left, right, right_index, plan)?;
    let total = count_join_rows(left_key, workspace, false, |key| index.lookup(key).len())?;
    workspace.reserve_output_rows(total)?;
    let (output, offsets) = workspace.emit_parts();
    let offsets = &offsets.as_slice()[..left.len()];
    let (first, second) = output.columns_mut().split_at_mut(1);
    let output0 = &mut first[0].as_mut_slice()[..total];
    let output1 = &mut second[0].as_mut_slice()[..total];

    for (left_row, &key) in left_key.iter().enumerate() {
        let output_start = offsets[left_row] as usize;
        for (match_offset, &right_row) in index.lookup(key).iter().enumerate() {
            let output_row = output_start + match_offset;
            let right_row = right_row as usize;
            output0[output_row] = projection[0].value(left_row, right_row);
            output1[output_row] = projection[1].value(left_row, right_row);
        }
    }
    output.set_len(total)?;
    Ok(())
}

/// Execute an indexed binary equality join using the persistent Rayon pool.
pub fn join_cpu_parallel(
    left: RelationView<'_>,
    right: RelationView<'_>,
    right_index: &U32RangeIndex,
    plan: BinaryEqualityJoin,
    workspace: &mut JoinWorkspace,
) -> Result<()> {
    let (left_key, index, projection) = join_inputs(left, right, right_index, plan)?;
    if left.is_empty() || rayon::current_num_threads() == 1 {
        return join_cpu_serial(left, right, right_index, plan, workspace);
    }
    let total = count_join_rows(left_key, workspace, true, |key| index.lookup(key).len())?;
    workspace.reserve_output_rows(total)?;
    let left_len = left.len();
    let workers = rayon::current_num_threads().min(left_len);
    let chunk_size = left_len.div_ceil(workers);
    let (output, offsets) = workspace.emit_parts();
    let offsets = &offsets.as_slice()[..left_len];
    let (first, second) = output.columns_mut().split_at_mut(1);
    let mut remaining0 = &mut first[0].as_mut_slice()[..total];
    let mut remaining1 = &mut second[0].as_mut_slice()[..total];

    rayon::scope(|scope| {
        for row_start in (0..left_len).step_by(chunk_size) {
            let row_end = (row_start + chunk_size).min(left_len);
            let output_start = offsets[row_start] as usize;
            let output_end = if row_end == left_len {
                total
            } else {
                offsets[row_end] as usize
            };
            let output_len = output_end - output_start;
            let (output0, next0) = remaining0.split_at_mut(output_len);
            let (output1, next1) = remaining1.split_at_mut(output_len);
            remaining0 = next0;
            remaining1 = next1;
            scope.spawn(move |_| {
                for left_row in row_start..row_end {
                    let row_output_start = offsets[left_row] as usize - output_start;
                    for (match_offset, &right_row) in
                        index.lookup(left_key[left_row]).iter().enumerate()
                    {
                        let output_row = row_output_start + match_offset;
                        let right_row = right_row as usize;
                        output0[output_row] = projection[0].value(left_row, right_row);
                        output1[output_row] = projection[1].value(left_row, right_row);
                    }
                }
            });
        }
    });
    output.set_len(total)?;
    Ok(())
}

fn bitmap_join_inputs<'a>(
    left: RelationView<'a>,
    right: RelationView<'a>,
    right_index: &'a U32BitmapIndex,
    plan: BinaryEqualityJoin,
) -> Result<(&'a [u32], [ProjectionColumn<'a>; 2])> {
    check_row_count(left.len())?;
    check_row_count(right.len())?;
    let left_key = relation_column(left, "left", plan.left_key)?;
    relation_column(right, "right", plan.right_key)?;
    if right.len() != right_index.source_rows() {
        return Err(Error::IndexSourceMismatch {
            relation_rows: right.len(),
            index_rows: right_index.source_rows(),
        });
    }
    Ok((
        left_key,
        [
            projection_column(left, right, plan.output[0])?,
            projection_column(left, right, plan.output[1])?,
        ],
    ))
}

/// Execute a binary equality join using sparse bitmap posting lists on the
/// calling thread.
pub fn join_cpu_bitmap_serial(
    left: RelationView<'_>,
    right: RelationView<'_>,
    right_index: &U32BitmapIndex,
    plan: BinaryEqualityJoin,
    workspace: &mut JoinWorkspace,
) -> Result<()> {
    let (left_key, projection) = bitmap_join_inputs(left, right, right_index, plan)?;
    let total = count_join_rows(left_key, workspace, false, |key| {
        right_index.lookup(key).map_or(0, |posting| posting.len())
    })?;
    workspace.reserve_output_rows(total)?;
    let (output, offsets) = workspace.emit_parts();
    let offsets = &offsets.as_slice()[..left.len()];
    let (first, second) = output.columns_mut().split_at_mut(1);
    let output0 = &mut first[0].as_mut_slice()[..total];
    let output1 = &mut second[0].as_mut_slice()[..total];

    for (left_row, &key) in left_key.iter().enumerate() {
        let Some(posting) = right_index.lookup(key) else {
            continue;
        };
        let output_start = offsets[left_row] as usize;
        for (match_offset, right_row) in posting.rows().enumerate() {
            let output_row = output_start + match_offset;
            let right_row = right_row as usize;
            output0[output_row] = projection[0].value(left_row, right_row);
            output1[output_row] = projection[1].value(left_row, right_row);
        }
    }
    output.set_len(total)?;
    Ok(())
}

/// Execute a binary equality join using sparse bitmap posting lists and the
/// persistent Rayon pool.
pub fn join_cpu_bitmap_parallel(
    left: RelationView<'_>,
    right: RelationView<'_>,
    right_index: &U32BitmapIndex,
    plan: BinaryEqualityJoin,
    workspace: &mut JoinWorkspace,
) -> Result<()> {
    let (left_key, projection) = bitmap_join_inputs(left, right, right_index, plan)?;
    if left.is_empty() || rayon::current_num_threads() == 1 {
        return join_cpu_bitmap_serial(left, right, right_index, plan, workspace);
    }
    let total = count_join_rows(left_key, workspace, true, |key| {
        right_index.lookup(key).map_or(0, |posting| posting.len())
    })?;
    workspace.reserve_output_rows(total)?;
    let left_len = left.len();
    let workers = rayon::current_num_threads().min(left_len);
    let chunk_size = left_len.div_ceil(workers);
    let (output, offsets) = workspace.emit_parts();
    let offsets = &offsets.as_slice()[..left_len];
    let (first, second) = output.columns_mut().split_at_mut(1);
    let mut remaining0 = &mut first[0].as_mut_slice()[..total];
    let mut remaining1 = &mut second[0].as_mut_slice()[..total];

    rayon::scope(|scope| {
        for row_start in (0..left_len).step_by(chunk_size) {
            let row_end = (row_start + chunk_size).min(left_len);
            let output_start = offsets[row_start] as usize;
            let output_end = if row_end == left_len {
                total
            } else {
                offsets[row_end] as usize
            };
            let output_len = output_end - output_start;
            let (output0, next0) = remaining0.split_at_mut(output_len);
            let (output1, next1) = remaining1.split_at_mut(output_len);
            remaining0 = next0;
            remaining1 = next1;
            scope.spawn(move |_| {
                for left_row in row_start..row_end {
                    let Some(posting) = right_index.lookup(left_key[left_row]) else {
                        continue;
                    };
                    let row_output_start = offsets[left_row] as usize - output_start;
                    for (match_offset, right_row) in posting.rows().enumerate() {
                        let output_row = row_output_start + match_offset;
                        let right_row = right_row as usize;
                        output0[output_row] = projection[0].value(left_row, right_row);
                        output1[output_row] = projection[1].value(left_row, right_row);
                    }
                }
            });
        }
    });
    output.set_len(total)?;
    Ok(())
}

/// Execute an indexed binary equality join with CUDA count, scan, and emit
/// stages. The exact output allocation requires one synchronization between
/// scan and emit.
pub fn join_cuda(
    left: RelationView<'_>,
    right: RelationView<'_>,
    right_index: &U32RangeIndex,
    plan: BinaryEqualityJoin,
    workspace: &mut JoinWorkspace,
    stream: &CudaStream,
) -> Result<()> {
    let (left_key, index, projection) = join_inputs(left, right, right_index, plan)?;
    workspace.reserve_outer_rows(left.len())?;
    workspace.output_mut().clear();

    let mut temporary_bytes = 0;
    {
        let (counts, offsets, _, _) = workspace.cuda_count_parts();
        // SAFETY: the pointers refer to live managed buffers; a null temporary
        // pointer asks CUB only for the required scan workspace size.
        unsafe {
            cuda_result(
                "cub::DeviceScan::ExclusiveSum(size)",
                sparkalog_join_u32_temporary_bytes(
                    counts.as_ptr(),
                    offsets.as_mut_ptr(),
                    left.len(),
                    &mut temporary_bytes,
                    stream.raw.as_ptr(),
                ),
            )?;
        }
    }
    workspace.reserve_temporary_bytes(temporary_bytes)?;

    let count_result = {
        let (counts, offsets, total, temporary) = workspace.cuda_count_parts();
        // SAFETY: all inputs and scratch buffers are CUDA-managed and remain
        // live through the synchronization immediately below.
        unsafe {
            cuda_result(
                "sparkalog_join_u32_count",
                sparkalog_join_u32_count(
                    left_key.as_ptr(),
                    left.len(),
                    index.keys.as_ptr(),
                    index.starts.as_ptr(),
                    index.keys.len(),
                    counts.as_mut_ptr(),
                    offsets.as_mut_ptr(),
                    total.as_mut_ptr(),
                    temporary.as_mut_ptr().cast(),
                    temporary_bytes,
                    stream.raw.as_ptr(),
                ),
            )
        }
    };
    if let Err(error) = count_result {
        let _ = stream.synchronize();
        return Err(error);
    }
    stream.synchronize()?;
    let total_u64 = workspace.total().as_slice()[0];
    let total = usize::try_from(total_u64).map_err(|_| Error::OutputTooLarge(total_u64))?;
    workspace.reserve_output_rows(total)?;

    let emit_result = {
        let (output, offsets) = workspace.emit_parts();
        let (first, second) = output.columns_mut().split_at_mut(1);
        // SAFETY: output columns have `total` initialized slots, offsets were
        // completed by the synchronized count stage, and every emitted range
        // is disjoint by construction.
        unsafe {
            cuda_result(
                "sparkalog_join_u32_emit",
                sparkalog_join_u32_emit(
                    left_key.as_ptr(),
                    left.len(),
                    index.keys.as_ptr(),
                    index.starts.as_ptr(),
                    index.rows.as_ptr(),
                    index.keys.len(),
                    projection[0].as_ptr(),
                    projection[0].side(),
                    projection[1].as_ptr(),
                    projection[1].side(),
                    offsets.as_ptr(),
                    first[0].as_mut_ptr(),
                    second[0].as_mut_ptr(),
                    stream.raw.as_ptr(),
                ),
            )
        }
    };
    if let Err(error) = emit_result {
        let _ = stream.synchronize();
        return Err(error);
    }
    stream.synchronize()?;
    workspace.output_mut().set_len(total)?;
    Ok(())
}

/// Place and execute an indexed binary equality join using a measured policy.
/// A CUDA stream being present is the GPU-availability signal.
pub fn join_auto(
    left: RelationView<'_>,
    right: RelationView<'_>,
    right_index: &U32RangeIndex,
    plan: BinaryEqualityJoin,
    workspace: &mut JoinWorkspace,
    stream: Option<&CudaStream>,
    policy: JoinPlacementPolicy,
) -> Result<Placement> {
    let placement = policy.place(JoinPlacementContext {
        delta_rows: left.len(),
        gpu_available: stream.is_some(),
    });
    match placement {
        Placement::CpuSerial => join_cpu_serial(left, right, right_index, plan, workspace)?,
        Placement::CpuParallel => join_cpu_parallel(left, right, right_index, plan, workspace)?,
        Placement::Gpu => join_cuda(
            left,
            right,
            right_index,
            plan,
            workspace,
            stream.expect("GPU placement requires an available CUDA stream"),
        )?,
    }
    Ok(placement)
}

/// Filter a column on the calling thread and produce ordered row IDs.
pub fn filter_cpu_serial(
    column: &Column,
    predicate: U32Predicate,
    workspace: &mut OperatorWorkspace,
) -> Result<()> {
    let rows = column.len();
    check_row_count(rows)?;
    workspace.reserve_rows(rows)?;

    let selection = workspace.selection_mut();
    selection.clear();
    let output = selection.storage_mut().as_mut_slice();
    let mut selected = 0;
    for (row, &value) in column.as_slice().iter().enumerate() {
        if predicate.matches(value) {
            output[selected] = row as u32;
            selected += 1;
        }
    }
    selection.set_len(selected)?;
    Ok(())
}

/// Filter a column using scoped native Rust workers and produce ordered row IDs.
pub fn filter_cpu_parallel(
    column: &Column,
    predicate: U32Predicate,
    workspace: &mut OperatorWorkspace,
) -> Result<()> {
    let rows = column.len();
    check_row_count(rows)?;
    if rows == 0 {
        workspace.selection_mut().clear();
        return Ok(());
    }

    let workers = rayon::current_num_threads().min(rows);
    if workers == 1 {
        return filter_cpu_serial(column, predicate, workspace);
    }

    workspace.reserve_rows(rows)?;
    let chunk_size = rows.div_ceil(workers);
    let chunk_count = rows.div_ceil(chunk_size);
    let (selection, flags, offsets) = workspace.cpu_compaction_parts();
    selection.clear();
    let flags = &mut flags.as_mut_slice()[..rows];

    column
        .as_slice()
        .par_chunks(chunk_size)
        .zip(flags.par_chunks_mut(chunk_size))
        .for_each(|(values, output_flags)| {
            for (&value, flag) in values.iter().zip(output_flags) {
                *flag = u32::from(predicate.matches(value));
            }
        });

    let counts = &mut offsets.as_mut_slice()[..chunk_count];
    let mut selected = 0_usize;
    for (chunk_index, flag_chunk) in flags.chunks(chunk_size).enumerate() {
        let count = flag_chunk.iter().map(|&flag| flag as usize).sum::<usize>();
        counts[chunk_index] = count as u32;
        selected += count;
    }

    let mut remaining_output = &mut selection.storage_mut().as_mut_slice()[..selected];
    rayon::scope(|scope| {
        for (chunk_index, flag_chunk) in flags.chunks(chunk_size).enumerate() {
            let output_len = counts[chunk_index] as usize;
            let (output, remaining) = remaining_output.split_at_mut(output_len);
            remaining_output = remaining;
            let row_start = chunk_index * chunk_size;
            scope.spawn(move |_| {
                let mut output_index = 0;
                for (offset, &flag) in flag_chunk.iter().enumerate() {
                    if flag != 0 {
                        output[output_index] = (row_start + offset) as u32;
                        output_index += 1;
                    }
                }
            });
        }
    });
    selection.set_len(selected)?;
    Ok(())
}

fn encode_predicate(predicate: U32Predicate) -> (u32, u32) {
    match predicate {
        U32Predicate::Eq(value) => (0, value),
        U32Predicate::Ne(value) => (1, value),
        U32Predicate::Lt(value) => (2, value),
        U32Predicate::Le(value) => (3, value),
        U32Predicate::Gt(value) => (4, value),
        U32Predicate::Ge(value) => (5, value),
    }
}

/// An asynchronous CUDA filter which retains the mutable workspace borrow
/// until its ordered row-ID output is complete.
#[must_use = "dropping a pending CUDA filter waits for its stream"]
pub struct PendingFilter<'a> {
    workspace: &'a mut OperatorWorkspace,
    stream: &'a CudaStream,
    completed: bool,
}

impl PendingFilter<'_> {
    pub fn wait(mut self) -> Result<()> {
        self.stream.synchronize()?;
        let selected = self.workspace.count().as_slice()[0] as usize;
        self.workspace.selection_mut().set_len(selected)?;
        self.completed = true;
        Ok(())
    }
}

impl Drop for PendingFilter<'_> {
    fn drop(&mut self) {
        if !self.completed {
            let _ = self.stream.synchronize();
        }
    }
}

/// Filter a column with CUDA and produce the same ordered row IDs as the Rust
/// implementations.
pub fn filter_cuda<'a>(
    column: &'a Column,
    predicate: U32Predicate,
    workspace: &'a mut OperatorWorkspace,
    stream: &'a CudaStream,
) -> Result<PendingFilter<'a>> {
    let rows = column.len();
    check_row_count(rows)?;
    workspace.reserve_rows(rows)?;
    workspace.selection_mut().clear();
    workspace.count_mut().as_mut_slice()[0] = 0;

    let mut temporary_bytes = 0;
    {
        let (selection, flags, count, _) = workspace.cuda_compaction_parts();
        // SAFETY: all pointers refer to live CUDA-managed allocations. A null
        // temporary pointer requests CUB's required workspace size only.
        unsafe {
            cuda_result(
                "cub::DeviceSelect::Flagged(size)",
                sparkalog_filter_u32_temporary_bytes(
                    flags.as_ptr(),
                    selection.storage_mut().as_mut_ptr(),
                    count.as_mut_ptr(),
                    rows,
                    &mut temporary_bytes,
                    stream.raw.as_ptr(),
                ),
            )?;
        }
    }
    workspace.reserve_temporary_bytes(temporary_bytes)?;

    let (operation, operand) = encode_predicate(predicate);
    let launch_result = {
        let (selection, flags, count, temporary) = workspace.cuda_compaction_parts();
        // SAFETY: the pending result retains exclusive access to every output
        // allocation until `stream` completes. CUB preserves input order.
        unsafe {
            cuda_result(
                "sparkalog_filter_u32",
                sparkalog_filter_u32(
                    column.as_ptr(),
                    rows,
                    operation,
                    operand,
                    flags.as_mut_ptr(),
                    selection.storage_mut().as_mut_ptr(),
                    count.as_mut_ptr(),
                    temporary.as_mut_ptr().cast(),
                    temporary_bytes,
                    stream.raw.as_ptr(),
                ),
            )
        }
    };
    if let Err(error) = launch_result {
        let _ = stream.synchronize();
        return Err(error);
    }

    Ok(PendingFilter {
        workspace,
        stream,
        completed: false,
    })
}

/// Execute a filter with the backend selected by a measured policy.
pub fn filter_auto(
    column: &Column,
    predicate: U32Predicate,
    input_provenance: InputProvenance,
    workspace: &mut OperatorWorkspace,
    stream: Option<&CudaStream>,
    policy: FilterPlacementPolicy,
) -> Result<Placement> {
    let placement = policy.place(FilterPlacementContext {
        rows: column.len(),
        input_provenance,
        gpu_available: stream.is_some(),
    });
    match placement {
        Placement::CpuSerial => filter_cpu_serial(column, predicate, workspace)?,
        Placement::CpuParallel => filter_cpu_parallel(column, predicate, workspace)?,
        Placement::Gpu => {
            let stream = stream.expect("GPU placement requires an available CUDA stream");
            filter_cuda(column, predicate, workspace, stream)?.wait()?;
        }
    }
    Ok(placement)
}

/// A GPU producer used to establish input provenance in crossover benchmarks.
#[must_use = "dropping a pending CUDA fill waits for its stream"]
pub struct PendingFill<'a> {
    _column: &'a mut Column,
    stream: &'a CudaStream,
    completed: bool,
}

impl PendingFill<'_> {
    pub fn wait(mut self) -> Result<()> {
        self.stream.synchronize()?;
        self.completed = true;
        Ok(())
    }
}

impl Drop for PendingFill<'_> {
    fn drop(&mut self) {
        if !self.completed {
            let _ = self.stream.synchronize();
        }
    }
}

pub fn fill_mod_u32<'a>(
    column: &'a mut Column,
    modulus: u32,
    stream: &'a CudaStream,
) -> Result<PendingFill<'a>> {
    // SAFETY: the pending result retains exclusive access to `column` until
    // the stream completes the fill kernel.
    unsafe {
        cuda_result(
            "sparkalog_fill_mod_u32",
            sparkalog_fill_mod_u32(
                column.as_mut_ptr(),
                column.len(),
                modulus,
                stream.raw.as_ptr(),
            ),
        )?;
    }
    Ok(PendingFill {
        _column: column,
        stream,
        completed: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparkalog_storage::Relation;

    fn input(values: &[u32]) -> Column {
        let mut column = Column::new_filled(values.len(), 0).unwrap();
        column.as_mut_slice().copy_from_slice(values);
        column
    }

    fn expected(values: &[u32], predicate: U32Predicate) -> Vec<u32> {
        values
            .iter()
            .enumerate()
            .filter_map(|(row, &value)| predicate.matches(value).then_some(row as u32))
            .collect()
    }

    fn relation2(rows: &[(u32, u32)]) -> Relation {
        let mut relation = Relation::new(2, rows.len()).unwrap();
        for (row, &(left, right)) in rows.iter().enumerate() {
            relation.column_mut(0).unwrap().as_mut_slice()[row] = left;
            relation.column_mut(1).unwrap().as_mut_slice()[row] = right;
        }
        relation
    }

    fn path_join_plan() -> BinaryEqualityJoin {
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

    fn binary_distinct_plan() -> BinaryDistinct {
        BinaryDistinct { columns: [0, 1] }
    }

    fn sorted_binary_anti_join_plan() -> SortedBinaryAntiJoin {
        SortedBinaryAntiJoin {
            left: [0, 1],
            right: [0, 1],
        }
    }

    fn sorted_binary_union_plan() -> SortedBinaryUnion {
        SortedBinaryUnion {
            left: [0, 1],
            right: [0, 1],
        }
    }

    #[test]
    fn serial_and_parallel_filters_match() {
        let values = (0..10_000).map(|value| value % 97).collect::<Vec<_>>();
        let column = input(&values);
        let predicate = U32Predicate::Ge(51);
        let expected = expected(&values, predicate);
        let mut serial = OperatorWorkspace::new().unwrap();
        let mut parallel = OperatorWorkspace::new().unwrap();

        filter_cpu_serial(&column, predicate, &mut serial).unwrap();
        filter_cpu_parallel(&column, predicate, &mut parallel).unwrap();

        assert_eq!(serial.selection().as_slice(), expected);
        assert_eq!(parallel.selection().as_slice(), expected);
    }

    #[test]
    fn serial_and_parallel_distinct_sort_and_deduplicate_pairs() {
        let input = relation2(&[(2, 20), (1, 10), (2, 20), (1, 11), (1, 10)]);
        let mut serial = DistinctWorkspace::new().unwrap();
        let mut parallel = DistinctWorkspace::new().unwrap();
        let mut gpu = DistinctWorkspace::new().unwrap();
        let stream = CudaStream::new().unwrap();

        distinct_cpu_serial(input.view(), binary_distinct_plan(), &mut serial).unwrap();
        distinct_cpu_parallel(input.view(), binary_distinct_plan(), &mut parallel).unwrap();
        distinct_cuda(input.view(), binary_distinct_plan(), &mut gpu, &stream)
            .unwrap()
            .wait()
            .unwrap();

        assert_eq!(serial.output().view().column_slice(0).unwrap(), &[1, 1, 2]);
        assert_eq!(
            serial.output().view().column_slice(1).unwrap(),
            &[10, 11, 20]
        );
        for output in [parallel.output(), gpu.output()] {
            assert_eq!(
                output.view().column_slice(0),
                serial.output().view().column_slice(0)
            );
            assert_eq!(
                output.view().column_slice(1),
                serial.output().view().column_slice(1)
            );
        }
    }

    #[test]
    fn serial_and_parallel_anti_join_preserve_sorted_left_difference() {
        let left = relation2(&[(1, 10), (1, 11), (2, 20), (3, 30), (5, 50)]);
        let right = relation2(&[(1, 11), (3, 30), (4, 40)]);
        let mut serial = AntiJoinWorkspace::new().unwrap();
        let mut parallel = AntiJoinWorkspace::new().unwrap();
        let mut gpu = AntiJoinWorkspace::new().unwrap();
        let stream = CudaStream::new().unwrap();

        anti_join_cpu_serial(
            left.view(),
            right.view(),
            sorted_binary_anti_join_plan(),
            &mut serial,
        )
        .unwrap();
        anti_join_cuda(
            left.view(),
            right.view(),
            sorted_binary_anti_join_plan(),
            &mut gpu,
            &stream,
        )
        .unwrap()
        .wait()
        .unwrap();
        anti_join_cpu_parallel(
            left.view(),
            right.view(),
            sorted_binary_anti_join_plan(),
            &mut parallel,
        )
        .unwrap();

        assert_eq!(serial.output().view().column_slice(0).unwrap(), &[1, 2, 5]);
        assert_eq!(
            serial.output().view().column_slice(1).unwrap(),
            &[10, 20, 50]
        );
        for output in [parallel.output(), gpu.output()] {
            assert_eq!(
                output.view().column_slice(0),
                serial.output().view().column_slice(0)
            );
            assert_eq!(
                output.view().column_slice(1),
                serial.output().view().column_slice(1)
            );
        }
    }

    #[test]
    fn all_anti_join_backends_handle_empty_relations() {
        let empty = relation2(&[]);
        let left = relation2(&[(1, 10), (2, 20)]);
        let stream = CudaStream::new().unwrap();
        let mut workspace = AntiJoinWorkspace::new().unwrap();

        anti_join_cpu_serial(
            left.view(),
            empty.view(),
            sorted_binary_anti_join_plan(),
            &mut workspace,
        )
        .unwrap();
        assert_eq!(workspace.output().len(), 2);
        anti_join_cpu_parallel(
            empty.view(),
            left.view(),
            sorted_binary_anti_join_plan(),
            &mut workspace,
        )
        .unwrap();
        assert!(workspace.output().is_empty());
        anti_join_cuda(
            left.view(),
            empty.view(),
            sorted_binary_anti_join_plan(),
            &mut workspace,
            &stream,
        )
        .unwrap()
        .wait()
        .unwrap();
        assert_eq!(workspace.output().len(), 2);
        anti_join_cuda(
            empty.view(),
            left.view(),
            sorted_binary_anti_join_plan(),
            &mut workspace,
            &stream,
        )
        .unwrap()
        .wait()
        .unwrap();
        assert!(workspace.output().is_empty());
    }

    #[test]
    fn automatic_anti_join_executes_the_selected_backend() {
        let left = relation2(&[(1, 10), (1, 11), (2, 20), (3, 30), (5, 50)]);
        let right = relation2(&[(1, 11), (3, 30), (4, 40)]);
        let policy = AntiJoinPlacementPolicy {
            cpu_produced_gpu_min_rows: 1,
            gpu_produced_gpu_min_rows: 1,
            cpu_produced_parallel_min_rows: 1,
            gpu_produced_parallel_min_rows: 1,
        };
        let mut workspace = AntiJoinWorkspace::new().unwrap();

        let placement = anti_join_auto(
            left.view(),
            right.view(),
            sorted_binary_anti_join_plan(),
            InputProvenance::Cpu,
            &mut workspace,
            None,
            policy,
        )
        .unwrap();
        assert_eq!(placement, Placement::CpuParallel);
        assert_eq!(
            workspace.output().view().column_slice(0).unwrap(),
            &[1, 2, 5]
        );

        let stream = CudaStream::new().unwrap();
        let placement = anti_join_auto(
            left.view(),
            right.view(),
            sorted_binary_anti_join_plan(),
            InputProvenance::Gpu,
            &mut workspace,
            Some(&stream),
            policy,
        )
        .unwrap();
        assert_eq!(placement, Placement::Gpu);
        assert_eq!(
            workspace.output().view().column_slice(1).unwrap(),
            &[10, 20, 50]
        );
    }

    #[test]
    fn all_union_backends_merge_and_deduplicate_sorted_pairs() {
        let left = relation2(&[(1, 10), (2, 20), (2, 20), (4, 40)]);
        let right = relation2(&[(1, 11), (2, 20), (3, 30), (5, 50)]);
        let mut serial = UnionWorkspace::new().unwrap();
        let mut parallel = UnionWorkspace::new().unwrap();
        let mut gpu = UnionWorkspace::new().unwrap();
        let stream = CudaStream::new().unwrap();

        union_cpu_serial(
            left.view(),
            right.view(),
            sorted_binary_union_plan(),
            &mut serial,
        )
        .unwrap();
        union_cpu_parallel(
            left.view(),
            right.view(),
            sorted_binary_union_plan(),
            &mut parallel,
        )
        .unwrap();
        union_cuda(
            left.view(),
            right.view(),
            sorted_binary_union_plan(),
            &mut gpu,
            &stream,
        )
        .unwrap()
        .wait()
        .unwrap();

        assert_eq!(
            serial.output().view().column_slice(0).unwrap(),
            &[1, 1, 2, 3, 4, 5]
        );
        assert_eq!(
            serial.output().view().column_slice(1).unwrap(),
            &[10, 11, 20, 30, 40, 50]
        );
        for output in [parallel.output(), gpu.output()] {
            assert_eq!(
                output.view().column_slice(0),
                serial.output().view().column_slice(0)
            );
            assert_eq!(
                output.view().column_slice(1),
                serial.output().view().column_slice(1)
            );
        }
    }

    #[test]
    fn all_union_backends_handle_empty_relations() {
        let empty = relation2(&[]);
        let right = relation2(&[(1, 10), (2, 20)]);
        let stream = CudaStream::new().unwrap();
        let mut workspace = UnionWorkspace::new().unwrap();

        union_cpu_serial(
            empty.view(),
            right.view(),
            sorted_binary_union_plan(),
            &mut workspace,
        )
        .unwrap();
        assert_eq!(workspace.output().len(), 2);
        union_cpu_parallel(
            right.view(),
            empty.view(),
            sorted_binary_union_plan(),
            &mut workspace,
        )
        .unwrap();
        assert_eq!(workspace.output().len(), 2);
        union_cuda(
            empty.view(),
            empty.view(),
            sorted_binary_union_plan(),
            &mut workspace,
            &stream,
        )
        .unwrap()
        .wait()
        .unwrap();
        assert!(workspace.output().is_empty());
    }

    #[test]
    fn automatic_union_executes_the_selected_backend() {
        let left = relation2(&[(1, 10), (2, 20), (4, 40)]);
        let right = relation2(&[(1, 11), (2, 20), (3, 30)]);
        let policy = UnionPlacementPolicy {
            cpu_produced_gpu_min_rows: 1,
            gpu_produced_gpu_min_rows: 1,
            gpu_unavailable_parallel_min_rows: 1,
        };
        let mut workspace = UnionWorkspace::new().unwrap();

        let placement = union_auto(
            left.view(),
            right.view(),
            sorted_binary_union_plan(),
            InputProvenance::Cpu,
            &mut workspace,
            None,
            policy,
        )
        .unwrap();
        assert_eq!(placement, Placement::CpuParallel);
        assert_eq!(workspace.output().len(), 5);

        let stream = CudaStream::new().unwrap();
        let placement = union_auto(
            left.view(),
            right.view(),
            sorted_binary_union_plan(),
            InputProvenance::Gpu,
            &mut workspace,
            Some(&stream),
            policy,
        )
        .unwrap();
        assert_eq!(placement, Placement::Gpu);
        assert_eq!(
            workspace.output().view().column_slice(0).unwrap(),
            &[1, 1, 2, 3, 4]
        );
    }

    #[test]
    fn automatic_distinct_executes_the_selected_backend() {
        let input = relation2(&[(2, 20), (1, 10), (2, 20), (1, 11), (1, 10)]);
        let policy = DistinctPlacementPolicy {
            cpu_produced_gpu_min_rows: 1,
            gpu_produced_gpu_min_rows: 1,
            gpu_unavailable_parallel_min_rows: 1,
        };
        let mut workspace = DistinctWorkspace::new().unwrap();

        let placement = distinct_auto(
            input.view(),
            binary_distinct_plan(),
            InputProvenance::Cpu,
            &mut workspace,
            None,
            policy,
        )
        .unwrap();
        assert_eq!(placement, Placement::CpuParallel);
        assert_eq!(
            workspace.output().view().column_slice(0).unwrap(),
            &[1, 1, 2]
        );

        let stream = CudaStream::new().unwrap();
        let placement = distinct_auto(
            input.view(),
            binary_distinct_plan(),
            InputProvenance::Gpu,
            &mut workspace,
            Some(&stream),
            policy,
        )
        .unwrap();
        assert_eq!(placement, Placement::Gpu);
        assert_eq!(
            workspace.output().view().column_slice(1).unwrap(),
            &[10, 11, 20]
        );
    }

    #[test]
    fn all_distinct_backends_handle_empty_relations() {
        let input = relation2(&[]);
        let stream = CudaStream::new().unwrap();
        let mut workspace = DistinctWorkspace::new().unwrap();

        distinct_cpu_serial(input.view(), binary_distinct_plan(), &mut workspace).unwrap();
        assert!(workspace.output().is_empty());
        distinct_cpu_parallel(input.view(), binary_distinct_plan(), &mut workspace).unwrap();
        assert!(workspace.output().is_empty());
        distinct_cuda(
            input.view(),
            binary_distinct_plan(),
            &mut workspace,
            &stream,
        )
        .unwrap()
        .wait()
        .unwrap();
        assert!(workspace.output().is_empty());
    }

    #[test]
    fn cuda_filter_matches_all_predicates() {
        let values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let column = input(&values);
        let predicates = [
            U32Predicate::Eq(4),
            U32Predicate::Ne(4),
            U32Predicate::Lt(4),
            U32Predicate::Le(4),
            U32Predicate::Gt(4),
            U32Predicate::Ge(4),
        ];
        let stream = CudaStream::new().unwrap();
        let mut workspace = OperatorWorkspace::new().unwrap();

        for predicate in predicates {
            filter_cuda(&column, predicate, &mut workspace, &stream)
                .unwrap()
                .wait()
                .unwrap();
            assert_eq!(
                workspace.selection().as_slice(),
                expected(&values, predicate)
            );
        }
    }

    #[test]
    fn all_filter_backends_handle_empty_columns() {
        let column = input(&[]);
        let stream = CudaStream::new().unwrap();
        let mut workspace = OperatorWorkspace::new().unwrap();

        filter_cpu_serial(&column, U32Predicate::Eq(0), &mut workspace).unwrap();
        assert!(workspace.selection().is_empty());
        filter_cpu_parallel(&column, U32Predicate::Eq(0), &mut workspace).unwrap();
        assert!(workspace.selection().is_empty());
        filter_cuda(&column, U32Predicate::Eq(0), &mut workspace, &stream)
            .unwrap()
            .wait()
            .unwrap();
        assert!(workspace.selection().is_empty());
    }

    #[test]
    fn gpu_fill_produces_the_expected_pattern() {
        let mut column = input(&[0; 257]);
        let stream = CudaStream::new().unwrap();

        fill_mod_u32(&mut column, 17, &stream)
            .unwrap()
            .wait()
            .unwrap();

        let expected = (0..257).map(|index| index % 17).collect::<Vec<_>>();
        assert_eq!(column.as_slice(), expected);
    }

    #[test]
    fn automatic_filter_executes_the_selected_backend() {
        let values = (0..1_024).map(|value| value % 100).collect::<Vec<_>>();
        let column = input(&values);
        let predicate = U32Predicate::Lt(10);
        let expected = expected(&values, predicate);
        let policy = FilterPlacementPolicy {
            cpu_produced_gpu_min_rows: 512,
            gpu_produced_gpu_min_rows: 512,
            gpu_unavailable_parallel_min_rows: 512,
        };
        let stream = CudaStream::new().unwrap();
        let mut workspace = OperatorWorkspace::new().unwrap();

        let placement = filter_auto(
            &column,
            predicate,
            InputProvenance::Cpu,
            &mut workspace,
            None,
            policy,
        )
        .unwrap();
        assert_eq!(placement, Placement::CpuParallel);
        assert_eq!(workspace.selection().as_slice(), expected);

        let placement = filter_auto(
            &column,
            predicate,
            InputProvenance::Cpu,
            &mut workspace,
            Some(&stream),
            policy,
        )
        .unwrap();
        assert_eq!(placement, Placement::Gpu);
        assert_eq!(workspace.selection().as_slice(), expected);
    }

    #[test]
    fn serial_and_parallel_indexed_joins_match() {
        let left = relation2(&[(10, 1), (20, 2), (30, 1), (40, 9)]);
        let right = relation2(&[(1, 100), (1, 101), (2, 200), (3, 300)]);
        let index = U32RangeIndex::build(right.column(0).unwrap()).unwrap();
        let bitmap_index = U32BitmapIndex::build(right.column(0).unwrap()).unwrap();
        let mut serial = JoinWorkspace::new(2).unwrap();
        let mut parallel = JoinWorkspace::new(2).unwrap();
        let mut bitmap_serial = JoinWorkspace::new(2).unwrap();
        let mut bitmap_parallel = JoinWorkspace::new(2).unwrap();

        join_cpu_serial(
            left.view(),
            right.view(),
            &index,
            path_join_plan(),
            &mut serial,
        )
        .unwrap();
        join_cpu_bitmap_serial(
            left.view(),
            right.view(),
            &bitmap_index,
            path_join_plan(),
            &mut bitmap_serial,
        )
        .unwrap();
        join_cpu_bitmap_parallel(
            left.view(),
            right.view(),
            &bitmap_index,
            path_join_plan(),
            &mut bitmap_parallel,
        )
        .unwrap();
        join_cpu_parallel(
            left.view(),
            right.view(),
            &index,
            path_join_plan(),
            &mut parallel,
        )
        .unwrap();

        assert_eq!(
            serial.output().view().column_slice(0).unwrap(),
            &[10, 10, 20, 30, 30]
        );
        assert_eq!(
            serial.output().view().column_slice(1).unwrap(),
            &[100, 101, 200, 100, 101]
        );
        assert_eq!(
            parallel.output().view().column_slice(0),
            serial.output().view().column_slice(0)
        );
        assert_eq!(
            parallel.output().view().column_slice(1),
            serial.output().view().column_slice(1)
        );
        for output in [bitmap_serial.output(), bitmap_parallel.output()] {
            assert_eq!(
                output.view().column_slice(0),
                serial.output().view().column_slice(0)
            );
            assert_eq!(
                output.view().column_slice(1),
                serial.output().view().column_slice(1)
            );
        }
    }

    #[test]
    fn automatic_join_executes_the_selected_backend() {
        let left = relation2(&[(10, 1), (20, 2), (30, 1), (40, 9)]);
        let right = relation2(&[(1, 100), (1, 101), (2, 200), (3, 300)]);
        let index = U32RangeIndex::build(right.column(0).unwrap()).unwrap();
        let policy = JoinPlacementPolicy {
            gpu_min_delta_rows: 1,
            gpu_unavailable_parallel_min_rows: 1,
        };
        let mut workspace = JoinWorkspace::new(2).unwrap();

        let placement = join_auto(
            left.view(),
            right.view(),
            &index,
            path_join_plan(),
            &mut workspace,
            None,
            policy,
        )
        .unwrap();
        assert_eq!(placement, Placement::CpuParallel);
        assert_eq!(
            workspace.output().view().column_slice(1).unwrap(),
            &[100, 101, 200, 100, 101]
        );

        let stream = CudaStream::new().unwrap();
        let placement = join_auto(
            left.view(),
            right.view(),
            &index,
            path_join_plan(),
            &mut workspace,
            Some(&stream),
            policy,
        )
        .unwrap();
        assert_eq!(placement, Placement::Gpu);
        assert_eq!(
            workspace.output().view().column_slice(1).unwrap(),
            &[100, 101, 200, 100, 101]
        );
    }

    #[test]
    fn cuda_indexed_join_matches_native_output() {
        let left = relation2(&[(10, 1), (20, 2), (30, 1), (40, 9)]);
        let right = relation2(&[(1, 100), (1, 101), (2, 200), (3, 300)]);
        let index = U32RangeIndex::build(right.column(0).unwrap()).unwrap();
        let stream = CudaStream::new().unwrap();
        let mut serial = JoinWorkspace::new(2).unwrap();
        let mut gpu = JoinWorkspace::new(2).unwrap();

        join_cpu_serial(
            left.view(),
            right.view(),
            &index,
            path_join_plan(),
            &mut serial,
        )
        .unwrap();
        join_cuda(
            left.view(),
            right.view(),
            &index,
            path_join_plan(),
            &mut gpu,
            &stream,
        )
        .unwrap();

        assert_eq!(
            gpu.output().view().column_slice(0),
            serial.output().view().column_slice(0)
        );
        assert_eq!(
            gpu.output().view().column_slice(1),
            serial.output().view().column_slice(1)
        );
    }
}
