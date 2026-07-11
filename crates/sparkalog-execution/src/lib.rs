//! Physical execution, synchronization, and CPU/GPU placement.

use sparkalog_relational::U32Predicate;
use sparkalog_storage::{Column, ManagedBuffer, OperatorWorkspace};
use std::ffi::c_void;
use std::fmt;
use std::ptr::NonNull;
use std::thread;

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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    CpuSerial,
    CpuParallel,
    Gpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Cuda { operation: &'static str, code: i32 },
    Storage(sparkalog_storage::Error),
    TooManyRows(usize),
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

    let workers = thread::available_parallelism()
        .map_or(1, usize::from)
        .min(rows);
    if workers == 1 {
        return filter_cpu_serial(column, predicate, workspace);
    }

    workspace.reserve_rows(rows)?;
    let chunk_size = rows.div_ceil(workers);
    let chunk_count = rows.div_ceil(chunk_size);
    let (selection, flags, offsets) = workspace.cpu_compaction_parts();
    selection.clear();
    let flags = &mut flags.as_mut_slice()[..rows];

    thread::scope(|scope| {
        for (values, output_flags) in column
            .as_slice()
            .chunks(chunk_size)
            .zip(flags.chunks_mut(chunk_size))
        {
            scope.spawn(move || {
                for (&value, flag) in values.iter().zip(output_flags) {
                    *flag = u32::from(predicate.matches(value));
                }
            });
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
    thread::scope(|scope| {
        for (chunk_index, flag_chunk) in flags.chunks(chunk_size).enumerate() {
            let output_len = counts[chunk_index] as usize;
            let (output, remaining) = remaining_output.split_at_mut(output_len);
            remaining_output = remaining;
            let row_start = chunk_index * chunk_size;
            scope.spawn(move || {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
