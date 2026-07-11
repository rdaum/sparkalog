//! Physical execution, synchronization, and CPU/GPU placement.

use sparkalog_storage::ManagedBuffer;
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    CpuSerial,
    CpuParallel,
    Gpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error {
    operation: &'static str,
    code: i32,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} failed with CUDA error {}",
            self.operation, self.code
        )
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn cuda_result(operation: &'static str, code: i32) -> Result<()> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Error { operation, code })
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
