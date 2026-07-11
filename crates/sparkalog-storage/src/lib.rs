//! Canonical relation storage shared by native Rust and CUDA operators.

use std::ffi::c_void;
use std::fmt;
use std::ptr::NonNull;

const CUDA_SUCCESS: i32 = 0;
const CUDA_MEM_ATTACH_GLOBAL: u32 = 1;

unsafe extern "C" {
    fn cudaMallocManaged(allocation: *mut *mut c_void, bytes: usize, flags: u32) -> i32;
    fn cudaFree(allocation: *mut c_void) -> i32;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Cuda { operation: &'static str, code: i32 },
    LengthOverflow,
    ZeroSizedType,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda { operation, code } => {
                write!(formatter, "{operation} failed with CUDA error {code}")
            }
            Self::LengthOverflow => formatter.write_str("buffer length overflows its byte size"),
            Self::ZeroSizedType => {
                formatter.write_str("CUDA-managed buffers cannot contain zero-sized types")
            }
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn cuda_result(operation: &'static str, code: i32) -> Result<()> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Error::Cuda { operation, code })
    }
}

/// A CUDA-managed allocation that is the canonical storage for both CPU and
/// GPU operators.
///
/// Access must be externally synchronized: Rust code must not create a slice
/// while CUDA work using this allocation is still in flight.
pub struct ManagedBuffer<T> {
    data: NonNull<T>,
    len: usize,
}

impl<T: Copy> ManagedBuffer<T> {
    pub fn new_filled(len: usize, value: T) -> Result<Self> {
        if std::mem::size_of::<T>() == 0 {
            return Err(Error::ZeroSizedType);
        }
        if len == 0 {
            return Ok(Self {
                data: NonNull::dangling(),
                len: 0,
            });
        }

        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(Error::LengthOverflow)?;
        let mut allocation = std::ptr::null_mut();
        // SAFETY: CUDA writes an allocation of `bytes` on success.
        unsafe {
            cuda_result(
                "cudaMallocManaged",
                cudaMallocManaged(&mut allocation, bytes, CUDA_MEM_ATTACH_GLOBAL),
            )?;
        }
        let data = NonNull::new(allocation.cast::<T>())
            .expect("successful CUDA managed allocation returned null");

        // SAFETY: the allocation contains `len` properly aligned T slots and
        // each slot is initialized before a reference to it is exposed.
        unsafe {
            for index in 0..len {
                data.as_ptr().add(index).write(value);
            }
        }

        Ok(Self { data, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[T] {
        // SAFETY: all slots are initialized and the shared borrow prevents host mutation.
        unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: all slots are initialized and the mutable borrow guarantees exclusive host access.
        unsafe { std::slice::from_raw_parts_mut(self.data.as_ptr(), self.len) }
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_ptr()
    }
}

impl<T> Drop for ManagedBuffer<T> {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        // SAFETY: `data` was allocated by CUDA and is owned by this wrapper.
        unsafe {
            let _ = cudaFree(self.data.as_ptr().cast());
        }
    }
}

pub type Column = ManagedBuffer<u32>;

/// The first concrete canonical relation shape used by Sparkalog operators.
pub struct Relation2 {
    pub left: Column,
    pub right: Column,
}

impl Relation2 {
    pub fn new(len: usize) -> Result<Self> {
        Ok(Self {
            left: Column::new_filled(len, 0)?,
            right: Column::new_filled(len, 0)?,
        })
    }

    pub fn len(&self) -> usize {
        self.left.len()
    }

    pub fn is_empty(&self) -> bool {
        self.left.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_buffer_needs_no_cuda_allocation() {
        let buffer = ManagedBuffer::<u32>::new_filled(0, 7).unwrap();
        assert!(buffer.is_empty());
        assert_eq!(buffer.as_slice(), &[]);
    }

    #[test]
    fn zero_sized_types_are_rejected() {
        assert!(matches!(
            ManagedBuffer::new_filled(1, ()),
            Err(Error::ZeroSizedType)
        ));
    }
}
