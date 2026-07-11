//! Canonical relation storage shared by native Rust and CUDA operators.

mod io;

pub use io::{LoadError, load_binary_u32};

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
    LogicalLengthExceedsCapacity { len: usize, capacity: usize },
    MismatchedColumnLength { expected: usize, actual: usize },
    TooManyRows(usize),
    ZeroSizedType,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda { operation, code } => {
                write!(formatter, "{operation} failed with CUDA error {code}")
            }
            Self::LengthOverflow => formatter.write_str("buffer length overflows its byte size"),
            Self::LogicalLengthExceedsCapacity { len, capacity } => write!(
                formatter,
                "logical length {len} exceeds allocation capacity {capacity}"
            ),
            Self::MismatchedColumnLength { expected, actual } => write!(
                formatter,
                "column length {actual} does not match relation length {expected}"
            ),
            Self::TooManyRows(rows) => {
                write!(
                    formatter,
                    "{rows} rows cannot be represented by u32 row IDs"
                )
            }
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

    pub fn as_ptr(&self) -> *const T {
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

/// An arbitrary-arity canonical structure-of-arrays relation.
pub struct Relation {
    columns: Vec<Column>,
}

impl Relation {
    pub fn new(arity: usize, len: usize) -> Result<Self> {
        let mut columns = Vec::with_capacity(arity);
        for _ in 0..arity {
            columns.push(Column::new_filled(len, 0)?);
        }
        Ok(Self { columns })
    }

    pub fn from_columns(columns: Vec<Column>) -> Result<Self> {
        let expected = columns.first().map_or(0, ManagedBuffer::len);
        if let Some(actual) = columns
            .iter()
            .map(ManagedBuffer::len)
            .find(|&len| len != expected)
        {
            return Err(Error::MismatchedColumnLength { expected, actual });
        }
        Ok(Self { columns })
    }

    pub fn len(&self) -> usize {
        self.columns.first().map_or(0, ManagedBuffer::len)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn arity(&self) -> usize {
        self.columns.len()
    }

    pub fn column(&self, index: usize) -> Option<&Column> {
        self.columns.get(index)
    }

    pub fn column_mut(&mut self, index: usize) -> Option<&mut Column> {
        self.columns.get_mut(index)
    }

    pub fn view(&self) -> RelationView<'_> {
        RelationView {
            columns: &self.columns,
            len: self.len(),
        }
    }
}

/// A zero-copy borrowed view of a canonical relation.
#[derive(Clone, Copy)]
pub struct RelationView<'a> {
    columns: &'a [Column],
    len: usize,
}

impl<'a> RelationView<'a> {
    pub fn len(self) -> usize {
        self.len
    }

    pub fn is_empty(self) -> bool {
        self.len == 0
    }

    pub fn arity(self) -> usize {
        self.columns.len()
    }

    pub fn column(self, index: usize) -> Option<&'a Column> {
        self.columns.get(index)
    }

    pub fn column_slice(self, index: usize) -> Option<&'a [u32]> {
        self.columns
            .get(index)
            .map(|column| &column.as_slice()[..self.len])
    }
}

/// A capacity-backed relation used for operator output.
pub struct RelationBuffer {
    columns: Vec<Column>,
    len: usize,
    capacity: usize,
}

impl RelationBuffer {
    pub fn with_capacity(arity: usize, capacity: usize) -> Result<Self> {
        let mut columns = Vec::with_capacity(arity);
        for _ in 0..arity {
            columns.push(Column::new_filled(capacity, 0)?);
        }
        Ok(Self {
            columns,
            len: 0,
            capacity,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn arity(&self) -> usize {
        self.columns.len()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn set_len(&mut self, len: usize) -> Result<()> {
        if len > self.capacity {
            return Err(Error::LogicalLengthExceedsCapacity {
                len,
                capacity: self.capacity,
            });
        }
        self.len = len;
        Ok(())
    }

    pub fn reserve(&mut self, required: usize) -> Result<()> {
        if required <= self.capacity {
            return Ok(());
        }
        let capacity = required.checked_next_power_of_two().unwrap_or(required);
        for column in &mut self.columns {
            *column = Column::new_filled(capacity, 0)?;
        }
        self.capacity = capacity;
        self.len = 0;
        Ok(())
    }

    pub fn column(&self, index: usize) -> Option<&Column> {
        self.columns.get(index)
    }

    pub fn column_mut(&mut self, index: usize) -> Option<&mut Column> {
        self.columns.get_mut(index)
    }

    pub fn columns_mut(&mut self) -> &mut [Column] {
        &mut self.columns
    }

    pub fn view(&self) -> RelationView<'_> {
        RelationView {
            columns: &self.columns,
            len: self.len,
        }
    }
}

/// A sparse sorted range index mapping `u32` values to canonical row IDs.
pub struct U32RangeIndex {
    keys: ManagedBuffer<u32>,
    starts: ManagedBuffer<u32>,
    rows: ManagedBuffer<u32>,
    source_rows: usize,
}

impl U32RangeIndex {
    pub fn build(column: &Column) -> Result<Self> {
        let source_rows = column.len();
        if source_rows > u32::MAX as usize {
            return Err(Error::TooManyRows(source_rows));
        }
        let mut rows = ManagedBuffer::new_filled(source_rows, 0_u32)?;
        for (row, output) in rows.as_mut_slice().iter_mut().enumerate() {
            *output = row as u32;
        }
        let values = column.as_slice();
        rows.as_mut_slice()
            .sort_unstable_by_key(|&row| (values[row as usize], row));

        let unique = rows
            .as_slice()
            .windows(2)
            .filter(|pair| values[pair[0] as usize] != values[pair[1] as usize])
            .count()
            + usize::from(source_rows != 0);
        let mut keys = ManagedBuffer::new_filled(unique, 0_u32)?;
        let mut starts = ManagedBuffer::new_filled(unique + 1, 0_u32)?;
        let mut key_index = 0;
        for (sorted_position, &row) in rows.as_slice().iter().enumerate() {
            let key = values[row as usize];
            if sorted_position == 0 || keys.as_slice()[key_index - 1] != key {
                keys.as_mut_slice()[key_index] = key;
                starts.as_mut_slice()[key_index] = sorted_position as u32;
                key_index += 1;
            }
        }
        starts.as_mut_slice()[unique] = source_rows as u32;
        Ok(Self {
            keys,
            starts,
            rows,
            source_rows,
        })
    }

    pub fn source_rows(&self) -> usize {
        self.source_rows
    }

    pub fn unique_keys(&self) -> usize {
        self.keys.len()
    }

    pub fn keys(&self) -> &ManagedBuffer<u32> {
        &self.keys
    }

    pub fn starts(&self) -> &ManagedBuffer<u32> {
        &self.starts
    }

    pub fn rows(&self) -> &ManagedBuffer<u32> {
        &self.rows
    }

    pub fn lookup(&self, key: u32) -> &[u32] {
        let Ok(index) = self.keys.as_slice().binary_search(&key) else {
            return &[];
        };
        let start = self.starts.as_slice()[index] as usize;
        let end = self.starts.as_slice()[index + 1] as usize;
        &self.rows.as_slice()[start..end]
    }
}

/// Reusable count, offset, output, and temporary storage for binary joins.
pub struct JoinWorkspace {
    output: RelationBuffer,
    counts: ManagedBuffer<u64>,
    offsets: ManagedBuffer<u64>,
    total: ManagedBuffer<u64>,
    temporary: ManagedBuffer<u8>,
}

impl JoinWorkspace {
    pub fn new(output_arity: usize) -> Result<Self> {
        Ok(Self {
            output: RelationBuffer::with_capacity(output_arity, 0)?,
            counts: ManagedBuffer::new_filled(0, 0_u64)?,
            offsets: ManagedBuffer::new_filled(0, 0_u64)?,
            total: ManagedBuffer::new_filled(1, 0_u64)?,
            temporary: ManagedBuffer::new_filled(0, 0_u8)?,
        })
    }

    pub fn reserve_outer_rows(&mut self, required: usize) -> Result<()> {
        reserve_managed_u64(&mut self.counts, required)?;
        reserve_managed_u64(&mut self.offsets, required)?;
        Ok(())
    }

    pub fn reserve_output_rows(&mut self, required: usize) -> Result<()> {
        self.output.reserve(required)
    }

    pub fn reserve_temporary_bytes(&mut self, required: usize) -> Result<()> {
        if required > self.temporary.len() {
            let capacity = required.checked_next_power_of_two().unwrap_or(required);
            self.temporary = ManagedBuffer::new_filled(capacity, 0_u8)?;
        }
        Ok(())
    }

    pub fn output(&self) -> &RelationBuffer {
        &self.output
    }

    pub fn output_mut(&mut self) -> &mut RelationBuffer {
        &mut self.output
    }

    pub fn counts(&self) -> &ManagedBuffer<u64> {
        &self.counts
    }

    pub fn counts_mut(&mut self) -> &mut ManagedBuffer<u64> {
        &mut self.counts
    }

    pub fn offsets(&self) -> &ManagedBuffer<u64> {
        &self.offsets
    }

    pub fn offsets_mut(&mut self) -> &mut ManagedBuffer<u64> {
        &mut self.offsets
    }

    pub fn total(&self) -> &ManagedBuffer<u64> {
        &self.total
    }

    pub fn total_mut(&mut self) -> &mut ManagedBuffer<u64> {
        &mut self.total
    }

    pub fn temporary_mut(&mut self) -> &mut ManagedBuffer<u8> {
        &mut self.temporary
    }

    pub fn cuda_count_parts(
        &mut self,
    ) -> (
        &mut ManagedBuffer<u64>,
        &mut ManagedBuffer<u64>,
        &mut ManagedBuffer<u64>,
        &mut ManagedBuffer<u8>,
    ) {
        (
            &mut self.counts,
            &mut self.offsets,
            &mut self.total,
            &mut self.temporary,
        )
    }
}

/// Compact row identifiers backed by a reusable managed allocation.
pub struct Selection {
    rows: ManagedBuffer<u32>,
    len: usize,
}

impl Selection {
    pub fn with_capacity(capacity: usize) -> Result<Self> {
        Ok(Self {
            rows: ManagedBuffer::new_filled(capacity, 0)?,
            len: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.rows.len()
    }

    pub fn as_slice(&self) -> &[u32] {
        &self.rows.as_slice()[..self.len]
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn set_len(&mut self, len: usize) -> Result<()> {
        if len > self.capacity() {
            return Err(Error::LogicalLengthExceedsCapacity {
                len,
                capacity: self.capacity(),
            });
        }
        self.len = len;
        Ok(())
    }

    pub fn reserve(&mut self, required: usize) -> Result<()> {
        if required <= self.capacity() {
            return Ok(());
        }
        let capacity = required.checked_next_power_of_two().unwrap_or(required);
        self.rows = ManagedBuffer::new_filled(capacity, 0)?;
        self.len = 0;
        Ok(())
    }

    pub fn storage(&self) -> &ManagedBuffer<u32> {
        &self.rows
    }

    pub fn storage_mut(&mut self) -> &mut ManagedBuffer<u32> {
        &mut self.rows
    }
}

/// Reusable buffers for selection and compaction operators.
pub struct OperatorWorkspace {
    selection: Selection,
    flags: ManagedBuffer<u32>,
    offsets: ManagedBuffer<u32>,
    count: ManagedBuffer<u32>,
    temporary: ManagedBuffer<u8>,
}

impl OperatorWorkspace {
    pub fn new() -> Result<Self> {
        Ok(Self {
            selection: Selection::with_capacity(0)?,
            flags: ManagedBuffer::new_filled(0, 0)?,
            offsets: ManagedBuffer::new_filled(0, 0)?,
            count: ManagedBuffer::new_filled(1, 0)?,
            temporary: ManagedBuffer::new_filled(0, 0)?,
        })
    }

    pub fn reserve_rows(&mut self, required: usize) -> Result<()> {
        self.selection.reserve(required)?;
        reserve_managed(&mut self.flags, required)?;
        reserve_managed(&mut self.offsets, required)?;
        Ok(())
    }

    pub fn reserve_temporary_bytes(&mut self, required: usize) -> Result<()> {
        if required <= self.temporary.len() {
            return Ok(());
        }
        let capacity = required.checked_next_power_of_two().unwrap_or(required);
        self.temporary = ManagedBuffer::new_filled(capacity, 0)?;
        Ok(())
    }

    pub fn selection(&self) -> &Selection {
        &self.selection
    }

    pub fn selection_mut(&mut self) -> &mut Selection {
        &mut self.selection
    }

    pub fn flags(&self) -> &ManagedBuffer<u32> {
        &self.flags
    }

    pub fn flags_mut(&mut self) -> &mut ManagedBuffer<u32> {
        &mut self.flags
    }

    pub fn offsets(&self) -> &ManagedBuffer<u32> {
        &self.offsets
    }

    pub fn offsets_mut(&mut self) -> &mut ManagedBuffer<u32> {
        &mut self.offsets
    }

    pub fn count(&self) -> &ManagedBuffer<u32> {
        &self.count
    }

    pub fn count_mut(&mut self) -> &mut ManagedBuffer<u32> {
        &mut self.count
    }

    pub fn temporary(&self) -> &ManagedBuffer<u8> {
        &self.temporary
    }

    pub fn temporary_mut(&mut self) -> &mut ManagedBuffer<u8> {
        &mut self.temporary
    }

    pub fn cpu_compaction_parts(
        &mut self,
    ) -> (
        &mut Selection,
        &mut ManagedBuffer<u32>,
        &mut ManagedBuffer<u32>,
    ) {
        (&mut self.selection, &mut self.flags, &mut self.offsets)
    }

    pub fn cuda_compaction_parts(
        &mut self,
    ) -> (
        &mut Selection,
        &mut ManagedBuffer<u32>,
        &mut ManagedBuffer<u32>,
        &mut ManagedBuffer<u8>,
    ) {
        (
            &mut self.selection,
            &mut self.flags,
            &mut self.count,
            &mut self.temporary,
        )
    }
}

fn reserve_managed(buffer: &mut ManagedBuffer<u32>, required: usize) -> Result<()> {
    if required <= buffer.len() {
        return Ok(());
    }
    let capacity = required.checked_next_power_of_two().unwrap_or(required);
    *buffer = ManagedBuffer::new_filled(capacity, 0)?;
    Ok(())
}

fn reserve_managed_u64(buffer: &mut ManagedBuffer<u64>, required: usize) -> Result<()> {
    if required <= buffer.len() {
        return Ok(());
    }
    let capacity = required.checked_next_power_of_two().unwrap_or(required);
    *buffer = ManagedBuffer::new_filled(capacity, 0_u64)?;
    Ok(())
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

    #[test]
    fn relation_view_borrows_canonical_columns() {
        let mut relation = Relation::new(2, 3).unwrap();
        relation.column_mut(1).unwrap().as_mut_slice()[2] = 42;

        let view = relation.view();
        assert_eq!(view.arity(), 2);
        assert_eq!(view.len(), 3);
        assert_eq!(view.column(1).unwrap().as_slice(), &[0, 0, 42]);
    }

    #[test]
    fn relation_rejects_columns_of_different_lengths() {
        let columns = vec![
            Column::new_filled(2, 0).unwrap(),
            Column::new_filled(3, 0).unwrap(),
        ];
        assert!(matches!(
            Relation::from_columns(columns),
            Err(Error::MismatchedColumnLength {
                expected: 2,
                actual: 3
            })
        ));
    }

    #[test]
    fn selection_separates_length_from_capacity() {
        let mut selection = Selection::with_capacity(8).unwrap();
        selection.storage_mut().as_mut_slice()[..3].copy_from_slice(&[1, 4, 7]);
        selection.set_len(3).unwrap();

        assert_eq!(selection.capacity(), 8);
        assert_eq!(selection.as_slice(), &[1, 4, 7]);
        assert!(selection.set_len(9).is_err());
    }

    #[test]
    fn workspace_reuses_sufficient_allocations() {
        let mut workspace = OperatorWorkspace::new().unwrap();
        workspace.reserve_rows(10).unwrap();
        let selection = workspace.selection().storage().as_ptr();
        let flags = workspace.flags().as_ptr();
        let offsets = workspace.offsets().as_ptr();

        workspace.reserve_rows(8).unwrap();

        assert_eq!(workspace.selection().storage().as_ptr(), selection);
        assert_eq!(workspace.flags().as_ptr(), flags);
        assert_eq!(workspace.offsets().as_ptr(), offsets);
    }

    #[test]
    fn sparse_range_index_returns_canonical_rows() {
        let mut column = Column::new_filled(6, 0).unwrap();
        column
            .as_mut_slice()
            .copy_from_slice(&[90, 7, 90, 42, 7, 90]);
        let index = U32RangeIndex::build(&column).unwrap();

        assert_eq!(index.unique_keys(), 3);
        assert_eq!(index.lookup(7), &[1, 4]);
        assert_eq!(index.lookup(42), &[3]);
        let mut rows = index.lookup(90).to_vec();
        rows.sort_unstable();
        assert_eq!(rows, [0, 2, 5]);
        assert!(index.lookup(8).is_empty());
    }

    #[test]
    fn relation_buffer_separates_length_and_capacity() {
        let mut output = RelationBuffer::with_capacity(2, 4).unwrap();
        output.columns_mut()[0].as_mut_slice()[..2].copy_from_slice(&[1, 2]);
        output.columns_mut()[1].as_mut_slice()[..2].copy_from_slice(&[3, 4]);
        output.set_len(2).unwrap();

        assert_eq!(output.view().column_slice(0).unwrap(), &[1, 2]);
        assert_eq!(output.view().column_slice(1).unwrap(), &[3, 4]);
        assert_eq!(output.capacity(), 4);
    }
}
