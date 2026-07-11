use sparkalog_execution::{CudaStream, add_one_i32};
use sparkalog_storage::ManagedBuffer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut column = ManagedBuffer::new_filled(1 << 20, 0_i32)?;
    for (index, value) in column.as_mut_slice().iter_mut().enumerate() {
        *value = index as i32;
    }

    let stream = CudaStream::new()?;
    add_one_i32(&mut column, &stream)?.wait()?;

    for (index, value) in column.as_slice().iter().enumerate() {
        assert_eq!(*value, index as i32 + 1);
    }
    println!(
        "CPU filled and verified {} shared i32 values after a CUDA kernel; no cudaMemcpy occurred.",
        column.len()
    );
    Ok(())
}
