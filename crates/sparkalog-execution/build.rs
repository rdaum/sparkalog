use std::path::PathBuf;
use std::process::Command;

fn main() {
    let cuda_root = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda-13.0".to_owned());
    let cuda_root = PathBuf::from(cuda_root);
    let nvcc = cuda_root.join("bin/nvcc");
    let include = cuda_root.join("targets/sbsa-linux/include");
    let cccl_include = include.join("cccl");
    let lib = cuda_root.join("targets/sbsa-linux/lib");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let object = out_dir.join("column_ops.o");
    let archive = out_dir.join("libsparkalog_cuda.a");

    let status = Command::new(&nvcc)
        .args([
            "-std=c++17",
            "-O3",
            "--generate-code=arch=compute_121,code=sm_121",
            "-I",
        ])
        .arg(&include)
        .arg("-I")
        .arg(&cccl_include)
        .args(["-c", "native/column_ops.cu", "-o"])
        .arg(&object)
        .status()
        .expect("failed to run nvcc; set CUDA_HOME to a CUDA 13.0+ installation");
    assert!(
        status.success(),
        "nvcc failed to compile native/column_ops.cu"
    );

    let status = Command::new("ar")
        .args(["crs"])
        .arg(&archive)
        .arg(&object)
        .status()
        .expect("failed to run ar");
    assert!(status.success(), "ar failed to archive CUDA object");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=static=sparkalog_cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rerun-if-changed=native/column_ops.cu");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
}
