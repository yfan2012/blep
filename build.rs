use std::process::Command;

fn main() {
    let cuda_src = "cuda/wfa_kernel.cu";
    let ptx_out = format!("{}/wfa_kernel.ptx", std::env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed={}", cuda_src);

    let status = Command::new("nvcc")
        .args(&[
            "--ptx",
            "-o", &ptx_out,
            cuda_src,
            "-arch=sm_80",  // A10 is Ampere (sm_86), sm_80 is compatible
            "--std=c++17",
            "-O3",
        ])
        .status()
        .expect("Failed to run nvcc. Make sure the CUDA toolkit is installed and nvcc is on your PATH.");

    if !status.success() {
        panic!("nvcc failed to compile CUDA kernel");
    }

    println!("cargo:rustc-env=WFA_PTX_PATH={}", ptx_out);
}
