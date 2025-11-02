use std::process::Command;

const ENABLE_WINDOWS_BLAS: bool = false;
const ENABLE_LINUX_BLAS: bool = false;
const ENABLE_CUDA: bool = true;
const ENABLE_CUDNN: bool = false;

fn main() {
    if ENABLE_WINDOWS_BLAS {
        println!("cargo:rustc-link-lib=static=libopenblas");
    } else if ENABLE_LINUX_BLAS {
        println!("cargo:rustc-link-lib=static=openblas");
    }

    if ENABLE_CUDA {
        // Makefile で CUDA をビルド
        if let Ok(status) = Command::new("make").status() {
            if !status.success() {
                println!("Make failed");
                return;
            }
        } else {
            println!("Failed to run make");
            return;
        }

        // ライブラリのリンクディレクトリとリンク対象指定
        println!("cargo:rustc-link-search=native=target");
        println!("cargo:rustc-link-lib=static=kernel");

        // CUDAランタイムライブラリをリンク
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=dylib=stdc++");
        if ENABLE_CUDNN {
            println!("cargo:rustc-link-lib=cudnn");
        }

        // CUDAライブラリパス（環境によって変更してください）
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    }
}
