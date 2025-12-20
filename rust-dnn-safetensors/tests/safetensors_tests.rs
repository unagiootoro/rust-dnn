use std::collections::HashMap;

use rust_dnn_core::error::Result;
use rust_dnn_core::num::Num;
use rust_dnn_core::tensor::Tensor;
use rust_dnn_core::{cpu_backend::CpuBackend, device::Device};
use rust_dnn_safetensors::{deserialize, serialize};

fn test_serialize<T: Num>() -> Result<()> {
    let device = Device::get_cpu_device();
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3], device)
        .to_dtype::<T>()?
        .requires_grad();
    let y = Tensor::from_vec(vec![4.0, 5.0, 6.0, 7.0], vec![4], device)
        .to_dtype::<T>()?
        .requires_grad();
    let mut tensors = HashMap::new();
    tensors.insert("x".to_string(), x.clone());
    tensors.insert("y".to_string(), y.clone());
    let bytes = serialize(tensors)?;
    let tensors2 = deserialize::<_, T>(bytes, device)?;
    let x2 = &tensors2["x"];
    let y2 = &tensors2["y"];
    assert_eq!(x.shape(), x2.shape());
    assert_eq!(x.to_vec(), x2.to_vec());
    assert_eq!(y.shape(), y2.shape());
    assert_eq!(y.to_vec(), y2.to_vec());
    Ok(())
}

#[test]
fn test_serialize_f32() -> Result<()> {
    test_serialize::<f32>()
}

#[test]
fn test_serialize_f64() -> Result<()> {
    test_serialize::<f64>()
}
