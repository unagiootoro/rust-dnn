mod test_utils;
use rust_dnn_core::{
    backend::Backend,
    device::Device,
    error::Result,
    ten,
    tensor::{self},
};

use crate::test_utils::assert_tensor;

fn test_add_u32<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[0, 1, 2], [2, 3, 4]].to_device(device)?;
    let x2 = ten![[5, 6, 7], [8, 9, 10]].to_device(device)?;
    let y = (x1 + x2)?;
    assert_tensor(&y, &ten![[5, 7, 9], [10, 12, 14]]);
    Ok(())
}

define_test!(test_add_u32, test_add_u32_cpu, test_add_u32_cuda);

fn test_add_f32<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]].to_device(device)?;
    let x2 = ten![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]].to_device(device)?;
    let y = (x1 + x2)?;
    assert_tensor(&y, &ten![[4.0, 6.0, 8.0], [10.0, 12.0, 14.0]]);
    Ok(())
}

define_test!(test_add_f32, test_add_f32_cpu, test_add_f32_cuda);

fn test_add_backward_f32<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]]
        .to_device(device)?
        .requires_grad();
    let x2 = ten![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]
        .to_device(device)?
        .requires_grad();
    let x3 = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let y = ((&x1 + &x2) * x3)?;
    assert_tensor(&y, &ten![[4.0, 12.0, 24.0], [40.0, 60.0, 84.0]]);
    let grads = y.backward()?;
    let gx1 = grads.get(&x1).unwrap();
    assert_tensor(gx1, &ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let gx2 = grads.get(&x2).unwrap();
    assert_tensor(gx2, &ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    Ok(())
}

define_test!(
    test_add_backward_f32,
    test_add_backward_f32_cpu,
    test_add_backward_f32_cuda
);
