mod test_utils;
use rust_dnn_core::{
    backend::Backend,
    device::Device,
    error::Result,
    ten,
    tensor::{self, Tensor, ten1d, ten2d, ten3d},
};

use crate::test_utils::assert_tensor;

fn test_to_vec<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![0.0, 1.0, 2.0].to_device(device)?;
    assert_eq!(x.to_vec()?, vec![0.0, 1.0, 2.0]);
    Ok(())
}

define_test!(test_to_vec, test_to_vec_cpu, test_to_vec_cuda);

fn test_arange<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::arange(-1..2, device);
    assert_tensor(&x, &ten![-1.0, 0.0, 1.0]);
    Ok(())
}

define_test!(test_arange, test_arange_cpu, test_arange_cuda);

fn test_add<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten2d([[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]]).to_device(device)?;
    let x2 = ten2d([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]).to_device(device)?;
    let y = (x1 + x2)?;
    assert_tensor(&y, &ten2d([[4.0, 6.0, 8.0], [10.0, 12.0, 14.0]]));
    Ok(())
}

define_test!(test_add, test_add_cpu, test_add_cuda);

fn test_add_backward<B: Backend>(device: Device<B>) -> Result<()> {
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
    test_add_backward,
    test_add_backward_cpu,
    test_add_backward_cuda
);

fn test_sub<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]].to_device(device)?;
    let x2 = ten![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]].to_device(device)?;
    let y = (x1 - x2)?;
    assert_tensor(&y, &ten![[-6.0, -6.0, -6.0], [-6.0, -6.0, -6.0]]);
    Ok(())
}

define_test!(test_sub, test_sub_cpu, test_sub_cuda);

fn test_sub_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten2d([[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]])
        .to_device(device)?
        .requires_grad();
    let x2 = ten2d([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])
        .to_device(device)?
        .requires_grad();
    let x3 = ten2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        .to_device(device)?
        .requires_grad();
    let y = ((&x1 - &x2)? * x3)?;
    assert_tensor(&y, &ten2d([[-6.0, -12.0, -18.0], [-24.0, -30.0, -36.0]]));
    let grads = y.backward()?;
    let gx1 = grads.get(&x1).unwrap();
    assert_tensor(gx1, &ten2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
    let gx2 = grads.get(&x2).unwrap();
    assert_tensor(gx2, &ten2d([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]]));
    Ok(())
}

define_test!(
    test_sub_backward,
    test_sub_backward_cpu,
    test_sub_backward_cuda
);

fn test_mul<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]].to_device(device)?;
    let x2 = ten![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]].to_device(device)?;
    let y = (x1 * x2)?;
    assert_tensor(&y, &ten![[-5.0, 0.0, 7.0], [16.0, 27.0, 40.0]]);
    Ok(())
}

define_test!(test_mul, test_mul_cpu, test_mul_cuda);

fn test_mul_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]]
        .to_device(device)?
        .requires_grad();
    let x2 = ten![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]
        .to_device(device)?
        .requires_grad();
    let x3 = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let y = ((&x1 * &x2) * x3)?;
    assert_tensor(&y, &ten![[-5.0, 0.0, 21.0], [64.0, 135.0, 240.0]]);
    let grads = y.backward()?;
    let gx1 = grads.get(&x1).unwrap();
    assert_tensor(gx1, &ten![[5.0, 12.0, 21.0], [32.0, 45.0, 60.0]]);
    let gx2 = grads.get(&x2).unwrap();
    assert_tensor(gx2, &ten![[-1.0, 0.0, 3.0], [8.0, 15.0, 24.0]]);
    Ok(())
}

define_test!(
    test_mul_backward,
    test_mul_backward_cpu,
    test_mul_backward_cuda
);

fn test_div<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]].to_device(device)?;
    let x2 = ten![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]].to_device(device)?;
    let y = (x1 / x2)?;
    assert_tensor(&y, &ten![[-0.2, 0.0, 0.14285715], [0.25, 0.33333334, 0.4]]);
    Ok(())
}

define_test!(test_div, test_div_cpu, test_div_cuda);

fn test_div_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]]
        .to_device(device)?
        .requires_grad();
    let x2 = ten![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]
        .to_device(device)?
        .requires_grad();
    let x3 = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let y = ((&x1 / &x2) * x3)?;
    assert_tensor(&y, &ten![[-0.2, 0.0, 0.42857146], [1.0, 1.6666667, 2.4]]);
    let grads = y.backward()?;
    let gx1 = grads.get(&x1).unwrap();
    assert_tensor(
        gx1,
        &ten![[0.2, 0.33333334, 0.42857143], [0.5, 0.5555556, 0.6]],
    );
    let gx2 = grads.get(&x2).unwrap();
    assert_tensor(
        gx2,
        &ten![[0.04, -0.0, -0.06122449], [-0.125, -0.1851852, -0.24]],
    );
    Ok(())
}

define_test!(
    test_div_backward,
    test_div_backward_cpu,
    test_div_backward_cuda
);

fn test_neg<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[0.0, -2.0, -4.0], [1.0, 2.0, 3.0]].to_device(device)?;
    let y = (-x)?;
    assert_tensor(&y, &ten![[0.0, 2.0, 4.0], [-1.0, -2.0, -3.0]]);
    Ok(())
}

define_test!(test_neg, test_neg_cpu, test_neg_cuda);

fn test_neg_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[0.0, -2.0, -4.0], [1.0, 2.0, 3.0]]
        .to_device(device)?
        .requires_grad();
    let y = (-&x)?;
    assert_tensor(&y, &ten![[0.0, 2.0, 4.0], [-1.0, -2.0, -3.0]]);
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap();
    assert_tensor(&gx, &ten![[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]);
    Ok(())
}

define_test!(
    test_neg_backward,
    test_neg_backward_cpu,
    test_neg_backward_cuda
);

fn test_broadcast_op2<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![-1.0].to_device(device)?;
    let x2 = ten![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]].to_device(device)?;
    let y = (x1 + x2)?;
    assert_tensor(&y, &ten![[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    Ok(())
}

define_test!(
    test_broadcast_op2,
    test_broadcast_op2_cpu,
    test_broadcast_op2_cuda
);

fn test_broadcast_op2_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![-1.0].to_device(device)?.requires_grad();
    let x2 = ten![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]
        .to_device(device)?
        .requires_grad();
    let x3 = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let y = ((&x1 + &x2) * x3)?;
    assert_tensor(&y, &ten![[4.0, 10.0, 18.0], [28.0, 40.0, 54.0]]);
    let grads = y.backward()?;
    let gx1 = grads.get(&x1).unwrap();
    assert_tensor(gx1, &ten![21.0]);
    let gx2 = grads.get(&x2).unwrap();
    assert_tensor(gx2, &ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    Ok(())
}

define_test!(
    test_broadcast_op2_backward,
    test_broadcast_op2_backward_cpu,
    test_broadcast_op2_backward_cuda
);

fn test_pow<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[0.0, -2.0, -4.0], [1.0, 2.0, 3.0]].to_device(device)?;
    let x2 = ten![3.0].to_device(device)?;
    let y = x1.pow(&x2)?;
    assert_tensor(&y, &ten![[0.0, -8.0, -64.0], [1.0, 8.0, 27.0]]);
    Ok(())
}

define_test!(test_pow, test_pow_cpu, test_pow_cuda);

fn test_pow_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let x2 = ten![3.0].to_device(device)?.requires_grad();
    let y = x1.pow(&x2)?;
    assert_tensor(&y, &ten![[1.0, 8.0, 27.0], [64.0, 125.0, 216.0]]);
    let grads = y.backward()?;
    let gx1 = grads.get(&x1).unwrap();
    assert_tensor(&gx1, &ten![[3.0, 12.0, 27.0], [48.0, 75.0, 108.0]]);
    let gx2 = grads.get(&x2).unwrap();
    assert_tensor(&gx2, &ten![6.5792512120101]);
    Ok(())
}

define_test!(
    test_pow_backward,
    test_pow_backward_cpu,
    test_pow_backward_cuda
);

fn test_pow_scalar<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[0.0, -2.0, -4.0], [1.0, 2.0, 3.0]].to_device(device)?;
    let y = x.pow_scalar(3.0)?;
    assert_tensor(&y, &ten![[0.0, -8.0, -64.0], [1.0, 8.0, 27.0]]);
    Ok(())
}

define_test!(test_pow_scalar, test_pow_scalar_cpu, test_pow_scalar_cuda);

fn test_pow_scalar_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[0.0, -2.0, -4.0], [1.0, 2.0, 3.0]]
        .to_device(device)?
        .requires_grad();
    let y = x.pow_scalar(3.0)?;
    assert_tensor(&y, &ten![[0.0, -8.0, -64.0], [1.0, 8.0, 27.0]]);
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap();
    assert_tensor(&gx, &ten![[0.0, 12.0, 48.0], [3.0, 12.0, 27.0]]);
    Ok(())
}

define_test!(
    test_pow_scalar_backward,
    test_pow_scalar_backward_cpu,
    test_pow_scalar_backward_cuda
);

fn test_ln<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![4.0].to_device(device)?;
    let y = x.ln()?;
    assert_tensor(&y, &ten![1.3862944]);
    Ok(())
}

define_test!(test_ln, test_ln_cpu, test_ln_cuda);

fn test_ln_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![4.0].to_device(device)?.requires_grad();
    let y = (x.ln()? * ten![2.0].to_device(device)?)?;
    assert_tensor(&y, &ten![2.772588722239781]);
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap();
    assert_tensor(&gx, &ten![0.5]);
    Ok(())
}

define_test!(
    test_ln_backward,
    test_ln_backward_cpu,
    test_ln_backward_cuda
);

fn test_reshape<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_device(device)?;
    let y = x.reshape([6])?;
    assert_tensor(&y, &ten1d([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    Ok(())
}

define_test!(test_reshape, test_reshape_cpu, test_reshape_cuda);

fn test_reshape2<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten1d([1.0, 2.0, 3.0]).to_device(device)?;
    let x = x.broadcast_to(vec![2, 3])?;
    let y = x.reshape([6])?;
    assert_tensor(&y, &ten1d([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]));
    Ok(())
}

define_test!(test_reshape2, test_reshape2_cpu, test_reshape2_cuda);

fn test_reshape_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        .to_device(device)?
        .requires_grad();
    let x2 = ten1d([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).to_device(device)?;
    let y = (x1.reshape([6])? * x2)?;
    assert_tensor(&y, &ten1d([1.0, 4.0, 9.0, 16.0, 25.0, 36.0]));
    let grads = y.backward()?;
    let gx = grads.get(&x1).unwrap();
    assert_tensor(gx, &ten2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
    Ok(())
}

define_test!(
    test_reshape_backward,
    test_reshape_backward_cpu,
    test_reshape_backward_cuda
);

fn test_reshape_backward2<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten1d([1.0, 2.0, 3.0]).to_device(device)?.requires_grad();
    let x2 = ten2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_device(device)?;
    let y = (x1.broadcast_to2([2, 3])? * x2)?;
    assert_tensor(&y, &ten2d([[1.0, 4.0, 9.0], [4.0, 10.0, 18.0]]));
    let grads = y.backward()?;
    let gx1 = grads.get(&x1).unwrap();
    assert_tensor(gx1, &ten1d([5.0, 7.0, 9.0]));
    Ok(())
}

define_test!(
    test_reshape_backward2,
    test_reshape_backward2_cpu,
    test_reshape_backward2_cuda
);

fn test_permuted_axes<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_device(device)?;
    let y = x.permuted_axes2([1, 0])?;
    assert_tensor(&y, &ten2d([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]));
    Ok(())
}

define_test!(
    test_permuted_axes,
    test_permuted_axes_cpu,
    test_permuted_axes_cuda
);

fn test_permuted_axes_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        .to_device(device)?
        .requires_grad();
    let y = x.permuted_axes2([1, 0])?;
    assert_tensor(&y, &ten2d([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]));
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap();
    assert_tensor(gx, &ten2d([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]));
    Ok(())
}

define_test!(
    test_permuted_axes_backward,
    test_permuted_axes_backward_cpu,
    test_permuted_axes_backward_cuda
);

fn test_reversed_axes<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?;
    let y = x.reversed_axes()?;
    assert_tensor(&y, &ten![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
    Ok(())
}

define_test!(
    test_reversed_axes,
    test_reversed_axes_cpu,
    test_reversed_axes_cuda
);

fn test_broadcast_to<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten1d([1.0, 2.0, 3.0]).to_device(device)?;
    let y = x.broadcast_to2([2, 3])?;
    assert_tensor(&y, &ten2d([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]));
    Ok(())
}

define_test!(
    test_broadcast_to,
    test_broadcast_to_cpu,
    test_broadcast_to_cuda
);

fn test_broadcast_to2<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_device(device)?;
    let y = x.broadcast_to2([2, 3])?;
    assert_tensor(&y, &ten2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
    Ok(())
}

define_test!(
    test_broadcast_to2,
    test_broadcast_to2_cpu,
    test_broadcast_to2_cuda
);

fn test_broadcast_to3<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten1d([1.0, 2.0, 3.0]).to_device(device)?;
    let x = x.broadcast_to2([2, 3])?;
    let y = x.broadcast_to2([2, 3])?;
    assert_tensor(&y, &ten2d([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]));
    Ok(())
}

define_test!(
    test_broadcast_to3,
    test_broadcast_to3_cpu,
    test_broadcast_to3_cuda
);

fn test_broadcast_to4<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten2d([[1.0], [2.0], [3.0]]).to_device(device)?;
    let y = x.broadcast_to2([3, 2])?;
    assert_tensor(&y, &ten2d([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]));
    Ok(())
}

define_test!(
    test_broadcast_to4,
    test_broadcast_to4_cpu,
    test_broadcast_to4_cuda
);

fn test_broadcast_to5<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten2d([[1.0], [2.0], [3.0]]).to_device(device)?;
    let y = x.broadcast_to2([2, 3, 4])?;
    assert_tensor(
        &y,
        &ten3d([
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0],
            ],
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0],
            ],
        ]),
    );
    Ok(())
}

define_test!(
    test_broadcast_to5,
    test_broadcast_to5_cpu,
    test_broadcast_to5_cuda
);

fn test_broadcast_to_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten1d([1.0, 2.0, 3.0]).to_device(device)?.requires_grad();
    let x2 = ten2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_device(device)?;
    let y = (x1.broadcast_to2([2, 3])? * x2)?;
    assert_tensor(&y, &ten2d([[1.0, 4.0, 9.0], [4.0, 10.0, 18.0]]));
    let grads = y.backward()?;
    let gx = grads.get(&x1).unwrap();
    assert_tensor(gx, &ten1d([5.0, 7.0, 9.0]));
    Ok(())
}

define_test!(
    test_broadcast_to_backward,
    test_broadcast_to_backward_cpu,
    test_broadcast_to_backward_cuda
);

fn test_sum_to<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_device(device)?;
    let y = x.sum_to2([1, 3])?;
    assert_tensor(&y, &ten2d([[5.0, 7.0, 9.0]]));
    Ok(())
}

define_test!(test_sum_to, test_sum_to_cpu, test_sum_to_cuda);

fn test_sum_to2<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_device(device)?;
    let y = x.sum_to2([3])?;
    assert_tensor(&y, &ten1d([5.0, 7.0, 9.0]));
    Ok(())
}

define_test!(test_sum_to2, test_sum_to2_cpu, test_sum_to2_cuda);

fn test_sum_to3<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten2d([[1.0, 2.0, 3.0]]).to_device(device)?;
    let x = x.broadcast_to2([4, 3])?;
    let y = x.sum_to2([1, 3])?;
    assert_tensor(&y, &ten2d([[4.0, 8.0, 12.0]]));
    Ok(())
}

define_test!(test_sum_to3, test_sum_to3_cpu, test_sum_to3_cuda);

fn test_sum_to4<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten2d([[1.0, 2.0, 3.0]]).to_device(device)?;
    let x = x.broadcast_to2([4, 3])?;
    let y = x.sum_to2([4, 1])?;
    assert_tensor(&y, &ten2d([[6.0], [6.0], [6.0], [6.0]]));
    Ok(())
}

define_test!(test_sum_to4, test_sum_to4_cpu, test_sum_to4_cuda);

fn test_sum_to5<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_device(device)?;
    let y = x.sum_to2([1])?;
    assert_tensor(&y, &ten1d([21.0]));
    Ok(())
}

define_test!(test_sum_to5, test_sum_to5_cpu, test_sum_to5_cuda);

fn test_sum_to6<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?;
    let y = x.sum_to2([2, 1])?;
    assert_tensor(&y, &ten2d([[6.0], [15.0]]));
    Ok(())
}

define_test!(test_sum_to6, test_sum_to6_cpu, test_sum_to6_cuda);

fn test_sum_to_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        .to_device(device)?
        .requires_grad();
    let y = x.sum_to2([3])?;
    assert_tensor(&y, &ten1d([5.0, 7.0, 9.0]));
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap();
    assert_tensor(gx, &ten2d([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]));
    Ok(())
}

define_test!(
    test_sum_to_backward,
    test_sum_to_backward_cpu,
    test_sum_to_backward_cuda
);

fn test_sum_axis<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?;
    let y = x.sum_axis(0, false)?;
    assert_tensor(&y, &ten![5.0, 7.0, 9.0]);
    Ok(())
}

define_test!(test_sum_axis, test_sum_axis_cpu, test_sum_axis_cuda);

fn test_sum_axis_keepdims_true<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_device(device)?;
    let y = x.sum_axis2(0)?;
    assert_tensor(&y, &ten2d([[5.0, 7.0, 9.0]]));
    Ok(())
}

define_test!(
    test_sum_axis_keepdims_true,
    test_sum_axis_keepdims_true_cpu,
    test_sum_axis_keepdims_true_cuda
);

fn test_sum_axis_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let y = x.sum_axis(0, false)?;
    assert_tensor(&y, &ten![5.0, 7.0, 9.0]);
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap();
    assert_tensor(gx, &ten![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);
    Ok(())
}

define_test!(
    test_sum_axis_backward,
    test_sum_axis_backward_cpu,
    test_sum_axis_backward_cuda
);

fn test_sum_axis_backward_keepdims_true<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        .to_device(device)?
        .requires_grad();
    let y = x.sum_axis(0, true)?;
    assert_tensor(&y, &ten2d([[5.0, 7.0, 9.0]]));
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap();
    assert_tensor(gx, &ten2d([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]));
    Ok(())
}

define_test!(
    test_sum_axis_backward_keepdims_true,
    test_sum_axis_backward_keepdims_true_cpu,
    test_sum_axis_backward_keepdims_true_cuda
);

fn test_contiguous<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten1d([1.0, 2.0, 3.0]).to_device(device)?;
    let x = x.broadcast_to2([2, 3])?;
    assert!(!x.is_contiguous());
    let y = x.contiguous()?;
    assert!(y.is_contiguous());
    assert_tensor(&y, &ten2d([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]));
    Ok(())
}

define_test!(test_contiguous, test_contiguous_cpu, test_contiguous_cuda);

fn test_contiguous_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten1d([1.0, 2.0, 3.0]).to_device(device)?.requires_grad();
    let h = x.broadcast_to2([2, 3])?;
    assert!(!h.is_contiguous());
    let y = h.contiguous()?;
    assert!(y.is_contiguous());
    assert_tensor(&y, &ten2d([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]));
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap();
    assert_tensor(gx, &ten1d([2.0, 2.0, 2.0]));
    Ok(())
}

define_test!(
    test_contiguous_backward,
    test_contiguous_backward_cpu,
    test_contiguous_backward_cuda
);
