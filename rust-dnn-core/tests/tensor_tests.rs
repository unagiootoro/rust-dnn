mod test_utils;
use rust_dnn_core::{error::Result, tensor};

use crate::test_utils::assert_tensor;

#[test]
fn test_arange() -> Result<()> {
    let x = tensor::Tensor::<f32>::arange(-1..2);
    assert_tensor(&x, &tensor![-1.0, 0.0, 1.0]);
    Ok(())
}

#[test]
fn test_add() -> Result<()> {
    let x1 = tensor![[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]];
    let x2 = tensor![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]];
    let y = (x1 + x2)?;
    assert_tensor(&y, &tensor![[4.0, 6.0, 8.0], [10.0, 12.0, 14.0]]);
    Ok(())
}

#[test]
fn test_add_backward() -> Result<()> {
    let x1 = tensor![[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]].requires_grad();
    let x2 = tensor![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]].requires_grad();
    let x3 = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].requires_grad();
    let y = ((&x1 + &x2) * x3)?;
    assert_tensor(&y, &tensor![[4.0, 12.0, 24.0], [40.0, 60.0, 84.0]]);
    let grads = y.backward()?;
    let gx1 = grads.get(&x1).unwrap();
    assert_tensor(gx1, &tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let gx2 = grads.get(&x2).unwrap();
    assert_tensor(gx2, &tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    Ok(())
}

#[test]
fn test_sub() -> Result<()> {
    let x1 = tensor![[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]];
    let x2 = tensor![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]];
    let y = (x1 - x2)?;
    assert_tensor(&y, &tensor![[-6.0, -6.0, -6.0], [-6.0, -6.0, -6.0]]);
    Ok(())
}

#[test]
fn test_sub_backward() -> Result<()> {
    let x1 = tensor![[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]].requires_grad();
    let x2 = tensor![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]].requires_grad();
    let x3 = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].requires_grad();
    let y = ((&x1 - &x2)? * x3)?;
    assert_tensor(&y, &tensor![[-6.0, -12.0, -18.0], [-24.0, -30.0, -36.0]]);
    let grads = y.backward()?;
    let gx1 = grads.get(&x1).unwrap();
    assert_tensor(gx1, &tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let gx2 = grads.get(&x2).unwrap();
    assert_tensor(gx2, &tensor![[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]]);
    Ok(())
}

#[test]
fn test_mul() -> Result<()> {
    let x1 = tensor![[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]];
    let x2 = tensor![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]];
    let y = (x1 * x2)?;
    assert_tensor(&y, &tensor![[-5.0, 0.0, 7.0], [16.0, 27.0, 40.0]]);
    Ok(())
}

#[test]
fn test_mul_backward() -> Result<()> {
    let x1 = tensor![[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]].requires_grad();
    let x2 = tensor![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]].requires_grad();
    let x3 = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].requires_grad();
    let y = ((&x1 * &x2) * x3)?;
    assert_tensor(&y, &tensor![[-5.0, 0.0, 21.0], [64.0, 135.0, 240.0]]);
    let grads = y.backward()?;
    let gx1 = grads.get(&x1).unwrap();
    assert_tensor(gx1, &tensor![[5.0, 12.0, 21.0], [32.0, 45.0, 60.0]]);
    let gx2 = grads.get(&x2).unwrap();
    assert_tensor(gx2, &tensor![[-1.0, 0.0, 3.0], [8.0, 15.0, 24.0]]);
    Ok(())
}

#[test]
fn test_div() -> Result<()> {
    let x1 = tensor![[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]];
    let x2 = tensor![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]];
    let y = (x1 / x2)?;
    assert_tensor(
        &y,
        &tensor![[-0.2, 0.0, 0.14285715], [0.25, 0.33333334, 0.4]],
    );
    Ok(())
}

#[test]
fn test_div_backward() -> Result<()> {
    let x1 = tensor![[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]].requires_grad();
    let x2 = tensor![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]].requires_grad();
    let x3 = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].requires_grad();
    let y = ((&x1 / &x2) * x3)?;
    assert_tensor(&y, &tensor![[-0.2, 0.0, 0.42857146], [1.0, 1.6666667, 2.4]]);
    let grads = y.backward()?;
    let gx1 = grads.get(&x1).unwrap();
    assert_tensor(
        gx1,
        &tensor![[0.2, 0.33333334, 0.42857143], [0.5, 0.5555556, 0.6]],
    );
    let gx2 = grads.get(&x2).unwrap();
    assert_tensor(
        gx2,
        &tensor![[0.04, -0.0, -0.06122449], [-0.125, -0.1851852, -0.24]],
    );
    Ok(())
}

#[test]
fn test_neg() -> Result<()> {
    let x = tensor![[0.0, -2.0, -4.0], [1.0, 2.0, 3.0]];
    let y = (-x)?;
    assert_tensor(&y, &tensor![[0.0, 2.0, 4.0], [-1.0, -2.0, -3.0]]);
    Ok(())
}

#[test]
fn test_neg_backward() -> Result<()> {
    let x = tensor![[0.0, -2.0, -4.0], [1.0, 2.0, 3.0]].requires_grad();
    let y = (-&x)?;
    assert_tensor(&y, &tensor![[0.0, 2.0, 4.0], [-1.0, -2.0, -3.0]]);
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap();
    assert_tensor(&gx, &tensor![[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]);
    Ok(())
}

#[test]
fn test_pow_scalar() -> Result<()> {
    let x = tensor![[0.0, -2.0, -4.0], [1.0, 2.0, 3.0]];
    let y = x.pow_scalar(3.0)?;
    assert_tensor(&y, &tensor![[0.0, -8.0, -64.0], [1.0, 8.0, 27.0]]);
    Ok(())
}

#[test]
fn test_pow_scalar_backward() -> Result<()> {
    let x = tensor![[0.0, -2.0, -4.0], [1.0, 2.0, 3.0]].requires_grad();
    let y = x.pow_scalar(3.0)?;
    assert_tensor(&y, &tensor![[0.0, -8.0, -64.0], [1.0, 8.0, 27.0]]);
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap();
    assert_tensor(&gx, &tensor![[0.0, 12.0, 48.0], [3.0, 12.0, 27.0]]);
    Ok(())
}

#[test]
fn test_reshape() -> Result<()> {
    let x = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let y = x.reshape(vec![6])?;
    assert_tensor(&y, &tensor![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    Ok(())
}

#[test]
fn test_reshape2() -> Result<()> {
    let x = tensor![1.0, 2.0, 3.0];
    let x = x.broadcast_to(vec![2, 3])?;
    let y = x.reshape(vec![6])?;
    assert_tensor(&y, &tensor![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    Ok(())
}

#[test]
fn test_reshape_backward() -> Result<()> {
    let x1 = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].requires_grad();
    let x2 = tensor![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y = (x1.reshape(vec![6])? * x2)?;
    assert_tensor(&y, &tensor![1.0, 4.0, 9.0, 16.0, 25.0, 36.0]);
    let grads = y.backward()?;
    let gx = grads.get(&x1).unwrap();
    assert_tensor(gx, &tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    Ok(())
}

#[test]
fn test_reshape_backward2() -> Result<()> {
    let x1 = tensor![1.0, 2.0, 3.0].requires_grad();
    let x2 = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let y = (x1.broadcast_to(vec![2, 3])? * x2)?;
    assert_tensor(&y, &tensor![[1.0, 4.0, 9.0], [4.0, 10.0, 18.0]]);
    let grads = y.backward()?;
    let gx1 = grads.get(&x1).unwrap();
    assert_tensor(gx1, &tensor![5.0, 7.0, 9.0]);
    Ok(())
}

#[test]
fn test_permuted_axes() -> Result<()> {
    let x = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let y = x.permuted_axes(&[1, 0])?;
    assert_tensor(&y, &tensor![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
    Ok(())
}

#[test]
fn test_permuted_axes_backward() -> Result<()> {
    let x = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].requires_grad();
    let y = x.permuted_axes(&[1, 0])?;
    assert_tensor(&y, &tensor![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap();
    assert_tensor(gx, &tensor![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);
    Ok(())
}

#[test]
fn test_reversed_axes() -> Result<()> {
    let x = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let y = x.reversed_axes()?;
    assert_tensor(&y, &tensor![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
    Ok(())
}

#[test]
fn test_broadcast_to() -> Result<()> {
    let x = tensor![1.0, 2.0, 3.0];
    let y = x.broadcast_to(vec![2, 3])?;
    assert_tensor(&y, &tensor![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
    Ok(())
}

#[test]
fn test_broadcast_to2() -> Result<()> {
    let x = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let y = x.broadcast_to(vec![2, 3])?;
    assert_tensor(&y, &tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    Ok(())
}

#[test]
fn test_broadcast_to3() -> Result<()> {
    let x = tensor![1.0, 2.0, 3.0];
    let x = x.broadcast_to(vec![2, 3])?;
    let y = x.broadcast_to(vec![2, 3])?;
    assert_tensor(&y, &tensor![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
    Ok(())
}

#[test]
fn test_broadcast_to4() -> Result<()> {
    let x = tensor![[1.0], [2.0], [3.0]];
    let y = x.broadcast_to(vec![3, 2])?;
    assert_tensor(&y, &tensor![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]);
    Ok(())
}

#[test]
fn test_broadcast_to5() -> Result<()> {
    let x = tensor![[1.0], [2.0], [3.0]];
    let y = x.broadcast_to(vec![2, 3, 4])?;
    assert_tensor(
        &y,
        &tensor![
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0]
            ],
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0]
            ]
        ],
    );
    Ok(())
}

#[test]
fn test_broadcast_to_backward() -> Result<()> {
    let x1 = tensor![1.0, 2.0, 3.0].requires_grad();
    let x2 = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let y = (x1.broadcast_to(vec![2, 3])? * x2)?;
    assert_tensor(&y, &tensor![[1.0, 4.0, 9.0], [4.0, 10.0, 18.0]]);
    let grads = y.backward()?;
    let gx = grads.get(&x1).unwrap();
    assert_tensor(gx, &tensor![5.0, 7.0, 9.0]);
    Ok(())
}

#[test]
fn test_sum_to() -> Result<()> {
    let x = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let y = x.sum_to(&vec![1, 3])?;
    assert_tensor(&y, &tensor![[5.0, 7.0, 9.0]]);
    Ok(())
}

#[test]
fn test_sum_to2() -> Result<()> {
    let x = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let y = x.sum_to(&vec![3])?;
    assert_tensor(&y, &tensor![5.0, 7.0, 9.0]);
    Ok(())
}

#[test]
fn test_sum_to3() -> Result<()> {
    let x = tensor![[1.0, 2.0, 3.0]];
    let x = x.broadcast_to(vec![4, 3])?;
    let y = x.sum_to(&vec![1, 3])?;
    assert_tensor(&y, &tensor![[4.0, 8.0, 12.0]]);
    Ok(())
}

#[test]
fn test_sum_to4() -> Result<()> {
    let x = tensor![[1.0, 2.0, 3.0]];
    let x = x.broadcast_to(vec![4, 3])?;
    let y = x.sum_to(&vec![4, 1])?;
    assert_tensor(&y, &tensor![[6.0], [6.0], [6.0], [6.0]]);
    Ok(())
}

#[test]
fn test_sum_to5() -> Result<()> {
    let x = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let y = x.sum_to(&vec![1])?;
    assert_tensor(&y, &tensor![21.0]);
    Ok(())
}

#[test]
fn test_sum_to6() -> Result<()> {
    let x = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let y = x.sum_to(&vec![2, 1])?;
    assert_tensor(&y, &tensor![[6.0], [15.0]]);
    Ok(())
}

#[test]
fn test_sum_to_backward() -> Result<()> {
    let x = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].requires_grad();
    let y = x.sum_to(&vec![3])?;
    assert_tensor(&y, &tensor![5.0, 7.0, 9.0]);
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap();
    assert_tensor(gx, &tensor![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);
    Ok(())
}

#[test]
fn test_sum_axis() -> Result<()> {
    let x = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let y = x.sum_axis(0, false)?;
    assert_tensor(&y, &tensor![5.0, 7.0, 9.0]);
    Ok(())
}

#[test]
fn test_sum_axis_keepdims_true() -> Result<()> {
    let x = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let y = x.sum_axis(0, true)?;
    assert_tensor(&y, &tensor![[5.0, 7.0, 9.0]]);
    Ok(())
}

#[test]
fn test_sum_axis_backward() -> Result<()> {
    let x = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].requires_grad();
    let y = x.sum_axis(0, false)?;
    assert_tensor(&y, &tensor![5.0, 7.0, 9.0]);
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap();
    assert_tensor(gx, &tensor![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);
    Ok(())
}

#[test]
fn test_sum_axis_backward_keepdims_true() -> Result<()> {
    let x = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].requires_grad();
    let y = x.sum_axis(0, true)?;
    assert_tensor(&y, &tensor![[5.0, 7.0, 9.0]]);
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap();
    assert_tensor(gx, &tensor![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);
    Ok(())
}

#[test]
fn test_contiguous() -> Result<()> {
    let x = tensor![1.0, 2.0, 3.0];
    let x = x.broadcast_to(vec![2, 3])?;
    assert!(!x.is_contiguous());
    let y = x.contiguous()?;
    assert!(y.is_contiguous());
    assert_tensor(&y, &tensor![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
    Ok(())
}

#[test]
fn test_contiguous_backward() -> Result<()> {
    let x = tensor![1.0, 2.0, 3.0].requires_grad();
    let h = x.broadcast_to(vec![2, 3])?;
    assert!(!h.is_contiguous());
    let y = h.contiguous()?;
    assert!(y.is_contiguous());
    assert_tensor(&y, &tensor![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap();
    assert_tensor(gx, &tensor![2.0, 2.0, 2.0]);
    Ok(())
}
