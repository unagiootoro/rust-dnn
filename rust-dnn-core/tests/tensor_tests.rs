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
    let x1 = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let x2 = tensor![[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
    let y = (x1 + x2)?;
    assert_tensor(&y, &tensor![[8.0, 10.0, 12.0], [14.0, 16.0, 18.0]]);
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
fn test_permuted_axes() -> Result<()> {
    let x = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let y = x.permuted_axes(&[1, 0])?;
    assert_tensor(&y, &tensor![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
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
fn test_is_contiguous() -> Result<()> {
    let x = tensor![1.0, 2.0, 3.0];
    let x = x.broadcast_to(vec![2, 3])?;
    assert!(!x.is_contiguous());
    let y = x.contiguous()?;
    assert!(y.is_contiguous());
    Ok(())
}

#[test]
fn test_contiguous() -> Result<()> {
    let x = tensor![1.0, 2.0, 3.0];
    let y = x.broadcast_to(vec![2, 3])?;
    assert_eq!(y.storage_offset(), 0);
    Ok(())
}
