mod test_utils;
use rust_dnn_core::tensor;

use crate::test_utils::assert_tensor;

#[test]
fn test_add() {
    let x0 = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let x1 = tensor![[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
    let y = x0.add(&x1).unwrap();
    assert_tensor(&y, &tensor![[8.0, 10.0, 12.0], [14.0, 16.0, 18.0]]);
}

#[test]
fn test_reshape() {
    let x = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let y = x.reshape(vec![6]).unwrap();
    assert_tensor(&y, &tensor![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_reshape2() {
    let x = tensor![1.0, 2.0, 3.0];
    let x = x.broadcast_to(vec![2, 3]).unwrap();
    let y = x.reshape(vec![6]).unwrap();
    assert_tensor(&y, &tensor![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
}

#[test]
fn test_permuted_axes() {
    let x = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let y = x.permuted_axes(&[1, 0]).unwrap();
    assert_tensor(&y, &tensor![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
}

#[test]
fn test_reversed_axes() {
    let x = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let y = x.reversed_axes().unwrap();
    assert_tensor(&y, &tensor![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
}

#[test]
fn test_broadcast_to() {
    let x = tensor![1.0, 2.0, 3.0];
    let y = x.broadcast_to(vec![2, 3]).unwrap();
    assert_tensor(&y, &tensor![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
}

#[test]
fn test_broadcast_to2() {
    let x = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let y = x.broadcast_to(vec![2, 3]).unwrap();
    assert_tensor(&y, &tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
}

#[test]
fn test_broadcast_to3() {
    let x = tensor![1.0, 2.0, 3.0];
    let x = x.broadcast_to(vec![2, 3]).unwrap();
    let y = x.broadcast_to(vec![2, 3]).unwrap();
    assert_tensor(&y, &tensor![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
}

#[test]
fn test_broadcast_to4() {
    let x = tensor![[1.0], [2.0], [3.0]];
    let y = x.broadcast_to(vec![3, 2]).unwrap();
    assert_tensor(&y, &tensor![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]);
}

#[test]
fn test_broadcast_to5() {
    let x = tensor![[1.0], [2.0], [3.0]];
    let y = x.broadcast_to(vec![2, 3, 4]).unwrap();
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
}

#[test]
fn test_is_contiguous() {
    let x = tensor![1.0, 2.0, 3.0];
    let x = x.broadcast_to(vec![2, 3]).unwrap();
    assert!(!x.is_contiguous());
    let y = x.contiguous().unwrap();
    assert!(y.is_contiguous());
}

#[test]
fn test_contiguous() {
    let x = tensor![1.0, 2.0, 3.0];
    let y = x.broadcast_to(vec![2, 3]).unwrap();
    assert_eq!(y.storage_offset(), 0);
}
