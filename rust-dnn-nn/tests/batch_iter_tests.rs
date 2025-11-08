mod test_utils;

use rust_dnn_core::{device::Device, ten};
use rust_dnn_nn::batch_iter::batch_iter;

use crate::test_utils::assert_tensor;

#[test]
fn test_next() {
    let data = ten![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ];
    let datas = (data,);
    let mut batch_iter = batch_iter(&datas, 2, true, Some([0; 32]));
    let (batch,) = batch_iter.next().unwrap();
    assert_tensor(&batch, &ten![[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]);

    let (batch,) = batch_iter.next().unwrap();
    assert_tensor(&batch, &ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    assert!(batch_iter.next().is_none());
}

#[test]
fn test_next_shuffle_false() {
    let data = ten![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ];
    let datas = (data,);
    let mut batch_iter = batch_iter(&datas, 2, false, None);
    let (batch,) = batch_iter.next().unwrap();
    assert_tensor(&batch, &ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let (batch,) = batch_iter.next().unwrap();
    assert_tensor(&batch, &ten![[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]);

    assert!(batch_iter.next().is_none());
}
