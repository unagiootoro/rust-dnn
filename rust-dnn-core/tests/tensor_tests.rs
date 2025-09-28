mod test_utils;
use rust_dnn_core::tensor;

use crate::test_utils::assert_tensor;

#[test]
fn test_add() {
    let a = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let b = tensor![[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
    let output = a.add(&b).unwrap();
    assert_tensor(&output, &tensor![[8.0, 10.0, 12.0], [14.0, 16.0, 18.0]]);
}
