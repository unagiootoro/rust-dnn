mod test_utils;
use rust_dnn_core::{error::Result, tensor};

use crate::test_utils::assert_tensor;

#[test]
fn test_backprop() -> Result<()> {
    let x = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].requires_grad();
    let y = x.reshape(vec![6])?;
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap();
    assert_tensor(gx, &tensor![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    Ok(())
}
