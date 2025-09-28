use rust_dnn_core::tensor::Tensor;

pub fn assert_tensor(actual: &Tensor, expected: &Tensor) {
    assert_tensor_with_eps(actual, expected, 1e-4)
}

pub fn assert_tensor_with_eps(actual: &Tensor, expected: &Tensor, eps: f32) {
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(actual.len(), expected.len());
    let actual_data = actual.to_vec();
    let expected_data = expected.to_vec();
    for i in 0..expected_data.len() {
        let diff = (actual_data[i] - expected_data[i]).abs();
        // NOTE: NaN対策にdiff >= epsではなく!(diff < eps)で比較している。
        if !(diff < eps) {
            println!("actual_data = {:?}", actual_data);
            println!("expected_data = {:?}", expected_data);
            panic!(
                "actual_data[{}] - expected_data[{}]).abs() = {}",
                i, i, diff
            );
        }
    }
}
