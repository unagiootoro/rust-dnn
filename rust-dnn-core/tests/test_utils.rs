use rust_dnn_core::{backend::Backend, dim::Dim, num::Num, tensor::Tensor};

#[macro_export]
macro_rules! define_test {
    ($fn_name: ident, $test_cpu: ident, $test_cuda: ident) => {
        #[test]
        fn $test_cpu() -> Result<()> {
            $fn_name(Device::get_cpu_device())
        }

        #[cfg(feature = "cuda")]
        #[test]
        fn $test_cuda() -> Result<()> {
            $fn_name(Device::get_cuda_device())
        }
    };
}

pub fn assert_tensor<B1: Backend, B2: Backend, D: Dim, T: Num>(
    actual: &Tensor<B1, D, T>,
    expected: &Tensor<B2, D, T>,
) {
    assert_tensor_with_eps::<B1, B2, D, T>(actual, expected, 1e-4)
}

pub fn assert_tensor_with_eps<B1: Backend, B2: Backend, D: Dim, T: Num>(
    actual: &Tensor<B1, D, T>,
    expected: &Tensor<B2, D, T>,
    eps: f32,
) {
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(actual.len(), expected.len());
    let actual_data = actual.to_vec().unwrap();
    let expected_data = expected.to_vec().unwrap();
    for i in 0..expected_data.len() {
        let diff = ((actual_data[i] - expected_data[i]).as_f32()).abs();
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
