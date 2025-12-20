mod test_utils;

use std::collections::HashMap;

use rust_dnn_core::{backend::Backend, device::Device, error::Result, tensor::Tensor};
use rust_dnn_nn::{
    layer::{Layer, Linear},
    sequential::{Sequential, SequentialItem},
};

struct TestModel<B: Backend> {
    list: Sequential<Linear<B, f64>, B, f64>,
}

impl<B: Backend> TestModel<B> {
    pub fn new(device: Device<B>) -> Self {
        let fc0 = Linear::new(2, 3, true, device);
        let fc1 = Linear::new(3, 4, true, device);
        let list = Sequential::from_vec(vec![fc0, fc1]);
        Self { list }
    }

    pub fn forward(&mut self, x: Tensor<B, f64>) -> Tensor<B, f64> {
        self.list.forward(x, true)
    }
}

impl<B: Backend> Layer<B, f64> for TestModel<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f64>> {
        let mut map: HashMap<String, &dyn Layer<B, f64>> = HashMap::new();
        map.insert("list".to_string(), &self.list);
        map
    }
}

fn test_sequential_forward<B: Backend>(device: Device<B>) -> Result<()> {
    let linear = Linear::new(2, 3, true, device);
    let linear2 = Linear::new(3, 4, true, device);

    let x = Tensor::<B, f64>::rand_norm(&[1, 2], Some(838861), device);
    let h = linear.forward(&x);
    let y = linear2.forward(&h);

    let mut list = Sequential::from_vec(vec![linear, linear2]);
    let y2 = list.forward(x, true);

    assert_eq!(y.shape(), y2.shape());
    assert_eq!(y.to_vec(), y2.to_vec());
    Ok(())
}

define_test!(
    test_sequential_forward,
    test_sequential_forward_cpu,
    test_sequential_forward_cuda
);

fn test_all_parameters_map<B: Backend>(device: Device<B>) -> Result<()> {
    let mut model = TestModel::<B>::new(device);
    let params = model.all_parameters_map();
    let mut model2 = TestModel::<B>::new(device);
    model2.load_parameters_map(params)?;

    let x = Tensor::<B, f64>::rand_norm(&[1, 2], Some(838861), device);
    let y = model.forward(x.clone());
    let y2 = model2.forward(x.clone());
    assert_eq!(y.shape(), y2.shape());
    assert_eq!(y.to_vec(), y2.to_vec());
    Ok(())
}

define_test!(
    test_all_parameters_map,
    test_all_parameters_map_cpu,
    test_all_parameters_map_cuda
);
