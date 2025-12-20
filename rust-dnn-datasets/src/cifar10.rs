use std::fs::{self};

use rust_dnn_core::{
    cpu_backend::CpuBackend,
    device::Device,
    error::{Error, Result},
    tensor::Tensor,
};

pub struct CIFAR10Loader {
    dataset_path: String,
}

impl CIFAR10Loader {
    pub fn new(dataset_path: &str) -> Self {
        Self {
            dataset_path: dataset_path.to_string(),
        }
    }

    pub fn load_train(&self) -> Result<(Tensor<CpuBackend, u32>, Tensor<CpuBackend, u32>)> {
        let data_batch_paths = vec![
            "cifar-10-binary/cifar-10-batches-bin/data_batch_1.bin",
            "cifar-10-binary/cifar-10-batches-bin/data_batch_2.bin",
            "cifar-10-binary/cifar-10-batches-bin/data_batch_3.bin",
            "cifar-10-binary/cifar-10-batches-bin/data_batch_4.bin",
            "cifar-10-binary/cifar-10-batches-bin/data_batch_5.bin",
        ];

        let mut train_images_batches = Vec::new();
        let mut train_labels_batches = Vec::new();
        for data_batch_path in data_batch_paths {
            let (train_images_batch, train_labels_batch) = self.load_batch(data_batch_path)?;
            train_images_batches.push(train_images_batch);
            train_labels_batches.push(train_labels_batch);
        }
        let train_images = Tensor::cat(&train_images_batches, 0);
        let train_images = train_images.reshape(vec![50000, 3, 32, 32]);
        let train_labels = Tensor::cat(&train_labels_batches, 0);
        let train_labels = train_labels.reshape(vec![50000]);

        Ok((train_images, train_labels))
    }

    pub fn load_test(&self) -> Result<(Tensor<CpuBackend, u32>, Tensor<CpuBackend, u32>)> {
        let (test_images, test_labels) =
            self.load_batch("cifar-10-binary/cifar-10-batches-bin/test_batch.bin")?;
        let test_images = test_images.reshape(vec![10000, 3, 32, 32]);
        let test_labels = test_labels.reshape(vec![10000]);
        Ok((test_images.contiguous(), test_labels.contiguous()))
    }

    fn load_batch(
        &self,
        file_path: &str,
    ) -> Result<(Tensor<CpuBackend, u32>, Tensor<CpuBackend, u32>)> {
        let file_path = format!("{}/{}", self.dataset_path, file_path);
        let binary = match fs::read(&file_path) {
            Ok(binary) => binary,
            Err(_) => {
                return Err(Error::LoadDatasetError {
                    msg: format!("File load failed. (file = {})", &file_path).to_string(),
                });
            }
        };
        let mut data = Vec::new();
        for i in 0..binary.len() {
            data.push(binary[i] as u32);
        }
        let x = Tensor::from_vec(data, vec![10000, 3073], Device::get_cpu_device());
        let pixels = x.get_item(vec![(0, 10000), (1, 3073)]);
        let labels = x.get_item(vec![(0, 10000), (0, 1)]);
        Ok((pixels, labels))
    }
}
