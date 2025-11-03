use std::{fs::File, io::Read};

use flate2::read::GzDecoder;
use rust_dnn_core::{
    cpu_backend::CpuBackend,
    device::Device,
    error::{Error, Result},
    tensor::Tensor,
};

pub struct MNISTLoader {
    dataset_path: String,
}

impl MNISTLoader {
    pub fn new(dataset_path: &str) -> Self {
        Self {
            dataset_path: dataset_path.to_string(),
        }
    }

    pub fn load_train(&self) -> Result<(Tensor<CpuBackend, u32>, Tensor<CpuBackend, u32>)> {
        let train_images = {
            let buf = self.load_buffer_from_file("train-images-idx3-ubyte.gz")?;
            self.parse_train_images(buf)?
        };

        let train_labels = {
            let buf = self.load_buffer_from_file("train-labels-idx1-ubyte.gz")?;
            self.parse_train_labels(buf)?
        };

        Ok((train_images, train_labels))
    }

    pub fn load_test(&self) -> Result<(Tensor<CpuBackend, u32>, Tensor<CpuBackend, u32>)> {
        let test_images = {
            let buf = self.load_buffer_from_file("t10k-images-idx3-ubyte.gz")?;
            self.parse_test_images(buf)?
        };

        let test_labels = {
            let buf = self.load_buffer_from_file("t10k-labels-idx1-ubyte.gz")?;
            self.parse_test_labels(buf)?
        };

        Ok((test_images, test_labels))
    }

    pub fn load_buffer_from_file(&self, file_name: &str) -> Result<Vec<u8>> {
        let file_path = format!("{}/{}", self.dataset_path, file_name);
        let file = match File::open(&file_path) {
            Ok(file) => file,
            Err(_) => {
                return Err(Error::LoadDatasetError {
                    msg: format!("File load failed. (file = {})", &file_path).to_string(),
                });
            }
        };
        let mut decoder = GzDecoder::new(file);
        let mut buf = Vec::new();
        match decoder.read_to_end(&mut buf) {
            Ok(_) => (),
            Err(_) => {
                return Err(Error::LoadDatasetError {
                    msg: format!("decode Gz failed. (file = {})", &file_path).to_string(),
                });
            }
        };
        Ok(buf)
    }

    fn parse_train_images(&self, buf: Vec<u8>) -> Result<Tensor<CpuBackend, u32>> {
        let mut data = Vec::new();
        for i in 16..buf.len() {
            data.push(buf[i] as u32);
        }
        Ok(Tensor::from_vec(data, vec![60000, 28, 28], Device::get_cpu_device()).unwrap())
    }

    fn parse_train_labels(&self, buf: Vec<u8>) -> Result<Tensor<CpuBackend, u32>> {
        let mut data = Vec::new();
        for i in 8..buf.len() {
            data.push(buf[i] as u32);
        }
        Ok(Tensor::from_vec(data, vec![60000], Device::get_cpu_device()).unwrap())
    }

    fn parse_test_images(&self, buf: Vec<u8>) -> Result<Tensor<CpuBackend, u32>> {
        let mut data = Vec::new();
        for i in 16..buf.len() {
            data.push(buf[i] as u32);
        }
        Ok(Tensor::from_vec(data, vec![10000, 28, 28], Device::get_cpu_device()).unwrap())
    }

    fn parse_test_labels(&self, buf: Vec<u8>) -> Result<Tensor<CpuBackend, u32>> {
        let mut data = Vec::new();
        for i in 8..buf.len() {
            data.push(buf[i] as u32);
        }
        Ok(Tensor::from_vec(data, vec![10000], Device::get_cpu_device()).unwrap())
    }
}
