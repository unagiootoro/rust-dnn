use std::collections::VecDeque;

use rand::prelude::*;
use rand::seq::SliceRandom;

use std::array;

use rust_dnn_core::{backend::Backend, device::Device, num::Num, ten, tensor::Tensor};

pub fn batch_iter<'a, T: Batchable>(
    tensors: &'a T,
    batch_size: usize,
    shuffle: bool,
    seed: Option<[u8; 32]>,
) -> BatchIter<'a, T> {
    BatchIter::new(tensors, batch_size, shuffle, seed)
}

pub trait Batchable {
    fn get_batch(&self, index: &[u32]) -> Self;
    fn batch_size(&self) -> usize;
}

impl<B: Backend, T: Num> Batchable for (Tensor<B, T>,) {
    fn get_batch(&self, index: &[u32]) -> Self {
        let index0 = Tensor::from_vec(index.to_vec(), vec![index.len()], self.0.device()).unwrap();
        (self.0.index_select(0, &index0).unwrap(),)
    }

    fn batch_size(&self) -> usize {
        self.0.shape()[0]
    }
}

impl<B1: Backend, B2: Backend, T1: Num, T2: Num> Batchable for (Tensor<B1, T1>, Tensor<B2, T2>) {
    fn get_batch(&self, index: &[u32]) -> Self {
        let index0 = Tensor::from_vec(index.to_vec(), vec![index.len()], self.0.device()).unwrap();
        let index1 = Tensor::from_vec(index.to_vec(), vec![index.len()], self.1.device()).unwrap();
        (
            self.0.index_select(0, &index0).unwrap(),
            self.1.index_select(0, &index1).unwrap(),
        )
    }

    fn batch_size(&self) -> usize {
        self.0.shape()[0]
    }
}

impl<B: Backend, T: Num, const N: usize> Batchable for [Tensor<B, T>; N] {
    fn get_batch(&self, index: &[u32]) -> Self {
        let index = Tensor::from_vec(index.to_vec(), vec![index.len()], self[0].device()).unwrap();
        array::from_fn(|i| self[i].index_select(0, &index).unwrap())
    }

    fn batch_size(&self) -> usize {
        self[0].shape()[0]
    }
}

pub struct BatchIter<'a, T: Batchable> {
    tensors: &'a T,
    batch_size: usize,
    indices: VecDeque<u32>,
}

impl<'a, T: Batchable> BatchIter<'a, T> {
    pub fn new(tensors: &'a T, batch_size: usize, shuffle: bool, seed: Option<[u8; 32]>) -> Self {
        let num_records = tensors.batch_size();
        let mut indices = (0..num_records).map(|i| i as u32).collect::<Vec<u32>>();
        if shuffle {
            let seed = if let Some(seed) = seed { seed } else { [0; 32] };
            let mut rng = StdRng::from_seed(seed);
            indices.shuffle(&mut rng);
        }
        let indices = VecDeque::from(indices);

        Self {
            tensors,
            batch_size,
            indices,
        }
    }
}

impl<'a, T: Batchable> Iterator for BatchIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.indices.len() > 0 {
            let batch_size = self.batch_size.min(self.indices.len());
            let indices = (0..batch_size)
                .map(|_| self.indices.pop_front().unwrap())
                .collect::<Vec<u32>>();
            let batches = self.tensors.get_batch(&indices);
            Some(batches)
        } else {
            None
        }
    }
}
