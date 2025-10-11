use std::{cell::RefCell, rc::Rc};

use crate::{gpu_memory_block::GpuMemoryBlock, raw_gpu_buffer::RawGpuBuffer};

pub struct GpuMemoryAllocator {
    memory_pool: Vec<Rc<RefCell<GpuMemoryBlock>>>,
    pub debug_disable_allocate_flag: bool, // TODO: デバッグ用
}

impl GpuMemoryAllocator {
    pub fn new() -> Self {
        Self {
            memory_pool: Vec::new(),
            debug_disable_allocate_flag: false,
        }
    }

    pub unsafe fn allocate(&mut self, size: usize) -> Rc<RefCell<GpuMemoryBlock>> {
        for block in &self.memory_pool {
            if !block.borrow().locked && block.borrow().buffer.size() == size {
                block.borrow_mut().locked = true;
                return Rc::clone(block);
            }
        }

        if self.debug_disable_allocate_flag {
            panic!("failed allocate({}byte)", size);
        }

        let mut block = GpuMemoryBlock::new(unsafe { RawGpuBuffer::new(size) });
        block.locked = true;
        let block = Rc::new(RefCell::new(block));
        self.memory_pool.push(Rc::clone(&block));
        block
    }

    pub fn free(&mut self, memory_block: &Rc<RefCell<GpuMemoryBlock>>) {
        memory_block.borrow_mut().unlock();
    }
}
