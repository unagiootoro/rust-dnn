pub mod layout;
pub mod shader_type;
pub mod wgpu_buffer;
pub mod wgpu_state;

use bytemuck::{AnyBitPattern, NoUninit};
use std::cell::RefCell;
use std::{borrow::Cow, rc::Rc};
use wgpu::{Label, util::DeviceExt};

use crate::layout::Layout;
use crate::shader_type::ShaderType;
use crate::wgpu_buffer::WgpuBuffer;
use crate::wgpu_state::WGPUState; // バッファ初期化のためのユーティリティ

thread_local!(static WGPU_STATE: Rc<RefCell<Option<WGPUState>>> = Rc::new(RefCell::new(None)));

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Length {
    len: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
}

impl Length {
    pub fn new(len: u32) -> Self {
        Self {
            len,
            _padding1: 0,
            _padding2: 0,
            _padding3: 0,
        }
    }
}

pub fn init_wgpu_state() {
    async fn create_wgpu_state() -> WGPUState {
        WGPUState::new().await
    }

    let s = WGPU_STATE.with(|s| Rc::clone(s));
    if s.borrow().is_some() {
        return;
    }

    let s = pollster::block_on(create_wgpu_state());
    WGPU_STATE.with(|state| {
        *state.borrow_mut() = Some(s);
    });
}

pub fn dispose_wgpu_state() {
    WGPU_STATE.with(|state| {
        *state.borrow_mut() = None;
    });
}

pub fn create_wgpu_buffer_from_data<T: NoUninit>(data: Vec<T>) -> wgpu::Buffer {
    init_wgpu_state();
    let state = WGPU_STATE.with(|s| Rc::clone(s));
    let buffer = state
        .borrow_mut()
        .as_mut()
        .unwrap()
        .create_buffer(Some("Storage Buffer"), &data);
    buffer
}

pub fn buffer_to_vec<T: AnyBitPattern>(buffer: &wgpu::Buffer, len: usize) -> Vec<T> {
    init_wgpu_state();
    pollster::block_on(buffer_to_vec_async(buffer, len))
}

async fn buffer_to_vec_async<T: AnyBitPattern>(buffer: &wgpu::Buffer, len: usize) -> Vec<T> {
    let state = WGPU_STATE.with(|s| Rc::clone(s));
    let result = state
        .borrow_mut()
        .as_mut()
        .unwrap()
        .buffer_to_vec(buffer, len)
        .await;
    result
}

pub fn wgpu_exp(input: &WgpuBuffer, output: &WgpuBuffer, len: u32, shader_type: ShaderType) {
    wgpu_op1(input, output, len, shader_type, "array_exp");
}

fn wgpu_op1(
    input: &WgpuBuffer,
    output: &WgpuBuffer,
    len: u32,
    shader_type: ShaderType,
    entry_point: &str,
) {
    pollster::block_on(wgpu_op1_async(input, output, len, shader_type, entry_point))
}

async fn wgpu_op1_async(
    input: &WgpuBuffer,
    output: &WgpuBuffer,
    len: u32,
    shader_type: ShaderType,
    entry_point: &str,
) {
    let state = WGPU_STATE.with(|s| Rc::clone(s));
    state
        .borrow_mut()
        .as_mut()
        .unwrap()
        .op1(input, output, len, shader_type, entry_point)
        .await;
}

pub fn wgpu_add(
    lhs: &WgpuBuffer,
    lhs_layout: Layout,
    rhs: &WgpuBuffer,
    rhs_layout: Layout,
    output: &WgpuBuffer,
    len: u32,
    shader_type: ShaderType,
) {
    wgpu_op2(
        lhs,
        lhs_layout,
        rhs,
        rhs_layout,
        output,
        len,
        shader_type,
        "add",
    );
}

pub fn wgpu_sub(
    lhs: &WgpuBuffer,
    lhs_layout: Layout,
    rhs: &WgpuBuffer,
    rhs_layout: Layout,
    output: &WgpuBuffer,
    len: u32,
    shader_type: ShaderType,
) {
    wgpu_op2(
        lhs,
        lhs_layout,
        rhs,
        rhs_layout,
        output,
        len,
        shader_type,
        "sub",
    );
}

pub fn wgpu_mul(
    lhs: &WgpuBuffer,
    lhs_layout: Layout,
    rhs: &WgpuBuffer,
    rhs_layout: Layout,
    output: &WgpuBuffer,
    len: u32,
    shader_type: ShaderType,
) {
    wgpu_op2(
        lhs,
        lhs_layout,
        rhs,
        rhs_layout,
        output,
        len,
        shader_type,
        "mul",
    );
}

fn wgpu_op2(
    lhs: &WgpuBuffer,
    lhs_layout: Layout,
    rhs: &WgpuBuffer,
    rhs_layout: Layout,
    output: &WgpuBuffer,
    len: u32,
    shader_type: ShaderType,
    entry_point: &str,
) {
    pollster::block_on(wgpu_op2_async(
        lhs,
        lhs_layout,
        rhs,
        rhs_layout,
        output,
        len,
        shader_type,
        entry_point,
    ))
}

async fn wgpu_op2_async(
    lhs: &WgpuBuffer,
    lhs_layout: Layout,
    rhs: &WgpuBuffer,
    rhs_layout: Layout,
    output: &WgpuBuffer,
    len: u32,
    shader_type: ShaderType,
    entry_point: &str,
) {
    let state = WGPU_STATE.with(|s| Rc::clone(s));
    state
        .borrow_mut()
        .as_mut()
        .unwrap()
        .op2(
            lhs,
            lhs_layout,
            rhs,
            rhs_layout,
            &output,
            len,
            shader_type,
            entry_point,
        )
        .await;
}
