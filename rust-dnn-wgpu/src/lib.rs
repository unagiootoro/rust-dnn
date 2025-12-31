pub mod layout;
pub mod wgpu_buffer;
pub mod wgpu_dtype;
pub mod wgpu_state;

use bytemuck::{AnyBitPattern, NoUninit};
use std::cell::RefCell;
use std::{borrow::Cow, rc::Rc};
use wgpu::{Label, util::DeviceExt};

use crate::layout::Layout;
use crate::wgpu_buffer::WgpuBuffer;
use crate::wgpu_dtype::WgpuDTypeKind;
use crate::wgpu_state::WGPUState; // バッファ初期化のためのユーティリティ

thread_local!(static WGPU_STATE: Rc<RefCell<Option<WGPUState>>> = Rc::new(RefCell::new(None)));

pub enum Op2ShaderKind {
    Num,
    Float,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Op1Params {
    storage_offset: u32,
    len: u32,
    _padding1: u32,
    _padding2: u32,
}

impl Op1Params {
    pub fn new(storage_offset: u32, len: u32) -> Self {
        Self {
            storage_offset,
            len,
            _padding1: 0,
            _padding2: 0,
        }
    }
}

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

pub fn wgpu_neg(input: &WgpuBuffer, storage_offset: u32, output: &WgpuBuffer, len: u32) {
    wgpu_op1(input, storage_offset, output, len, "neg");
}

pub fn wgpu_exp(input: &WgpuBuffer, storage_offset: u32, output: &WgpuBuffer, len: u32) {
    wgpu_op1(input, storage_offset, output, len, "array_exp");
}

pub fn wgpu_sqrt(input: &WgpuBuffer, storage_offset: u32, output: &WgpuBuffer, len: u32) {
    wgpu_op1(input, storage_offset, output, len, "array_sqrt");
}

pub fn wgpu_log(input: &WgpuBuffer, storage_offset: u32, output: &WgpuBuffer, len: u32) {
    wgpu_op1(input, storage_offset, output, len, "array_log");
}

pub fn wgpu_sin(input: &WgpuBuffer, storage_offset: u32, output: &WgpuBuffer, len: u32) {
    wgpu_op1(input, storage_offset, output, len, "array_sin");
}

pub fn wgpu_cos(input: &WgpuBuffer, storage_offset: u32, output: &WgpuBuffer, len: u32) {
    wgpu_op1(input, storage_offset, output, len, "array_cos");
}

pub fn wgpu_tanh(input: &WgpuBuffer, storage_offset: u32, output: &WgpuBuffer, len: u32) {
    wgpu_op1(input, storage_offset, output, len, "array_tanh");
}

fn wgpu_op1(
    input: &WgpuBuffer,
    storage_offset: u32,
    output: &WgpuBuffer,
    len: u32,
    entry_point: &str,
) {
    pollster::block_on(wgpu_op1_async(
        input,
        storage_offset,
        output,
        len,
        entry_point,
    ))
}

async fn wgpu_op1_async(
    input: &WgpuBuffer,
    storage_offset: u32,
    output: &WgpuBuffer,
    len: u32,
    entry_point: &str,
) {
    let state = WGPU_STATE.with(|s| Rc::clone(s));
    state
        .borrow_mut()
        .as_mut()
        .unwrap()
        .op1(input, storage_offset, output, len, entry_point)
        .await;
}

pub fn wgpu_add(
    lhs: &WgpuBuffer,
    lhs_layout: Layout,
    rhs: &WgpuBuffer,
    rhs_layout: Layout,
    output: &WgpuBuffer,
    len: u32,
) {
    wgpu_op2(
        lhs,
        lhs_layout,
        rhs,
        rhs_layout,
        output,
        len,
        Op2ShaderKind::Num,
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
) {
    wgpu_op2(
        lhs,
        lhs_layout,
        rhs,
        rhs_layout,
        output,
        len,
        Op2ShaderKind::Num,
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
) {
    wgpu_op2(
        lhs,
        lhs_layout,
        rhs,
        rhs_layout,
        output,
        len,
        Op2ShaderKind::Num,
        "mul",
    );
}

pub fn wgpu_div(
    lhs: &WgpuBuffer,
    lhs_layout: Layout,
    rhs: &WgpuBuffer,
    rhs_layout: Layout,
    output: &WgpuBuffer,
    len: u32,
) {
    wgpu_op2(
        lhs,
        lhs_layout,
        rhs,
        rhs_layout,
        output,
        len,
        Op2ShaderKind::Num,
        "div",
    );
}

pub fn wgpu_eq(
    lhs: &WgpuBuffer,
    lhs_layout: Layout,
    rhs: &WgpuBuffer,
    rhs_layout: Layout,
    output: &WgpuBuffer,
    len: u32,
) {
    wgpu_op2(
        lhs,
        lhs_layout,
        rhs,
        rhs_layout,
        output,
        len,
        Op2ShaderKind::Num,
        "eq",
    );
}

pub fn wgpu_lt(
    lhs: &WgpuBuffer,
    lhs_layout: Layout,
    rhs: &WgpuBuffer,
    rhs_layout: Layout,
    output: &WgpuBuffer,
    len: u32,
) {
    wgpu_op2(
        lhs,
        lhs_layout,
        rhs,
        rhs_layout,
        output,
        len,
        Op2ShaderKind::Num,
        "lt",
    );
}

pub fn wgpu_le(
    lhs: &WgpuBuffer,
    lhs_layout: Layout,
    rhs: &WgpuBuffer,
    rhs_layout: Layout,
    output: &WgpuBuffer,
    len: u32,
) {
    wgpu_op2(
        lhs,
        lhs_layout,
        rhs,
        rhs_layout,
        output,
        len,
        Op2ShaderKind::Num,
        "le",
    );
}

pub fn wgpu_gt(
    lhs: &WgpuBuffer,
    lhs_layout: Layout,
    rhs: &WgpuBuffer,
    rhs_layout: Layout,
    output: &WgpuBuffer,
    len: u32,
) {
    wgpu_op2(
        lhs,
        lhs_layout,
        rhs,
        rhs_layout,
        output,
        len,
        Op2ShaderKind::Num,
        "gt",
    );
}

pub fn wgpu_ge(
    lhs: &WgpuBuffer,
    lhs_layout: Layout,
    rhs: &WgpuBuffer,
    rhs_layout: Layout,
    output: &WgpuBuffer,
    len: u32,
) {
    wgpu_op2(
        lhs,
        lhs_layout,
        rhs,
        rhs_layout,
        output,
        len,
        Op2ShaderKind::Num,
        "ge",
    );
}

pub fn wgpu_pow(
    lhs: &WgpuBuffer,
    lhs_layout: Layout,
    rhs: &WgpuBuffer,
    rhs_layout: Layout,
    output: &WgpuBuffer,
    len: u32,
) {
    wgpu_op2(
        lhs,
        lhs_layout,
        rhs,
        rhs_layout,
        output,
        len,
        Op2ShaderKind::Float,
        "array_pow",
    );
}

fn wgpu_op2(
    lhs: &WgpuBuffer,
    lhs_layout: Layout,
    rhs: &WgpuBuffer,
    rhs_layout: Layout,
    output: &WgpuBuffer,
    len: u32,
    op2_shader_kind: Op2ShaderKind,
    entry_point: &str,
) {
    pollster::block_on(wgpu_op2_async(
        lhs,
        lhs_layout,
        rhs,
        rhs_layout,
        output,
        len,
        op2_shader_kind,
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
    op2_shader_kind: Op2ShaderKind,
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
            op2_shader_kind,
            entry_point,
        )
        .await;
}
