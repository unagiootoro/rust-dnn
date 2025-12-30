pub mod layout;
pub mod shader_type;
pub mod wgpu_buffer;

use bytemuck::{AnyBitPattern, NoUninit};
use std::cell::RefCell;
use std::{borrow::Cow, rc::Rc};
use wgpu::{Label, util::DeviceExt};

use crate::layout::Layout;
use crate::shader_type::ShaderType;
use crate::wgpu_buffer::WgpuBuffer; // バッファ初期化のためのユーティリティ

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
        Self { len, _padding1: 0, _padding2: 0, _padding3: 0 }
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

struct WGPUState {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    op2_u32_shader: wgpu::ShaderModule,
    op2_f32_shader: wgpu::ShaderModule,
}

impl WGPUState {
    pub async fn new() -> Self {
        // --- 1. インスタンス、アダプター、デバイスの初期化 ---

        // WGPUのインスタンスを作成
        let instance = wgpu::Instance::default();

        // アダプター（実際のGPUハンドル）を取得
        let adapter: wgpu::Adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("Failed to find an appropriate adapter");

        let adapter_limits = adapter.limits();

        // デバイス（論理デバイス）とキュー（コマンド送信口）を取得
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        // ここでストレージバッファの最大数を、
                        // アダプターが許容する最大値（通常8〜16以上）まで引き上げます
                        max_storage_buffers_per_shader_stage: adapter_limits
                            .max_storage_buffers_per_shader_stage,
                        ..wgpu::Limits::downlevel_defaults()
                    },
                },
                None,
            )
            .await
            .expect("Failed to create device");
        
        let op2_shader_src = include_str!("./op2_shader.wgsl").to_string();
        let op2_u32_shader_src = op2_shader_src.replace("alias T = f32;", "alias T = u32;");
        let op2_f32_shader_src = op2_shader_src.replace("alias T = f32;", "alias T = f32;");

        let op2_u32_shader = Self::create_shader_module2(
            &device,
            Some("op2_u32_shader"),
            &op2_u32_shader_src,
        );

        let op2_f32_shader = Self::create_shader_module2(
            &device,
            Some("op2_f32_shader"),
            &op2_f32_shader_src,
        );

        Self {
            op2_u32_shader,
            op2_f32_shader,
            adapter,
            device,
            queue,
        }
    }

    pub fn create_buffer<'a, T: NoUninit>(&self, label: Label<'a>, data: &[T]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label,
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
    }

    pub fn create_stage_buffer<'a>(&self, label: Label<'a>, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, // コピー先かつマップして読む
            mapped_at_creation: false,
        })
    }

    pub fn create_shader_module<'a>(&self, label: Label<'a>, source: &str) -> wgpu::ShaderModule {
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
            })
    }

    pub fn create_shader_module2<'a>(
        device: &wgpu::Device,
        label: Label<'a>,
        source: &str,
    ) -> wgpu::ShaderModule {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
        })
    }

    pub fn create_compute_pipeline(
        &self,
        cs_module: &wgpu::ShaderModule,
        entry_point: &str,
    ) -> wgpu::ComputePipeline {
        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Compute Pipeline"),
                    layout: None, // レイアウトはバインドグループから自動推論させる
                    module: cs_module,
                    entry_point,
                });

        compute_pipeline
    }

    pub async fn op2(
        &self,
        buffer_a: &WgpuBuffer,
        lhs_layout: Layout,
        buffer_b: &WgpuBuffer,
        rhs_layout: Layout,
        buffer_c: &WgpuBuffer,
        len: u32,
        shader_type: ShaderType,
        entry_point: &str,
    ) {
        let shader_module = match shader_type {
            ShaderType::Op2U32 => &self.op2_u32_shader,
            ShaderType::Op2F32 => &self.op2_f32_shader,
        };

        // パイプラインの作成
        let compute_pipeline = self.create_compute_pipeline(shader_module, entry_point);

        // --- 5. バインドグループの作成 ---
        // バッファとシェーダー内の変数(binding)を紐付ける

        let lhs_layout_uniform_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("lhs_layout_uniform_buffer"),
                    contents: bytemuck::cast_slice(&[lhs_layout]),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

        let rhs_layout_uniform_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("rhs_layout_uniform_buffer"),
                    contents: bytemuck::cast_slice(&[rhs_layout]),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
        
        let u_length = Length::new(len);

        let len_uniform_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("len_uniform_buffer"),
                    contents: bytemuck::cast_slice(&[u_length]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_a.raw.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lhs_layout_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_b.raw.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: rhs_layout_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffer_c.raw.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: len_uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // --- 6. コマンドのエンコードと実行 ---

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Command Encoder"),
            });

        {
            // コンピュートパスの開始
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            // ワークグループの数を計算 (データ数64 / ワークグループサイズ64 = 1)
            // cpass.dispatch_workgroups(data_a_len as u32 / 64, 1, 1);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        // コマンドをキューに送信して実行
        self.queue.submit(Some(encoder.finish()));
    }

    pub async fn buffer_to_vec<T: AnyBitPattern>(
        &self,
        buffer: &wgpu::Buffer,
        len: usize,
    ) -> Vec<T> {
        let data_size = len as u64 * size_of::<T>() as u64; // u32は4バイト

        // ステージングバッファ (GPUからCPUへデータを読み戻すための中継バッファ)
        let staging_buffer = self.create_stage_buffer(Some("Staging Buffer"), data_size);

        // --- 6. コマンドのエンコードと実行 ---

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Command Encoder"),
            });

        // 計算結果を出力バッファからステージングバッファへコピー
        encoder.copy_buffer_to_buffer(&buffer, 0, &staging_buffer, 0, data_size);

        // コマンドをキューに送信して実行
        self.queue.submit(Some(encoder.finish()));

        // --- 7. 結果の読み取り ---

        // ステージングバッファをCPUから読めるようにマップする
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        // デバイスをポールしてコールバックを待機
        self.device.poll(wgpu::Maintain::Wait);

        // マップ完了待ち
        if let Some(Ok(())) = receiver.receive().await {
            // データを読み取る
            let data = buffer_slice.get_mapped_range();
            // [u8] を [u32] にキャスト
            let result: &[T] = bytemuck::cast_slice(&data);
            let result = result.to_vec();

            // マップを解除 (dropされるときに自動で行われるが明示的にも可能)
            drop(data);
            staging_buffer.unmap();

            result
        } else {
            panic!("Failed to retrieve data from GPU");
        }
    }
}
