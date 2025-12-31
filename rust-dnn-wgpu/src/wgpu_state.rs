use std::borrow::Cow;

use bytemuck::{AnyBitPattern, NoUninit};
use wgpu::{Label, util::DeviceExt};

use crate::{
    Length, Op1Params, Op2ShaderKind, layout::Layout, wgpu_buffer::WgpuBuffer,
    wgpu_dtype::WgpuDTypeKind,
};

pub struct WGPUState {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    op1_f32_shader: wgpu::ShaderModule,
    op2_u32_u32_shader: wgpu::ShaderModule,
    op2_f32_f32_shader: wgpu::ShaderModule,
    op2_f32_u32_shader: wgpu::ShaderModule,
    op2_float_f32_f32_shader: wgpu::ShaderModule,
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

        let op1_f32_shader_src = Self::generate_op1_shader_src();

        let op1_f32_shader =
            Self::create_shader_module2(&device, Some("op1_f32_shader_src"), &op1_f32_shader_src);

        let op2_shader_src = Self::generate_op2_shader_src();
        let op2_u32_u32_shader_src = op2_shader_src
            .replace("alias T1 = f32;", "alias T1 = u32;")
            .replace("alias T2 = f32;", "alias T2 = u32;");
        let op2_f32_f32_shader_src = op2_shader_src
            .replace("alias T1 = f32;", "alias T1 = f32;")
            .replace("alias T2 = f32;", "alias T2 = f32;");

        let op2_cmp_shader_src = Self::generate_op2_cmp_shader_src();
        let op2_f32_u32_shader_src = op2_cmp_shader_src
            .replace("alias T1 = f32;", "alias T1 = f32;")
            .replace("alias T2 = f32;", "alias T2 = u32;");

        let op2_u32_u32_shader = Self::create_shader_module2(
            &device,
            Some("op2_u32_u32_shader"),
            &op2_u32_u32_shader_src,
        );

        let op2_f32_f32_shader = Self::create_shader_module2(
            &device,
            Some("op2_f32_f32_shader"),
            &op2_f32_f32_shader_src,
        );

        let op2_f32_u32_shader = Self::create_shader_module2(
            &device,
            Some("op2_f32_u32_shader"),
            &op2_f32_u32_shader_src,
        );

        let op2_float_shader_src = Self::generate_op2_float_shader_src();
        let op2_float_f32_f32_shader_src = op2_float_shader_src
            .replace("alias T1 = f32;", "alias T1 = f32;")
            .replace("alias T2 = f32;", "alias T2 = f32;");

        let op2_float_f32_f32_shader = Self::create_shader_module2(
            &device,
            Some("op2_float_f32_shader"),
            &op2_float_f32_f32_shader_src,
        );

        Self {
            op1_f32_shader,
            op2_u32_u32_shader,
            op2_f32_f32_shader,
            op2_f32_u32_shader,
            op2_float_f32_f32_shader,
            adapter,
            device,
            queue,
        }
    }

    fn generate_op1_shader_src() -> String {
        let mut src = include_str!("./op1_shader.wgsl").to_string();
        let mut functions = String::new();
        functions += &Self::op1_shader_func_def("array_exp", "exp");
        functions += &Self::op1_shader_func_def("neg", "-");
        src = src.replace("/*<FUNCTIONS>*/", &functions);
        src
    }

    fn op1_shader_func_def(function_name: &str, op: &str) -> String {
        let mut src = "
@compute @workgroup_size(64)
fn <FUNCTION_NAME>(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if (index < u_params.len) {
        output[index] = <OP>(input[index + u_params.storage_offset]);
    }
}
        "
        .to_string();
        src = src.replace("<FUNCTION_NAME>", function_name);
        src = src.replace("<OP>", op);
        src
    }

    fn generate_op2_shader_src() -> String {
        let mut src = include_str!("./op2_shader.wgsl").to_string();
        let mut functions = String::new();
        functions += &Self::build_op2_func_def("add", "+");
        functions += &Self::build_op2_func_def("sub", "-");
        functions += &Self::build_op2_func_def("mul", "*");
        functions += &Self::build_op2_func_def("div", "/");
        src = src.replace("/*<FUNCTIONS>*/", &functions);
        src
    }

    fn generate_op2_cmp_shader_src() -> String {
        let mut src = include_str!("./op2_shader.wgsl").to_string();
        let mut functions = String::new();
        functions += &Self::op2_cmp_shader_func_def("eq", "==");
        functions += &Self::op2_cmp_shader_func_def("lt", "<");
        functions += &Self::op2_cmp_shader_func_def("le", "<=");
        functions += &Self::op2_cmp_shader_func_def("gt", ">");
        functions += &Self::op2_cmp_shader_func_def("ge", ">=");
        src = src.replace("/*<FUNCTIONS>*/", &functions);
        src
    }

    fn generate_op2_float_shader_src() -> String {
        let mut src = include_str!("./op2_shader.wgsl").to_string();
        let mut functions = String::new();
        functions += &Self::op2_shader_func_def("array_pow", "pow");
        src = src.replace("/*<FUNCTIONS>*/", &functions);
        src
    }

    fn build_op2_func_def(function_name: &str, op: &str) -> String {
        let mut src = "
@compute @workgroup_size(64)
fn <FUNCTION_NAME>(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index < u_length.len) {
        let lhs_index = compute_offset(true, index);
        let rhs_index = compute_offset(false, index);
        output_c[index] = input_a[lhs_index] <OP> input_b[rhs_index];
    }
}
        "
        .to_string();
        src = src.replace("<FUNCTION_NAME>", function_name);
        src = src.replace("<OP>", op);
        src
    }

    fn op2_shader_func_def(function_name: &str, op: &str) -> String {
        let mut src = "
@compute @workgroup_size(64)
fn <FUNCTION_NAME>(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index < u_length.len) {
        let lhs_index = compute_offset(true, index);
        let rhs_index = compute_offset(false, index);
        output_c[index] = <OP>(input_a[lhs_index], input_b[rhs_index]);
    }
}
        "
        .to_string();
        src = src.replace("<FUNCTION_NAME>", function_name);
        src = src.replace("<OP>", op);
        src
    }

    fn op2_cmp_shader_func_def(function_name: &str, op: &str) -> String {
        let mut src = "
@compute @workgroup_size(64)
fn <FUNCTION_NAME>(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index < u_length.len) {
        let lhs_index = compute_offset(true, index);
        let rhs_index = compute_offset(false, index);
        output_c[index] = select(0u, 1u, input_a[lhs_index] <OP> input_b[rhs_index]);
    }
}
        "
        .to_string();
        src = src.replace("<FUNCTION_NAME>", function_name);
        src = src.replace("<OP>", op);
        src
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

    pub async fn op1(
        &self,
        input: &WgpuBuffer,
        storage_offset: u32,
        output: &WgpuBuffer,
        len: u32,
        entry_point: &str,
    ) {
        assert_eq!(input.dtype(), output.dtype());

        let shader_module = match input.dtype() {
            WgpuDTypeKind::U32 => todo!(),
            WgpuDTypeKind::F32 => &self.op1_f32_shader,
        };

        let compute_pipeline = self.create_compute_pipeline(shader_module, entry_point);

        let u_params = Op1Params::new(storage_offset, len);

        let params_uniform_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("params_uniform_buffer"),
                    contents: bytemuck::cast_slice(&[u_params]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.raw.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.raw.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_uniform_buffer.as_entire_binding(),
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

    pub async fn op2(
        &self,
        buffer_a: &WgpuBuffer,
        lhs_layout: Layout,
        buffer_b: &WgpuBuffer,
        rhs_layout: Layout,
        buffer_c: &WgpuBuffer,
        len: u32,
        op2_shader_kind: Op2ShaderKind,
        entry_point: &str,
    ) {
        assert_eq!(buffer_a.dtype(), buffer_b.dtype());

        let shader_module = match (buffer_a.dtype(), buffer_c.dtype(), op2_shader_kind) {
            (WgpuDTypeKind::U32, WgpuDTypeKind::U32, Op2ShaderKind::Num) => {
                &self.op2_u32_u32_shader
            }
            (WgpuDTypeKind::F32, WgpuDTypeKind::F32, Op2ShaderKind::Num) => {
                &self.op2_f32_f32_shader
            }
            (WgpuDTypeKind::F32, WgpuDTypeKind::U32, Op2ShaderKind::Num) => {
                &self.op2_f32_u32_shader
            }
            (WgpuDTypeKind::F32, WgpuDTypeKind::F32, Op2ShaderKind::Float) => {
                &self.op2_float_f32_f32_shader
            }
            _ => todo!(),
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
