alias T = f32;

struct Layout {
    shape: array<u32, 8>,
    stride: array<u32, 8>,
    ndim: u32,
    len: u32,
    storage_offset: u32,
    _padding: u32,
};

struct Length {
    len: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
}

@group(0) @binding(0) var<storage, read> input: array<T>;
@group(0) @binding(1) var<storage, read_write> output: array<T>;
@group(0) @binding(2) var<storage, read> input_layout: Layout;
@group(0) @binding(3) var<uniform> u_length: Length;

fn compute_offset(linear_index_in: u32) -> u32 {
    var offset: u32 = 0u;
    var linear_index = linear_index_in;

    for (var i: i32 = i32(input_layout.ndim) - 1; i >= 0; i--) {
        // 固定長配列へのアクセス
        var s_i: u32;
        var st_i: u32;

        s_i = input_layout.shape[i];
        st_i = input_layout.stride[i];

        if (st_i > 0u) {
            let idx = linear_index % s_i;
            offset += idx * st_i;
        }
        linear_index /= s_i;
    }

    return input_layout.storage_offset + offset;
}
@compute @workgroup_size(64)
fn array_contiguous(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if (index < u_length.len) {
        let input_index = compute_offset(index);
        output[index] = input[input_index];
    }
}
