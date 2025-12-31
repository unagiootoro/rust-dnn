alias T = f32;

struct Layout {
    shape: array<u32, 8>,
    stride: array<u32, 8>,
    ndim: u32,
    len: u32,
    storage_offset: u32,
    _padding: u32,
};

struct SumAxisParams {
    axis: u32,
    len: u32,
}

@group(0) @binding(0) var<storage, read> lhs: array<T>;
@group(0) @binding(1) var<storage, read_write> rhs: array<T>;
@group(0) @binding(2) var<storage, read> layouts: array<Layout>;
@group(0) @binding(3) var<uniform> params: SumAxisParams;

fn compute_offset_by_axis_index(
    base_offset: u32, 
    ndim: u32, 
    linear_idx: u32, 
    axis: u32, 
    index: u32
) -> u32 {
    var offset: u32 = 0u;
    var current_linear_index = linear_idx;

    for (var i: i32 = i32(ndim) - 1; i >= 0; i = i - 1) {
        let stride = layouts[0].stride[i];
        let shape = layouts[1].shape[i];

        if (stride > 0u) {
            var idx: u32;
            if (u32(i) == axis) {
                idx = index;
            } else {
                idx = current_linear_index % shape;
            }
            offset += idx * stride;
        }
        current_linear_index /= shape;
    }
    return base_offset + offset;
}

@compute @workgroup_size(256)
fn sum_axis(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx < params.len) {
        let dim = layouts[0].shape[params.axis];
        var sum: T = 0.0;

        for (var i: u32 = 0u; i < dim; i = i + 1) {
            let a_idx = compute_offset_by_axis_index(
                layouts[0].storage_offset, 
                layouts[1].ndim, 
                idx, 
                params.axis, 
                i
            );
            sum += lhs[a_idx];
        }
        rhs[idx] = sum;
    }
}
