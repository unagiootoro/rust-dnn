alias T = f32;

struct Op1Params {
    storage_offset: u32,
    len: u32,
    _padding1: u32,
    _padding2: u32,
}

@group(0) @binding(0)
var<storage, read> input: array<T>;

@group(0) @binding(1)
var<storage, read_write> output: array<T>;

@group(0) @binding(2)
var<uniform> u_params: Op1Params;

/*<FUNCTIONS>*/
