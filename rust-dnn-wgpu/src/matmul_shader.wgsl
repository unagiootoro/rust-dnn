// shader.wgsl

alias T1 = f32;
alias T2 = f32;

struct Layout {
    shape: array<u32, 8>,
    stride: array<u32, 8>,
    ndim: u32,                      // 4
    len: u32,                       // 4
    storage_offset: u32,            // 4
    _padding: u32,                  // 4
};

struct Length {
    len: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
}

// バインディング0: 入力配列A (読み取り専用)
@group(0) @binding(0)
var<storage, read> input_a: array<T1>;

@group(0) @binding(1)
var<storage, read> lhs_layout: Layout;

// バインディング1: 入力配列B (読み取り専用)
@group(0) @binding(2)
var<storage, read> input_b: array<T1>;

@group(0) @binding(3)
var<storage, read> rhs_layout: Layout;

// バインディング2: 出力配列C (読み書き可能)
@group(0) @binding(4)
var<storage, read_write> output_c: array<T2>;

@group(0) @binding(5)
var<uniform> u_length: Length;

fn compute_offset(is_lhs: bool, linear_index_in: u32) -> u32 {
    var offset: u32 = 0u;
    var linear_index = linear_index_in;
    
    // どちらのレイアウトを使うか判定
    var ndim: u32;
    if (is_lhs) { ndim = lhs_layout.ndim; } else { ndim = rhs_layout.ndim; }

    for (var i: i32 = i32(ndim) - 1; i >= 0; i--) {
        // 固定長配列へのアクセス
        var s_i: u32;
        var st_i: u32;
        
        // ストレージバッファとして宣言されている変数に直接アクセスすることで
        // 動的なインデックス（i）によるアクセスが許可されます
        if (is_lhs) {
            s_i = lhs_layout.shape[i];
            st_i = lhs_layout.stride[i];
        } else {
            s_i = rhs_layout.shape[i];
            st_i = rhs_layout.stride[i];
        }

        if (st_i > 0u) {
            let idx = linear_index % s_i;
            offset += idx * st_i;
        }
        linear_index /= s_i;
    }

    if (is_lhs) {
        return lhs_layout.storage_offset + offset;
    } else {
        return rhs_layout.storage_offset + offset;
    }
}

fn compute_offset_by_indices_dim2(is_lhs: bool, i: u32, j: u32) -> u32 {
    var offset: u32 = 0;
    if (is_lhs) {
        if (lhs_layout.stride[0] > 0u) {
            offset += i * lhs_layout.stride[0];
        }
        if (lhs_layout.stride[1] > 0u) {
            offset += j * lhs_layout.stride[1];
        }
        return lhs_layout.storage_offset + offset;
    } else {
        if (rhs_layout.stride[0] > 0u) {
            offset += i * rhs_layout.stride[0];
        }
        if (rhs_layout.stride[1] > 0u) {
            offset += j * rhs_layout.stride[1];
        }
        return rhs_layout.storage_offset + offset;
    }
}

@compute @workgroup_size(256)
fn matmul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    // 範囲外アクセスチェック
    if (idx >= u_length.len) {
        return;
    }

    let a_cols = lhs_layout.shape[1];
    let b_cols = rhs_layout.shape[1];

    // 出力行列 C の行 i と列 j を計算
    let i = idx / b_cols;
    let j = idx % b_cols;

    var sum: f32 = 0.0;
    
    // 内積の計算
    for (var k: u32 = 0u; k < a_cols; k = k + 1u) {
        let a_idx = compute_offset_by_indices_dim2(true, i, k);
        let b_idx = compute_offset_by_indices_dim2(false, k, j);
        
        sum += input_a[a_idx] * input_b[b_idx];
    }

    // 出力バッファへの書き込み（Cは常にContiguousであることを想定）
    output_c[idx] = sum;
}
