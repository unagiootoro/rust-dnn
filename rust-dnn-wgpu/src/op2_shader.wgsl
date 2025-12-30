// shader.wgsl

alias T = f32;

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
var<storage, read> input_a: array<T>;

@group(0) @binding(1)
var<storage, read> lhs_layout: Layout;

// バインディング1: 入力配列B (読み取り専用)
@group(0) @binding(2)
var<storage, read> input_b: array<T>;

@group(0) @binding(3)
var<storage, read> rhs_layout: Layout;

// バインディング2: 出力配列C (読み書き可能)
@group(0) @binding(4)
var<storage, read_write> output_c: array<T>;

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

// コンピュートシェーダーのエントリーポイント
// workgroup_size(64): 1つのワークグループで64スレッド並列実行
@compute @workgroup_size(64)
fn add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let dummy = lhs_layout.storage_offset + rhs_layout.storage_offset;

    // 配列の境界チェック（本来はUnfirom等でサイズを渡すべきですが、簡略化のため省略）
    // Rust側でディスパッチするサイズを調整します
    if (index < u_length.len) {
        let lhs_index = compute_offset(true, index);
        let rhs_index = compute_offset(false, index);
        output_c[index] = input_a[lhs_index] + input_b[rhs_index];
    }
}

@compute @workgroup_size(64)
fn sub(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let dummy = lhs_layout.storage_offset + rhs_layout.storage_offset;

    // 配列の境界チェック（本来はUnfirom等でサイズを渡すべきですが、簡略化のため省略）
    // Rust側でディスパッチするサイズを調整します
    if (index < u_length.len) {
        let lhs_index = compute_offset(true, index);
        let rhs_index = compute_offset(false, index);
        output_c[index] = input_a[lhs_index] - input_b[rhs_index];
    }
}

@compute @workgroup_size(64)
fn mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    // 配列の境界チェック（本来はUnfirom等でサイズを渡すべきですが、簡略化のため省略）
    // Rust側でディスパッチするサイズを調整します
    if (index < u_length.len) {
        let lhs_index = compute_offset(true, index);
        let rhs_index = compute_offset(false, index);
        output_c[index] = input_a[lhs_index] * input_b[rhs_index];
    }
}
