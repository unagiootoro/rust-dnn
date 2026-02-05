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

fn is_nan_f32(x: f32) -> bool {
    return x != x;
}

fn make_nan() -> f32 {
    return bitcast<f32>(0x7fc00000u);
}

fn make_inf() -> f32 {
    return bitcast<f32>(0x7f800000u);
}

fn pow_general(x: f32, y: f32) -> f32 {
    // NaN 伝播（最優先）
    if (is_nan_f32(x) || is_nan_f32(y)) {
        return make_nan();
    }

    // x == 0
    if (x == 0.0) {
        if (y > 0.0) {
            return 0.0;
        }
        if (y == 0.0) {
            return 1.0;
        }
        return make_inf(); // 0^負数
    }

    // x < 0
    if (x < 0.0) {
        let yi = round(y);
        if (abs(y - yi) > 1e-6) {
            // 負数 + 非整数指数 → NaN
            return make_nan();
        }

        // 整数指数
        let p = exp(y * log(-x));
        // 奇数なら符号反転
        return select(p, -p, (i32(yi) & 1) != 0);
    }

    // x > 0（最頻出パス）
    return exp(y * log(x));
}

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

/*<FUNCTIONS>*/
