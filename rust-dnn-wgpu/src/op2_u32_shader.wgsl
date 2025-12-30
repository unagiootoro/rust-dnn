// shader.wgsl

// バインディング0: 入力配列A (読み取り専用)
@group(0) @binding(0)
var<storage, read> input_a: array<u32>;

// バインディング1: 入力配列B (読み取り専用)
@group(0) @binding(1)
var<storage, read> input_b: array<u32>;

// バインディング2: 出力配列C (読み書き可能)
@group(0) @binding(2)
var<storage, read_write> output_c: array<u32>;

// コンピュートシェーダーのエントリーポイント
// workgroup_size(64): 1つのワークグループで64スレッド並列実行
@compute @workgroup_size(64)
fn add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    // 配列の境界チェック（本来はUnfirom等でサイズを渡すべきですが、簡略化のため省略）
    // Rust側でディスパッチするサイズを調整します
    if (index < arrayLength(&input_a)) {
        output_c[index] = input_a[index] + input_b[index];
    }
}

@compute @workgroup_size(64)
fn mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    // 配列の境界チェック（本来はUnfirom等でサイズを渡すべきですが、簡略化のため省略）
    // Rust側でディスパッチするサイズを調整します
    if (index < arrayLength(&input_a)) {
        output_c[index] = input_a[index] * input_b[index];
    }
}
