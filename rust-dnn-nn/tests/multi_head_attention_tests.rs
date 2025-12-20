mod test_utils;

use rust_dnn_core::{backend::Backend, device::Device, error::Result, ten, tensor::Tensor};
use rust_dnn_nn::{
    function::generate_causal_attention_mask,
    layer::{Layer, Linear},
    multi_head_attention::MultiHeadAttention,
    optimizer::{Optimizer, SGD},
};

use crate::test_utils::assert_tensor;

static MULTI_HEAD_ATTENTION_INPUT_Q: [f64; 24] = [
    1.9269, 1.4873, 0.9007, -2.1055, 0.6784, -1.2345, -0.0431, -1.6047, 0.3559, -0.6866, -0.4934,
    0.2415, -1.1109, 0.0915, -2.3169, -0.2168, -0.3097, -0.3957, 0.8034, -0.6216, -0.5920, -0.0631,
    -0.8286, 0.3309,
];

static MULTI_HEAD_ATTENTION_INPUT_K: [f64; 24] = [
    0.0349, 0.3211, 1.5736, -0.8455, 1.3123, 0.6872, -1.0892, -0.3553, 1.4451, 0.8564, 2.2181,
    0.5232, 0.3466, -0.1973, -1.0546, 1.2780, -0.1722, 0.5238, 0.0566, 0.4263, 0.5750, -0.6417,
    -2.2064, -0.7508,
];

static MULTI_HEAD_ATTENTION_INPUT_V: [f64; 24] = [
    0.0109, -0.3387, -1.3407, -0.5854, 0.5362, 0.5246, 1.1412, 0.0516, -0.6788, 0.5743, 0.1877,
    -0.3576, -0.3165, 0.5886, -0.8905, 0.4098, -0.9864, 0.1233, 0.3499, 0.6173, -0.1693, 0.2332,
    4.0356, 1.2795,
];

static MULTI_HEAD_ATTENTION_FORWARD_EXPECTED_DATA: [f64; 24] = [
    -0.1160, -0.1804, 0.1094, 0.0043, -0.1484, -0.2174, 0.1324, -0.0185, -0.0794, -0.1366, 0.0700,
    0.0380, -0.0084, 0.1964, -0.3087, 0.3562, -0.1851, 0.0542, -0.3498, 0.4337, -0.0787, 0.1267,
    -0.3411, 0.4347,
];

static MULTI_HEAD_ATTENTION_BACKWARDED_FORWARD_EXPECTED_DATA: [f64; 24] = [
    -1.0441770554,
    -1.2264823914,
    -0.4327356517,
    -0.8441027999,
    -1.1247389317,
    -1.3187589645,
    -0.4106158912,
    -0.9031518698,
    -0.9318404794,
    -1.0954629183,
    -0.4755461216,
    -0.7531401515,
    -1.0944588184,
    -1.0805288553,
    -0.8720381260,
    -0.6384332776,
    -1.6151299477,
    -1.6055418253,
    -0.9826686382,
    -0.8266086578,
    -1.2694242001,
    -1.2653032541,
    -0.9574123621,
    -0.6401422024,
];

static MULTI_HEAD_ATTENTION_CAUSAL_FORWARD_EXPECTED_DATA: [f64; 24] = [
    0.1938553303,
    0.0472451448,
    0.2000198662,
    -0.3203200698,
    -0.1184602454,
    -0.2540829778,
    0.2846109271,
    -0.2267526388,
    -0.0793922171,
    -0.1365877241,
    0.0699951649,
    0.0379963741,
    0.1274122149,
    0.1748600900,
    0.0034976462,
    -0.1066493243,
    0.0796283484,
    0.2320942134,
    -0.2200825363,
    0.1830746680,
    -0.0786590353,
    0.1266640723,
    -0.3411003649,
    0.4346723557,
];

static MULTI_HEAD_ATTENTION_CAUSAL_BACKWARDED_FORWARD_EXPECTED_DATA: [f64; 24] = [
    -0.6038409472,
    -0.8829690814,
    -0.2275483608,
    -1.0521076918,
    -0.8662533760,
    -1.1173563004,
    -0.1593380570,
    -0.9284650087,
    -0.8457516432,
    -1.0130581856,
    -0.3755386472,
    -0.6654089093,
    -0.7203717232,
    -0.8277562857,
    -0.3982524276,
    -0.8697534800,
    -0.7503745556,
    -0.7353624701,
    -0.6410573125,
    -0.5620816350,
    -0.8486566544,
    -0.7456749678,
    -0.8000923991,
    -0.2719947398,
];

static MULTI_HEAD_ATTENTION_Q_PROJ_WEIGHT: [f64; 16] = [
    0.2384107709,
    -0.5025516748,
    0.4546431899,
    -0.4495142698,
    -0.1057404280,
    0.1279060245,
    0.3161383867,
    0.4943745732,
    0.5578463674,
    -0.4855636954,
    0.1541141272,
    -0.2633972764,
    -0.0671067238,
    -0.4583547115,
    0.5577847362,
    -0.4494510889,
];

static MULTI_HEAD_ATTENTION_K_PROJ_WEIGHT: [f64; 16] = [
    0.3272832036,
    0.2152119279,
    0.1989940405,
    -0.3310768306,
    0.5566168427,
    0.1345691085,
    0.0787756443,
    -0.5396561623,
    0.2570669055,
    -0.0918684602,
    -0.2805427015,
    0.5259951949,
    0.1365277171,
    -0.3388112485,
    -0.3099455237,
    -0.0292443037,
];

static MULTI_HEAD_ATTENTION_V_PROJ_WEIGHT: [f64; 16] = [
    0.3419250846,
    -0.1564818621,
    -0.3494043350,
    -0.2097025812,
    -0.4574880600,
    0.2183918953,
    0.4740008712,
    -0.5765121579,
    0.1422239542,
    0.3163465858,
    0.1110411286,
    -0.2180809081,
    0.3196229935,
    0.3218097687,
    0.2289827466,
    -0.1076069474,
];

static MULTI_HEAD_ATTENTION_OUT_PROJ_WEIGHT: [f64; 16] = [
    0.0013856292,
    -0.1860516071,
    -0.0346478820,
    -0.3388148546,
    -0.3431975842,
    -0.2917008996,
    -0.1711487174,
    -0.3946404457,
    0.4192349315,
    -0.0992320180,
    0.4301983714,
    0.1557910442,
    -0.4233984947,
    0.3460175991,
    -0.1375724077,
    -0.1916630268,
];

static MULTI_HEAD_ATTENTION_Q_PROJ_BIAS: [f64; 4] = [0., 0., 0., 0.];

static MULTI_HEAD_ATTENTION_K_PROJ_BIAS: [f64; 4] = [0., 0., 0., 0.];

static MULTI_HEAD_ATTENTION_V_PROJ_BIAS: [f64; 4] = [0., 0., 0., 0.];

static MULTI_HEAD_ATTENTION_OUT_PROJ_BIAS: [f64; 4] = [0., 0., 0., 0.];

fn create_multi_head_attention<B: Backend>(
    device: Device<B>,
) -> Result<MultiHeadAttention<B, f64>> {
    let multi_head_attention = MultiHeadAttention::new(4, 4, 0.0, true, device);

    let q_proj_weight = Tensor::from_vec(
        MULTI_HEAD_ATTENTION_Q_PROJ_WEIGHT.to_vec(),
        vec![4, 4],
        device,
    );
    multi_head_attention
        .q_proj()
        .weight()
        .copy(&q_proj_weight);

    let q_proj_bias = Tensor::from_vec(MULTI_HEAD_ATTENTION_Q_PROJ_BIAS.to_vec(), vec![4], device);
    multi_head_attention
        .q_proj()
        .bias()
        .as_ref()
        .unwrap()
        .copy(&q_proj_bias);

    let k_proj_weight = Tensor::from_vec(
        MULTI_HEAD_ATTENTION_K_PROJ_WEIGHT.to_vec(),
        vec![4, 4],
        device,
    );
    multi_head_attention
        .k_proj()
        .weight()
        .copy(&k_proj_weight);

    let k_proj_bias = Tensor::from_vec(MULTI_HEAD_ATTENTION_K_PROJ_BIAS.to_vec(), vec![4], device);
    multi_head_attention
        .k_proj()
        .bias()
        .as_ref()
        .unwrap()
        .copy(&k_proj_bias);

    let v_proj_weight = Tensor::from_vec(
        MULTI_HEAD_ATTENTION_V_PROJ_WEIGHT.to_vec(),
        vec![4, 4],
        device,
    );
    multi_head_attention
        .v_proj()
        .weight()
        .copy(&v_proj_weight);

    let v_proj_bias = Tensor::from_vec(MULTI_HEAD_ATTENTION_V_PROJ_BIAS.to_vec(), vec![4], device);
    multi_head_attention
        .v_proj()
        .bias()
        .as_ref()
        .unwrap()
        .copy(&v_proj_bias);

    let out_proj_weight = Tensor::from_vec(
        MULTI_HEAD_ATTENTION_OUT_PROJ_WEIGHT.to_vec(),
        vec![4, 4],
        device,
    );
    multi_head_attention
        .out_proj()
        .weight()
        .copy(&out_proj_weight);

    let out_proj_bias =
        Tensor::from_vec(MULTI_HEAD_ATTENTION_OUT_PROJ_BIAS.to_vec(), vec![4], device);
    multi_head_attention
        .out_proj()
        .bias()
        .as_ref()
        .unwrap()
        .copy(&out_proj_bias);

    Ok(multi_head_attention)
}

fn test_multi_head_attention_forward<B: Backend>(device: Device<B>) -> Result<()> {
    let q = Tensor::from_vec(MULTI_HEAD_ATTENTION_INPUT_Q.to_vec(), vec![2, 3, 4], device);
    let k = Tensor::from_vec(MULTI_HEAD_ATTENTION_INPUT_K.to_vec(), vec![2, 3, 4], device);
    let v = Tensor::from_vec(MULTI_HEAD_ATTENTION_INPUT_V.to_vec(), vec![2, 3, 4], device);
    let multi_head_attention = create_multi_head_attention(device)?;
    let y = multi_head_attention.forward(&q, &k, &v, None);
    let expected_y = Tensor::from_vec(
        MULTI_HEAD_ATTENTION_FORWARD_EXPECTED_DATA.to_vec(),
        vec![2, 3, 4],
        device,
    );
    assert_tensor(&y, &expected_y);
    Ok(())
}

define_test!(
    test_multi_head_attention_forward,
    test_multi_head_attention_forward_cpu,
    test_multi_head_attention_forward_cuda
);

fn test_multi_head_attention_mask_forward<B: Backend>(device: Device<B>) -> Result<()> {
    let q = Tensor::from_vec(MULTI_HEAD_ATTENTION_INPUT_Q.to_vec(), vec![2, 3, 4], device)
        .requires_grad();
    let k = Tensor::from_vec(MULTI_HEAD_ATTENTION_INPUT_K.to_vec(), vec![2, 3, 4], device)
        .requires_grad();
    let v = Tensor::from_vec(MULTI_HEAD_ATTENTION_INPUT_V.to_vec(), vec![2, 3, 4], device)
        .requires_grad();
    let multi_head_attention = create_multi_head_attention(device)?;
    let mask = generate_causal_attention_mask(q.shape()[1], device)?;
    let y = multi_head_attention.forward(&q, &k, &v, Some(&mask));
    let expected_y = Tensor::from_vec(
        MULTI_HEAD_ATTENTION_CAUSAL_FORWARD_EXPECTED_DATA.to_vec(),
        vec![2, 3, 4],
        device,
    );
    assert_tensor(&y, &expected_y);
    Ok(())
}

define_test!(
    test_multi_head_attention_mask_forward,
    test_multi_head_attention_mask_forward_cpu,
    test_multi_head_attention_mask_forward_cuda
);

fn test_multi_head_attention_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let q = Tensor::from_vec(MULTI_HEAD_ATTENTION_INPUT_Q.to_vec(), vec![2, 3, 4], device);
    let k = Tensor::from_vec(MULTI_HEAD_ATTENTION_INPUT_K.to_vec(), vec![2, 3, 4], device);
    let v = Tensor::from_vec(MULTI_HEAD_ATTENTION_INPUT_V.to_vec(), vec![2, 3, 4], device);
    let multi_head_attention = create_multi_head_attention(device)?;
    let y = multi_head_attention.forward(&q, &k, &v, None);
    let expected_y = Tensor::from_vec(
        MULTI_HEAD_ATTENTION_FORWARD_EXPECTED_DATA.to_vec(),
        vec![2, 3, 4],
        device,
    );
    assert_tensor(&y, &expected_y);

    let loss = y.sum();
    let grads = loss.backward();
    let mut optimizer = SGD::new(0.1);
    optimizer.update_parameters(&mut multi_head_attention.all_parameters_map(), &grads);

    let y2 = multi_head_attention.forward(&q, &k, &v, None);
    let expected_y2 = Tensor::from_vec(
        MULTI_HEAD_ATTENTION_BACKWARDED_FORWARD_EXPECTED_DATA.to_vec(),
        vec![2, 3, 4],
        device,
    );
    assert_tensor(&y2, &expected_y2);

    Ok(())
}

define_test!(
    test_multi_head_attention_backward,
    test_multi_head_attention_backward_cpu,
    test_multi_head_attention_backward_cuda
);

fn test_multi_head_attention_mask_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let q = Tensor::from_vec(MULTI_HEAD_ATTENTION_INPUT_Q.to_vec(), vec![2, 3, 4], device);
    let k = Tensor::from_vec(MULTI_HEAD_ATTENTION_INPUT_K.to_vec(), vec![2, 3, 4], device);
    let v = Tensor::from_vec(MULTI_HEAD_ATTENTION_INPUT_V.to_vec(), vec![2, 3, 4], device);
    let multi_head_attention = create_multi_head_attention(device)?;
    let mask = generate_causal_attention_mask(q.shape()[1], device)?;
    let y = multi_head_attention.forward(&q, &k, &v, Some(&mask));
    let expected_y = Tensor::from_vec(
        MULTI_HEAD_ATTENTION_CAUSAL_FORWARD_EXPECTED_DATA.to_vec(),
        vec![2, 3, 4],
        device,
    );
    assert_tensor(&y, &expected_y);

    let loss = y.sum();
    let grads = loss.backward();
    let mut optimizer = SGD::new(0.1);
    optimizer.update_parameters(&mut multi_head_attention.all_parameters_map(), &grads);

    let y2 = multi_head_attention.forward(&q, &k, &v, Some(&mask));
    let expected_y2 = Tensor::from_vec(
        MULTI_HEAD_ATTENTION_CAUSAL_BACKWARDED_FORWARD_EXPECTED_DATA.to_vec(),
        vec![2, 3, 4],
        device,
    );
    assert_tensor(&y2, &expected_y2);

    Ok(())
}

define_test!(
    test_multi_head_attention_mask_backward,
    test_multi_head_attention_mask_backward_cpu,
    test_multi_head_attention_mask_backward_cuda
);
