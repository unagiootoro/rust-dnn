mod test_utils;

use crate::test_utils::{arange_with_shape, assert_tensor, assert_tensor_with_eps};
use rust_dnn_core::{backend::Backend, device::Device, error::Result, ten, tensor::Tensor};
use rust_dnn_nn::layer::{BatchNorm1d, BatchNorm2d, Conv2D, Linear};

fn test_linear_forward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[0.0, -2.0, -4.0], [1.0, 2.0, 3.0]]
        .to_device(device)?
        .requires_grad();
    let weight = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let bias = ten![1.0, 2.0].to_device(device)?.requires_grad();
    let linear = Linear::from_weights(weight, Some(bias))?;
    let y = linear.forward(&x)?;
    assert_tensor(&y, &ten![[-15.0, -32.0], [15.0, 34.0]]);
    Ok(())
}

define_test!(
    test_linear_forward,
    test_linear_forward_cpu,
    test_linear_forward_cuda
);

fn test_linear_no_bias_forward<B: Backend>(device: Device<B>) -> Result<()> {
    let x: rust_dnn_core::tensor::Tensor<B, f64> = ten![[0.0, -2.0, -4.0], [1.0, 2.0, 3.0]]
        .to_device(device)?
        .requires_grad();
    let weight = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let linear = Linear::from_weights(weight, None)?;
    let y = linear.forward(&x)?;
    assert_tensor(&y, &ten![[-16.0, -34.0], [14.0, 32.0]]);
    Ok(())
}

define_test!(
    test_linear_no_bias_forward,
    test_linear_no_bias_forward_cpu,
    test_linear_no_bias_forward_cuda
);

fn test_linear_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[0.0, -2.0, -4.0], [1.0, 2.0, 3.0]]
        .to_device(device)?
        .requires_grad();
    let weight = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let bias = ten![1.0, 2.0].to_device(device)?.requires_grad();
    let linear = Linear::from_weights(weight.clone(), Some(bias.clone()))?;
    let y = linear.forward(&x)?;
    assert_tensor(&y, &ten![[-15.0, -32.0], [15.0, 34.0]]);
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap();
    let gw = grads.get(&weight).unwrap();
    let gb = grads.get(&bias).unwrap();
    assert_tensor(&gx, &ten![[5.0, 7.0, 9.0], [5.0, 7.0, 9.0]]);
    assert_tensor(&gw, &ten![[1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]);
    assert_tensor(&gb, &ten![2.0, 2.0]);
    Ok(())
}

define_test!(
    test_linear_backward,
    test_linear_backward_cpu,
    test_linear_backward_cuda
);

fn test_linear_no_bias_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[0.0, -2.0, -4.0], [1.0, 2.0, 3.0]]
        .to_device(device)?
        .requires_grad();
    let weight = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let linear = Linear::from_weights(weight.clone(), None)?;
    let y = linear.forward(&x)?;
    assert_tensor(&y, &ten![[-16.0, -34.0], [14.0, 32.0]]);
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap();
    let gw = grads.get(&weight).unwrap();
    assert_tensor(&gx, &ten![[5.0, 7.0, 9.0], [5.0, 7.0, 9.0]]);
    assert_tensor(&gw, &ten![[1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]);
    Ok(())
}

define_test!(
    test_linear_no_bias_backward,
    test_linear_no_bias_backward_cpu,
    test_linear_no_bias_backward_cuda
);

fn test_conv2d<B: Backend>(device: Device<B>) -> Result<()> {
    let batch_size = 2;
    let in_filters = 3;
    let out_filters = 2;
    let img_h = 4;
    let img_w = 5;
    let fil_h = 2;
    let fil_w = 3;
    let x = arange_with_shape::<_, f64>(&[batch_size, in_filters, img_h, img_w], device);
    let w = arange_with_shape(&[out_filters, in_filters, fil_h, fil_w], device);
    let b = arange_with_shape(&[out_filters], device);

    let conv2d = Conv2D::new(
        in_filters,
        out_filters,
        fil_h,
        fil_w,
        1,
        1,
        None,
        false,
        true,
        device,
    );

    conv2d.weight().copy(&w)?;
    conv2d.bias().unwrap().copy(&b)?;

    let y = conv2d.forward(&x)?;
    assert_eq!(y.shape(), &vec![batch_size, out_filters, 3, 3]);
    assert_eq!(
        y.to_vec(),
        vec![
            5115., 5268., 5421., 5880., 6033., 6186., 6645., 6798., 6951., 12730., 13207., 13684.,
            15115., 15592., 16069., 17500., 17977., 18454., 14295., 14448., 14601., 15060., 15213.,
            15366., 15825., 15978., 16131., 41350., 41827., 42304., 43735., 44212., 44689., 46120.,
            46597., 47074.,
        ]
    );
    Ok(())
}

define_test!(test_conv2d, test_conv2d_cpu, test_conv2d_cuda);

fn test_batch_norm1d<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::from_vec(
        vec![1.0, 5.0, 3.0, 2.0, 1.0, 6.0, 4.0, 2.0, 1.0, 3.0, 7.0, 8.0],
        vec![4, 3],
        device,
    )?
    .requires_grad();
    let gamma = Tensor::from_vec(vec![1.0, 1.5, 2.0], vec![3], device)?.requires_grad();
    let beta = Tensor::from_vec(vec![0.0, 0.5, -1.0], vec![3], device)?.requires_grad();
    let mut batch_norm1d = BatchNorm1d::new(3, 0.9, 1e-7, device);
    batch_norm1d.gamma().copy(&gamma)?;
    batch_norm1d.beta().copy(&beta)?;
    let y = batch_norm1d.forward(&x, true)?;
    assert_tensor(
        &y,
        &ten![
            [-1.3416, 1.2862, -2.1142],
            [-0.4472, -1.2297, 0.1142],
            [1.3416, -0.6007, -3.5997],
            [0.4472, 2.5442, 1.5997,]
        ],
    );
    Ok(())
}

define_test!(
    test_batch_norm1d,
    test_batch_norm1d_cpu,
    test_batch_norm1d_cuda
);

fn test_batch_norm1d_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::from_vec(
        vec![1.0, 5.0, 3.0, 2.0, 1.0, 6.0, 4.0, 2.0, 1.0, 3.0, 7.0, 8.0],
        vec![4, 3],
        device,
    )?
    .requires_grad();
    let gamma = Tensor::from_vec(vec![1.0, 1.5, 2.0], vec![3], device)?;
    let beta = Tensor::from_vec(vec![0.0, 0.5, -1.0], vec![3], device)?;
    let mut batch_norm1d = BatchNorm1d::new(3, 0.9, 1e-7, device);
    batch_norm1d.gamma().copy(&gamma)?;
    batch_norm1d.beta().copy(&beta)?;
    let y = batch_norm1d.forward(&x, true)?;
    assert_tensor(
        &y,
        &ten![
            [-1.3416, 1.2862, -2.1142],
            [-0.4472, -1.2297, 0.1142],
            [1.3416, -0.6007, -3.5997],
            [0.4472, 2.5442, 1.5997,]
        ],
    );

    let x2 = Tensor::from_vec(
        vec![
            1.0, 0.0, -1.0, -1.0, 2.0, 0.0, 0.5, -0.5, 1.0, -1.5, 1.0, -0.5,
        ],
        vec![4, 3],
        device,
    )?;
    let y = (&y * x2)?;

    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap();
    let ggamma = grads.get(&batch_norm1d.gamma()).unwrap();
    let gbeta = grads.get(&batch_norm1d.beta()).unwrap();
    assert_tensor(
        &gx,
        &ten![
            [0.8497, -0.3456, -0.7940],
            [-0.7603, 0.7603, 0.2369],
            [0.9391, -0.7741, 0.4995],
            [-1.0286, 0.3594, 0.0576],
        ],
    );
    assert_tensor(&ggamma, &ten![-0.8944, -0.5766, -1.3927]);
    assert_tensor(&gbeta, &ten![-1.0000, 2.5000, -0.5000]);
    Ok(())
}

define_test!(
    test_batch_norm1d_backward,
    test_batch_norm1d_backward_cpu,
    test_batch_norm1d_backward_cuda
);

static BATCH_NORM2D_FORWARD_EXPECTED_DATA: [f64; 360] = [
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.6848, -0.1072, 0.4705, 1.0481, 1.6258,
    2.2034, -0.5693, 0.0084, 0.5860, 1.1637, 1.7413, 2.3190, -0.4538, 0.1239, 0.7015, 1.2792,
    1.8569, 2.4345, -0.3382, 0.2394, 0.8171, 1.3947, 1.9724, 2.5500, -0.2227, 0.3550, 0.9326,
    1.5103, 2.0879, 2.6656, -1.3696, -0.2143, 0.9410, 2.0963, 3.2516, 4.4069, -1.1386, 0.0167,
    1.1720, 2.3273, 3.4826, 4.6380, -0.9075, 0.2478, 1.4031, 2.5584, 3.7137, 4.8690, -0.6765,
    0.4788, 1.6342, 2.7895, 3.9448, 5.1001, -0.4454, 0.7099, 1.8652, 3.0205, 4.1758, 5.3311,
    -2.0545, -0.3215, 1.4115, 3.1444, 4.8774, 6.6103, -1.7079, 0.0251, 1.7580, 3.4910, 5.2240,
    6.9569, -1.3613, 0.3717, 2.1046, 3.8376, 5.5706, 7.3035, -1.0147, 0.7183, 2.4512, 4.1842,
    5.9171, 7.6501, -0.6681, 1.0649, 2.7978, 4.5308, 6.2637, 7.9967, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, -0.6752, -0.0975, 0.4801, 1.0578, 1.6354, 2.2131, -0.5597, 0.0180,
    0.5956, 1.1733, 1.7509, 2.3286, -0.4441, 0.1335, 0.7112, 1.2888, 1.8665, 2.4441, -0.3286,
    0.2491, 0.8267, 1.4044, 1.9820, 2.5597, -0.2131, 0.3646, 0.9422, 1.5199, 2.0975, 2.6752,
    -1.3504, -0.1951, 0.9602, 2.1155, 3.2708, 4.4261, -1.1193, 0.0360, 1.1913, 2.3466, 3.5019,
    4.6572, -0.8883, 0.2670, 1.4223, 2.5777, 3.7330, 4.8883, -0.6572, 0.4981, 1.6534, 2.8087,
    3.9640, 5.1193, -0.4261, 0.7292, 1.8845, 3.0398, 4.1951, 5.3504, -2.0256, -0.2926, 1.4403,
    3.1733, 4.9063, 6.6392, -1.6790, 0.0540, 1.7869, 3.5199, 5.2528, 6.9858, -1.3324, 0.4006,
    2.1335, 3.8665, 5.5994, 7.3324, -0.9858, 0.7472, 2.4801, 4.2131, 5.9460, 7.6790, -0.6392,
    1.0937, 2.8267, 4.5597, 6.2926, 8.0256, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    -0.6656, -0.0879, 0.4897, 1.0674, 1.6450, 2.2227, -0.5500, 0.0276, 0.6053, 1.1829, 1.7606,
    2.3382, -0.4345, 0.1431, 0.7208, 1.2985, 1.8761, 2.4538, -0.3190, 0.2587, 0.8363, 1.4140,
    1.9916, 2.5693, -0.2034, 0.3742, 0.9519, 1.5295, 2.1072, 2.6848, -1.3311, -0.1758, 0.9795,
    2.1348, 3.2901, 4.4454, -1.1001, 0.0552, 1.2105, 2.3658, 3.5212, 4.6765, -0.8690, 0.2863,
    1.4416, 2.5969, 3.7522, 4.9075, -0.6380, 0.5174, 1.6727, 2.8280, 3.9833, 5.1386, -0.4069,
    0.7484, 1.9037, 3.0590, 4.2143, 5.3696, -1.9967, -0.2637, 1.4692, 3.2022, 4.9351, 6.6681,
    -1.6501, 0.0829, 1.8158, 3.5488, 5.2817, 7.0147, -1.3035, 0.4294, 2.1624, 3.8954, 5.6283,
    7.3613, -0.9569, 0.7760, 2.5090, 4.2420, 5.9749, 7.7079, -0.6103, 1.1226, 2.8556, 4.5885,
    6.3215, 8.0545,
];

static BATCH_NORM2D_FORWARD_PREDICT_EXPECTED_DATA: [f64; 360] = [
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5496, 2.3755, 4.2015, 6.0274, 7.8534, 9.6793,
    0.9148, 2.7407, 4.5667, 6.3926, 8.2185, 10.0445, 1.2800, 3.1059, 4.9319, 6.7578, 8.5837,
    10.4097, 1.6452, 3.4711, 5.2970, 7.1230, 8.9489, 10.7749, 2.0104, 3.8363, 5.6622, 7.4882,
    9.3141, 11.1400, 1.2635, 4.9154, 8.5673, 12.2192, 15.8711, 19.5229, 1.9939, 5.6458, 9.2977,
    12.9495, 16.6014, 20.2533, 2.7243, 6.3762, 10.0280, 13.6799, 17.3318, 20.9837, 3.4547, 7.1065,
    10.7584, 14.4103, 18.0622, 21.7141, 4.1850, 7.8369, 11.4888, 15.1407, 18.7926, 22.4444, 2.1418,
    7.6196, 13.0974, 18.5753, 24.0531, 29.5309, 3.2374, 8.7152, 14.1930, 19.6708, 25.1486, 30.6265,
    4.3329, 9.8108, 15.2886, 20.7664, 26.2442, 31.7220, 5.4285, 10.9063, 16.3841, 21.8620, 27.3398,
    32.8176, 6.5241, 12.0019, 17.4797, 22.9575, 28.4353, 33.9132, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.5800, 2.4060, 4.2319, 6.0579, 7.8838, 9.7097, 0.9452, 2.7712, 4.5971, 6.4230,
    8.2490, 10.0749, 1.3104, 3.1363, 4.9623, 6.7882, 8.6142, 10.4401, 1.6756, 3.5015, 5.3275,
    7.1534, 8.9794, 10.8053, 2.0408, 3.8667, 5.6927, 7.5186, 9.3445, 11.1705, 1.3244, 4.9763,
    8.6282, 12.2800, 15.9319, 19.5838, 2.0548, 5.7067, 9.3585, 13.0104, 16.6623, 20.3142, 2.7852,
    6.4370, 10.0889, 13.7408, 17.3927, 21.0445, 3.5155, 7.1674, 10.8193, 14.4712, 18.1230, 21.7749,
    4.2459, 7.8978, 11.5497, 15.2015, 18.8534, 22.5053, 2.2331, 7.7109, 13.1887, 18.6666, 24.1444,
    29.6222, 3.3287, 8.8065, 14.2843, 19.7621, 25.2399, 30.7178, 4.4242, 9.9021, 15.3799, 20.8577,
    26.3355, 31.8133, 5.5198, 10.9976, 16.4754, 21.9532, 27.4311, 32.9089, 6.6154, 12.0932,
    17.5710, 23.0488, 28.5266, 34.0044, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.6105,
    2.4364, 4.2623, 6.0883, 7.9142, 9.7402, 0.9757, 2.8016, 4.6275, 6.4535, 8.2794, 10.1054,
    1.3408, 3.1668, 4.9927, 6.8187, 8.6446, 10.4705, 1.7060, 3.5320, 5.3579, 7.1838, 9.0098,
    10.8357, 2.0712, 3.8972, 5.7231, 7.5490, 9.3750, 11.2009, 1.3853, 5.0371, 8.6890, 12.3409,
    15.9928, 19.6447, 2.1156, 5.7675, 9.4194, 13.0713, 16.7232, 20.3750, 2.8460, 6.4979, 10.1498,
    13.8017, 17.4535, 21.1054, 3.5764, 7.2283, 10.8802, 14.5320, 18.1839, 21.8358, 4.3068, 7.9586,
    11.6105, 15.2624, 18.9143, 22.5662, 2.3244, 7.8022, 13.2800, 18.7579, 24.2357, 29.7135, 3.4200,
    8.8978, 14.3756, 19.8534, 25.3312, 30.8091, 4.5155, 9.9933, 15.4712, 20.9490, 26.4268, 31.9046,
    5.6111, 11.0889, 16.5667, 22.0445, 27.5224, 33.0002, 6.7067, 12.1845, 17.6623, 23.1401,
    28.6179, 34.0957,
];

static BATCH_NORM2D_BACKWARD_DATA_EXPECTED_DATA: [f64; 360] = [
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -1.2326, -1.2443, -1.2561, -1.2678, -1.2795,
    -1.2913, -1.1791, -1.1908, -1.2026, -1.2143, -1.2260, -1.2378, -1.1256, -1.1373, -1.1491,
    -1.1608, -1.1726, -1.1843, -1.0721, -1.0839, -1.0956, -1.1073, -1.1191, -1.1308, -1.0186,
    -1.0304, -1.0421, -1.0538, -1.0656, -1.0773, -2.4652, -2.4887, -2.5121, -2.5356, -2.5591,
    -2.5826, -2.3582, -2.3817, -2.4052, -2.4286, -2.4521, -2.4756, -2.2512, -2.2747, -2.2982,
    -2.3216, -2.3451, -2.3686, -2.1442, -2.1677, -2.1912, -2.2147, -2.2381, -2.2616, -2.0373,
    -2.0607, -2.0842, -2.1077, -2.1311, -2.1546, -3.6978, -3.7330, -3.7682, -3.8034, -3.8386,
    -3.8738, -3.5373, -3.5725, -3.6077, -3.6429, -3.6781, -3.7133, -3.3768, -3.4120, -3.4472,
    -3.4825, -3.5177, -3.5529, -3.2164, -3.2516, -3.2868, -3.3220, -3.3572, -3.3924, -3.0559,
    -3.0911, -3.1263, -3.1615, -3.1967, -3.2319, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    -0.0776, -0.0894, -0.1011, -0.1129, -0.1246, -0.1363, -0.0242, -0.0359, -0.0476, -0.0594,
    -0.0711, -0.0828, 0.0293, 0.0176, 0.0059, -0.0059, -0.0176, -0.0293, 0.0828, 0.0711, 0.0594,
    0.0476, 0.0359, 0.0242, 0.1363, 0.1246, 0.1129, 0.1011, 0.0894, 0.0776, -0.1553, -0.1788,
    -0.2022, -0.2257, -0.2492, -0.2727, -0.0483, -0.0718, -0.0952, -0.1187, -0.1422, -0.1657,
    0.0587, 0.0352, 0.0117, -0.0117, -0.0352, -0.0587, 0.1657, 0.1422, 0.1187, 0.0952, 0.0718,
    0.0483, 0.2727, 0.2492, 0.2257, 0.2022, 0.1788, 0.1553, -0.2329, -0.2681, -0.3034, -0.3386,
    -0.3738, -0.4090, -0.0725, -0.1077, -0.1429, -0.1781, -0.2133, -0.2485, 0.0880, 0.0528, 0.0176,
    -0.0176, -0.0528, -0.0880, 0.2485, 0.2133, 0.1781, 0.1429, 0.1077, 0.0725, 0.4090, 0.3738,
    0.3386, 0.3034, 0.2681, 0.2329, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0773, 1.0656,
    1.0538, 1.0421, 1.0304, 1.0186, 1.1308, 1.1191, 1.1073, 1.0956, 1.0839, 1.0721, 1.1843, 1.1726,
    1.1608, 1.1491, 1.1373, 1.1256, 1.2378, 1.2260, 1.2143, 1.2026, 1.1908, 1.1791, 1.2913, 1.2795,
    1.2678, 1.2561, 1.2443, 1.2326, 2.1546, 2.1311, 2.1077, 2.0842, 2.0607, 2.0373, 2.2616, 2.2381,
    2.2147, 2.1912, 2.1677, 2.1442, 2.3686, 2.3451, 2.3216, 2.2982, 2.2747, 2.2512, 2.4756, 2.4521,
    2.4286, 2.4052, 2.3817, 2.3582, 2.5826, 2.5591, 2.5356, 2.5121, 2.4887, 2.4652, 3.2319, 3.1967,
    3.1615, 3.1263, 3.0911, 3.0559, 3.3924, 3.3572, 3.3220, 3.2868, 3.2516, 3.2164, 3.5529, 3.5177,
    3.4825, 3.4472, 3.4120, 3.3768, 3.7134, 3.6781, 3.6429, 3.6077, 3.5725, 3.5373, 3.8738, 3.8386,
    3.8034, 3.7682, 3.7330, 3.6978,
];

static BATCH_NORM2D_BACKWARD_GAMMA_EXPECTED_DATA: [f64; 4] =
    [345.7256, 345.7256, 345.7256, 345.7257];
static BATCH_NORM2D_BACKWARD_BETA_EXPECTED_DATA: [f64; 4] = [12105., 14805., 17505., 20205.];

fn test_batch_norm2d<B: Backend>(device: Device<B>) -> Result<()> {
    let n = 3;
    let c = 4;
    let h = 5;
    let w = 6;
    let x = arange_with_shape::<_, f64>(&[w, h, c, n], device);
    let x = x.reversed_axes()?;
    let gamma = arange_with_shape(&[c], device);
    let beta = arange_with_shape(&[c], device);
    let mut batch_norm2d = BatchNorm2d::new(c, 0.9, 1e-7, device);
    batch_norm2d.gamma().copy(&gamma)?;
    batch_norm2d.beta().copy(&beta)?;
    let y = batch_norm2d.forward(&x, true)?;
    let expected = Tensor::from_vec(
        BATCH_NORM2D_FORWARD_EXPECTED_DATA.to_vec(),
        vec![n, c, h, w],
        device,
    )?;
    assert_tensor(&y, &expected);
    Ok(())
}

define_test!(
    test_batch_norm2d,
    test_batch_norm2d_cpu,
    test_batch_norm2d_cuda
);

fn test_batch_norm2d_predict<B: Backend>(device: Device<B>) -> Result<()> {
    let n = 3;
    let c = 4;
    let h = 5;
    let w = 6;
    let x = arange_with_shape::<_, f64>(&[w, h, c, n], device);
    let x = x.reversed_axes()?.requires_grad();
    let gamma = arange_with_shape(&[c], device);
    let beta = arange_with_shape(&[c], device);
    let mut batch_norm2d = BatchNorm2d::new(c, 0.9, 1e-7, device);
    batch_norm2d.gamma().copy(&gamma)?;
    batch_norm2d.beta().copy(&beta)?;
    let y = batch_norm2d.forward(&x, true)?;
    let expected_y = Tensor::from_vec(
        BATCH_NORM2D_FORWARD_EXPECTED_DATA.to_vec(),
        vec![n, c, h, w],
        device,
    )?;
    assert_tensor(&y, &expected_y);

    let x2 = arange_with_shape(&[n, c, h, w], device);
    let y2 = (y * x2)?;

    let grads: rust_dnn_core::gradients::Gradients<B, f64> = y2.backward()?;
    let gx = grads.get(&x).unwrap();
    let ggamma = grads.get(&batch_norm2d.gamma()).unwrap();
    let gbeta = grads.get(&batch_norm2d.beta()).unwrap();

    let expected_gx = Tensor::from_vec(
        BATCH_NORM2D_BACKWARD_DATA_EXPECTED_DATA.to_vec(),
        vec![n, c, h, w],
        device,
    )?;
    assert_tensor(&gx, &expected_gx);

    let expected_ggamma = Tensor::from_vec(
        BATCH_NORM2D_BACKWARD_GAMMA_EXPECTED_DATA.to_vec(),
        vec![c],
        device,
    )?;
    assert_tensor_with_eps(&ggamma, &expected_ggamma, 1e-3);

    let expected_gbeta = Tensor::from_vec(
        BATCH_NORM2D_BACKWARD_BETA_EXPECTED_DATA.to_vec(),
        vec![c],
        device,
    )?;
    assert_tensor(&gbeta, &expected_gbeta);

    let y3 = batch_norm2d.forward(&x, false)?;
    let expected_y3 = Tensor::from_vec(
        BATCH_NORM2D_FORWARD_PREDICT_EXPECTED_DATA.to_vec(),
        vec![n, c, h, w],
        device,
    )?;
    assert_tensor_with_eps(&y3, &expected_y3, 1e-1);

    Ok(())
}

define_test!(
    test_batch_norm2d_predict,
    test_batch_norm2d_predict_cpu,
    test_batch_norm2d_predict_cuda
);

fn test_batch_norm2d_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let n = 3;
    let c = 4;
    let h = 5;
    let w = 6;
    let x = arange_with_shape::<_, f64>(&[w, h, c, n], device);
    let x = x.reversed_axes()?.requires_grad();
    let gamma = arange_with_shape(&[c], device);
    let beta = arange_with_shape(&[c], device);
    let mut batch_norm2d = BatchNorm2d::new(c, 0.9, 1e-7, device);
    batch_norm2d.gamma().copy(&gamma)?;
    batch_norm2d.beta().copy(&beta)?;
    let y = batch_norm2d.forward(&x, true)?;
    let expected_y = Tensor::from_vec(
        BATCH_NORM2D_FORWARD_EXPECTED_DATA.to_vec(),
        vec![n, c, h, w],
        device,
    )?;
    assert_tensor(&y, &expected_y);

    let x2 = arange_with_shape(&[n, c, h, w], device);
    let y2 = (y * x2)?;

    let grads: rust_dnn_core::gradients::Gradients<B, f64> = y2.backward()?;
    let gx = grads.get(&x).unwrap();
    let ggamma = grads.get(&batch_norm2d.gamma()).unwrap();
    let gbeta = grads.get(&batch_norm2d.beta()).unwrap();

    let expected_gx = Tensor::from_vec(
        BATCH_NORM2D_BACKWARD_DATA_EXPECTED_DATA.to_vec(),
        vec![n, c, h, w],
        device,
    )?;
    assert_tensor(&gx, &expected_gx);

    let expected_ggamma = Tensor::from_vec(
        BATCH_NORM2D_BACKWARD_GAMMA_EXPECTED_DATA.to_vec(),
        vec![c],
        device,
    )?;
    assert_tensor_with_eps(&ggamma, &expected_ggamma, 1e-3);

    let expected_gbeta = Tensor::from_vec(
        BATCH_NORM2D_BACKWARD_BETA_EXPECTED_DATA.to_vec(),
        vec![c],
        device,
    )?;
    assert_tensor(&gbeta, &expected_gbeta);

    Ok(())
}

define_test!(
    test_batch_norm2d_backward,
    test_batch_norm2d_backward_cpu,
    test_batch_norm2d_backward_cuda
);
