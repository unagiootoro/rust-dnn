mod test_utils;

use rust_dnn_core::{backend::Backend, device::Device, error::Result, ten, tensor::Tensor};

use crate::test_utils::{arange_with_shape, assert_tensor, assert_tensor_with_eps};

fn test_fill<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::fill(vec![3], 1.0, device);
    assert_eq!(x.to_vec(), vec![1.0, 1.0, 1.0]);
    Ok(())
}

define_test!(test_fill, test_fill_cpu, test_fill_cuda);

fn test_to_vec<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![0.0f32, 1.0, 2.0].to_device(device)?;
    assert_eq!(x.to_vec(), vec![0.0, 1.0, 2.0]);
    Ok(())
}

define_test!(test_to_vec, test_to_vec_cpu, test_to_vec_cuda, test_to_vec_wgpu);

fn test_to_dtype<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![0.0, 1.0, 2.0].to_device(device)?;
    let y = x.to_dtype::<u32>();
    assert_eq!(y.to_vec(), vec![0, 1, 2]);
    Ok(())
}

define_test!(test_to_dtype, test_to_dtype_cpu, test_to_dtype_cuda);

fn test_arange<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::arange(-1..2, device);
    assert_tensor(&x, &ten![-1.0, 0.0, 1.0]);
    Ok(())
}

define_test!(test_arange, test_arange_cpu, test_arange_cuda);

fn test_add<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[-1.0f32, 0.0, 1.0], [2.0, 3.0, 4.0]].to_device(device)?;
    let x2 = ten![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]].to_device(device)?;
    let y = x1 + x2;
    assert_tensor(&y, &ten![[4.0, 6.0, 8.0], [10.0, 12.0, 14.0]]);
    Ok(())
}

define_test!(test_add, test_add_cpu, test_add_cuda, test_add_wgpu);

fn test_add_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]]
        .to_device(device)?
        .requires_grad();
    let x2 = ten![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]
        .to_device(device)?
        .requires_grad();
    let x3 = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let y = (&x1 + &x2) * x3;
    assert_tensor(&y, &ten![[4.0, 12.0, 24.0], [40.0, 60.0, 84.0]]);
    let grads = y.backward();
    let gx1 = grads.get(&x1).unwrap();
    assert_tensor(gx1, &ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let gx2 = grads.get(&x2).unwrap();
    assert_tensor(gx2, &ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    Ok(())
}

define_test!(
    test_add_backward,
    test_add_backward_cpu,
    test_add_backward_cuda
);

fn test_sub<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[-1.0f32, 0.0, 1.0], [2.0, 3.0, 4.0]].to_device(device)?;
    let x2 = ten![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]].to_device(device)?;
    let y = x1 - x2;
    assert_tensor(&y, &ten![[-6.0, -6.0, -6.0], [-6.0, -6.0, -6.0]]);
    Ok(())
}

define_test!(test_sub, test_sub_cpu, test_sub_cuda, test_sub_wgpu);

fn test_sub_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]]
        .to_device(device)?
        .requires_grad();
    let x2 = ten![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]
        .to_device(device)?
        .requires_grad();
    let x3 = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let y = (&x1 - &x2) * x3;
    assert_tensor(&y, &ten![[-6.0, -12.0, -18.0], [-24.0, -30.0, -36.0]]);
    let grads = y.backward();
    let gx1 = grads.get(&x1).unwrap();
    assert_tensor(gx1, &ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let gx2 = grads.get(&x2).unwrap();
    assert_tensor(gx2, &ten![[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]]);
    Ok(())
}

define_test!(
    test_sub_backward,
    test_sub_backward_cpu,
    test_sub_backward_cuda
);

fn test_mul<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[-1.0f32, 0.0, 1.0], [2.0, 3.0, 4.0]].to_device(device)?;
    let x2 = ten![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]].to_device(device)?;
    let y = x1 * x2;
    assert_tensor(&y, &ten![[-5.0, 0.0, 7.0], [16.0, 27.0, 40.0]]);
    Ok(())
}

define_test!(test_mul, test_mul_cpu, test_mul_cuda, test_mul_wgpu);

fn test_mul_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]]
        .to_device(device)?
        .requires_grad();
    let x2 = ten![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]
        .to_device(device)?
        .requires_grad();
    let x3 = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let y = (&x1 * &x2) * x3;
    assert_tensor(&y, &ten![[-5.0, 0.0, 21.0], [64.0, 135.0, 240.0]]);
    let grads = y.backward();
    let gx1 = grads.get(&x1).unwrap();
    assert_tensor(gx1, &ten![[5.0, 12.0, 21.0], [32.0, 45.0, 60.0]]);
    let gx2 = grads.get(&x2).unwrap();
    assert_tensor(gx2, &ten![[-1.0, 0.0, 3.0], [8.0, 15.0, 24.0]]);
    Ok(())
}

define_test!(
    test_mul_backward,
    test_mul_backward_cpu,
    test_mul_backward_cuda
);

fn test_div<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[-1.0f32, 0.0, 1.0], [2.0, 3.0, 4.0]].to_device(device)?;
    let x2 = ten![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]].to_device(device)?;
    let y = x1 / x2;
    assert_tensor(&y, &ten![[-0.2, 0.0, 0.14285715], [0.25, 0.33333334, 0.4]]);
    Ok(())
}

define_test!(test_div, test_div_cpu, test_div_cuda, test_div_wgpu);

fn test_div_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]]
        .to_device(device)?
        .requires_grad();
    let x2 = ten![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]
        .to_device(device)?
        .requires_grad();
    let x3 = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let y = (&x1 / &x2) * x3;
    assert_tensor(&y, &ten![[-0.2, 0.0, 0.42857146], [1.0, 1.6666667, 2.4]]);
    let grads = y.backward();
    let gx1 = grads.get(&x1).unwrap();
    assert_tensor(
        gx1,
        &ten![[0.2, 0.33333334, 0.42857143], [0.5, 0.5555556, 0.6]],
    );
    let gx2 = grads.get(&x2).unwrap();
    assert_tensor(
        gx2,
        &ten![[0.04, -0.0, -0.06122449], [-0.125, -0.1851852, -0.24]],
    );
    Ok(())
}

define_test!(
    test_div_backward,
    test_div_backward_cpu,
    test_div_backward_cuda
);

fn test_neg<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[0.0f32, -2.0, -4.0], [1.0, 2.0, 3.0]].to_device(device)?;
    let y = -x;
    assert_tensor(&y, &ten![[0.0, 2.0, 4.0], [-1.0, -2.0, -3.0]]);
    Ok(())
}

define_test!(test_neg, test_neg_cpu, test_neg_cuda, test_neg_wgpu);

fn test_neg_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[0.0, -2.0, -4.0], [1.0, 2.0, 3.0]]
        .to_device(device)?
        .requires_grad();
    let y = -&x;
    assert_tensor(&y, &ten![[0.0, 2.0, 4.0], [-1.0, -2.0, -3.0]]);
    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(&gx, &ten![[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]);
    Ok(())
}

define_test!(
    test_neg_backward,
    test_neg_backward_cpu,
    test_neg_backward_cuda
);

fn test_eq<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![-1.0f32, 0.0, 1.0].to_device(device)?;
    let x2 = ten![-1.0, 0.0, 2.0].to_device(device)?;
    let y = x1.eq(&x2);
    assert_tensor(&y, &ten![1, 1, 0]);
    Ok(())
}

define_test!(test_eq, test_eq_cpu, test_eq_cuda, test_eq_wgpu);

fn test_lt<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![-1.0f32, 0.0, 1.0].to_device(device)?;
    let x2 = ten![0.0, -1.0, 1.0].to_device(device)?;
    let y = x1.lt(&x2);
    assert_tensor(&y, &ten![1, 0, 0]);
    Ok(())
}

define_test!(test_lt, test_lt_cpu, test_lt_cuda, test_lt_wgpu);

fn test_le<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![-1.0f32, 0.0, 1.0].to_device(device)?;
    let x2 = ten![0.0, -1.0, 1.0].to_device(device)?;
    let y = x1.le(&x2);
    assert_tensor(&y, &ten![1, 0, 1]);
    Ok(())
}

define_test!(test_le, test_le_cpu, test_le_cuda, test_le_wgpu);

fn test_gt<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![0.0f32, -1.0, 1.0].to_device(device)?;
    let x2 = ten![-1.0, 0.0, 1.0].to_device(device)?;
    let y = x1.gt(&x2);
    assert_tensor(&y, &ten![1, 0, 0]);
    Ok(())
}

define_test!(test_gt, test_gt_cpu, test_gt_cuda, test_gt_wgpu);

fn test_ge<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![0.0f32, -1.0, 1.0].to_device(device)?;
    let x2 = ten![-1.0, 0.0, 1.0].to_device(device)?;
    let y = x1.ge(&x2);
    assert_tensor(&y, &ten![1, 0, 1]);
    Ok(())
}

define_test!(test_ge, test_ge_cpu, test_ge_cuda, test_ge_wgpu);

static MATMUL_BATCH_FORWARD_EXPECTED_DATA: [f64; 144] = [
    180., 190., 200., 210., 220., 230., 480., 515., 550., 585., 620., 655., 780., 840., 900., 960.,
    1020., 1080., 1080., 1165., 1250., 1335., 1420., 1505., 4680., 4790., 4900., 5010., 5120.,
    5230., 5730., 5865., 6000., 6135., 6270., 6405., 6780., 6940., 7100., 7260., 7420., 7580.,
    7830., 8015., 8200., 8385., 8570., 8755., 15180., 15390., 15600., 15810., 16020., 16230.,
    16980., 17215., 17450., 17685., 17920., 18155., 18780., 19040., 19300., 19560., 19820., 20080.,
    20580., 20865., 21150., 21435., 21720., 22005., 31680., 31990., 32300., 32610., 32920., 33230.,
    34230., 34565., 34900., 35235., 35570., 35905., 36780., 37140., 37500., 37860., 38220., 38580.,
    39330., 39715., 40100., 40485., 40870., 41255., 54180., 54590., 55000., 55410., 55820., 56230.,
    57480., 57915., 58350., 58785., 59220., 59655., 60780., 61240., 61700., 62160., 62620., 63080.,
    64080., 64565., 65050., 65535., 66020., 66505., 82680., 83190., 83700., 84210., 84720., 85230.,
    86730., 87265., 87800., 88335., 88870., 89405., 90780., 91340., 91900., 92460., 93020., 93580.,
    94830., 95415., 96000., 96585., 97170., 97755.,
];

static MATMUL_BATCH_FORWARD2_EXPECTED_DATA: [f64; 144] = [
    180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 480.0, 515.0, 550.0, 585.0, 620.0, 655.0, 780.0,
    840.0, 900.0, 960.0, 1020.0, 1080.0, 1080.0, 1165.0, 1250.0, 1335.0, 1420.0, 1505.0, 1380.0,
    1490.0, 1600.0, 1710.0, 1820.0, 1930.0, 1680.0, 1815.0, 1950.0, 2085.0, 2220.0, 2355.0, 1980.0,
    2140.0, 2300.0, 2460.0, 2620.0, 2780.0, 2280.0, 2465.0, 2650.0, 2835.0, 3020.0, 3205.0, 2580.0,
    2790.0, 3000.0, 3210.0, 3420.0, 3630.0, 2880.0, 3115.0, 3350.0, 3585.0, 3820.0, 4055.0, 3180.0,
    3440.0, 3700.0, 3960.0, 4220.0, 4480.0, 3480.0, 3765.0, 4050.0, 4335.0, 4620.0, 4905.0, 3780.0,
    4090.0, 4400.0, 4710.0, 5020.0, 5330.0, 4080.0, 4415.0, 4750.0, 5085.0, 5420.0, 5755.0, 4380.0,
    4740.0, 5100.0, 5460.0, 5820.0, 6180.0, 4680.0, 5065.0, 5450.0, 5835.0, 6220.0, 6605.0, 4980.0,
    5390.0, 5800.0, 6210.0, 6620.0, 7030.0, 5280.0, 5715.0, 6150.0, 6585.0, 7020.0, 7455.0, 5580.0,
    6040.0, 6500.0, 6960.0, 7420.0, 7880.0, 5880.0, 6365.0, 6850.0, 7335.0, 7820.0, 8305.0, 6180.0,
    6690.0, 7200.0, 7710.0, 8220.0, 8730.0, 6480.0, 7015.0, 7550.0, 8085.0, 8620.0, 9155.0, 6780.0,
    7340.0, 7900.0, 8460.0, 9020.0, 9580.0, 7080.0, 7665.0, 8250.0, 8835.0, 9420.0, 10005.0,
];

static MATMUL_BATCH_BACKWARD_X1_EXPECTED_DATA: [f64; 120] = [
    15., 51., 87., 123., 159., 15., 51., 87., 123., 159., 15., 51., 87., 123., 159., 15., 51., 87.,
    123., 159., 195., 231., 267., 303., 339., 195., 231., 267., 303., 339., 195., 231., 267., 303.,
    339., 195., 231., 267., 303., 339., 375., 411., 447., 483., 519., 375., 411., 447., 483., 519.,
    375., 411., 447., 483., 519., 375., 411., 447., 483., 519., 555., 591., 627., 663., 699., 555.,
    591., 627., 663., 699., 555., 591., 627., 663., 699., 555., 591., 627., 663., 699., 735., 771.,
    807., 843., 879., 735., 771., 807., 843., 879., 735., 771., 807., 843., 879., 735., 771., 807.,
    843., 879., 915., 951., 987., 1023., 1059., 915., 951., 987., 1023., 1059., 915., 951., 987.,
    1023., 1059., 915., 951., 987., 1023., 1059.,
];

static MATMUL_BATCH_BACKWARD_X2_EXPECTED_DATA: [f64; 180] = [
    30., 30., 30., 30., 30., 30., 34., 34., 34., 34., 34., 34., 38., 38., 38., 38., 38., 38., 42.,
    42., 42., 42., 42., 42., 46., 46., 46., 46., 46., 46., 110., 110., 110., 110., 110., 110.,
    114., 114., 114., 114., 114., 114., 118., 118., 118., 118., 118., 118., 122., 122., 122., 122.,
    122., 122., 126., 126., 126., 126., 126., 126., 190., 190., 190., 190., 190., 190., 194., 194.,
    194., 194., 194., 194., 198., 198., 198., 198., 198., 198., 202., 202., 202., 202., 202., 202.,
    206., 206., 206., 206., 206., 206., 270., 270., 270., 270., 270., 270., 274., 274., 274., 274.,
    274., 274., 278., 278., 278., 278., 278., 278., 282., 282., 282., 282., 282., 282., 286., 286.,
    286., 286., 286., 286., 350., 350., 350., 350., 350., 350., 354., 354., 354., 354., 354., 354.,
    358., 358., 358., 358., 358., 358., 362., 362., 362., 362., 362., 362., 366., 366., 366., 366.,
    366., 366., 430., 430., 430., 430., 430., 430., 434., 434., 434., 434., 434., 434., 438., 438.,
    438., 438., 438., 438., 442., 442., 442., 442., 442., 442., 446., 446., 446., 446., 446., 446.,
];

fn test_matmul<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[0.0, -2.0, -4.0], [1.0, 2.0, 3.0]]
        .to_device(device)?
        .requires_grad();
    let x2 = ten![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let y = x1.matmul(&x2);
    assert_tensor(&y, &ten![[-26.0, -32.0], [22.0, 28.0]]);
    Ok(())
}

define_test!(test_matmul, test_matmul_cpu, test_matmul_cuda);

fn test_matmul_batch<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = Tensor::<_, f64>::arange(0..(2 * 3 * 4 * 5), device).reshape(vec![2, 3, 4, 5]);
    let x2 = Tensor::<_, f64>::arange(0..(2 * 3 * 5 * 6), device).reshape(vec![2, 3, 5, 6]);
    let y = x1.matmul(&x2);
    assert_eq!(y.shape(), &vec![2, 3, 4, 6]);
    assert_eq!(y.to_vec(), MATMUL_BATCH_FORWARD_EXPECTED_DATA);
    Ok(())
}

define_test!(
    test_matmul_batch,
    test_matmul_batch_cpu,
    test_matmul_batch_cuda
);

fn test_matmul_batch2<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = Tensor::<_, f64>::arange(0..(2 * 3 * 4 * 5), device).reshape(vec![2, 3, 4, 5]);
    let x2 = Tensor::<_, f64>::arange(0..(5 * 6), device).reshape(vec![5, 6]);
    let y = x1.matmul(&x2);
    assert_eq!(y.shape(), &vec![2, 3, 4, 6]);
    assert_eq!(y.to_vec(), MATMUL_BATCH_FORWARD2_EXPECTED_DATA);
    Ok(())
}

define_test!(
    test_matmul_batch2,
    test_matmul_batch2_cpu,
    test_matmul_batch2_cuda
);

fn test_matmul_transpose<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[0.0, -2.0], [-4.0, 1.0], [2.0, 3.0]]
        .to_device(device)?
        .requires_grad();
    let x2 = ten![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let y = x1.reversed_axes().matmul(&x2);
    assert_tensor(&y, &ten![[-2.0, -4.0], [16.0, 18.0]]);
    Ok(())
}

define_test!(
    test_matmul_transpose,
    test_matmul_transpose_cpu,
    test_matmul_transpose_cuda
);

fn test_matmul_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[0.0, -2.0, -4.0], [1.0, 2.0, 3.0]]
        .to_device(device)?
        .requires_grad();
    let x2 = ten![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let y = x1.matmul(&x2) * ten![[1.0, 2.0], [3.0, 4.0]].to_device(device)?;
    assert_tensor(&y, &ten![[-26.0, -64.0], [66.0, 112.0]]);
    let grads = y.backward();
    let gx1 = grads.get(&x1).unwrap();
    let gx2 = grads.get(&x2).unwrap();
    assert_tensor(&gx1, &ten![[5.0, 11.0, 17.0], [11.0, 25.0, 39.0]]);
    assert_tensor(&gx2, &ten![[3.0, 4.0], [4.0, 4.0], [5.0, 4.0]]);
    Ok(())
}

define_test!(
    test_matmul_backward,
    test_matmul_backward_cpu,
    test_matmul_backward_cuda
);

fn test_matmul_batch_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = Tensor::<_, f64>::arange(0..(2 * 3 * 4 * 5), device).reshape(vec![2, 3, 4, 5]);
    let x2 = Tensor::<_, f64>::arange(0..(2 * 3 * 5 * 6), device).reshape(vec![2, 3, 5, 6]);
    let y = x1.matmul(&x2);
    assert_eq!(y.shape(), &vec![2, 3, 4, 6]);
    assert_eq!(y.to_vec(), MATMUL_BATCH_FORWARD_EXPECTED_DATA);
    let grads = y.backward();
    let gx1 = grads.get(&x1).unwrap();
    let gx2 = grads.get(&x2).unwrap();
    assert_eq!(gx1.to_vec(), MATMUL_BATCH_BACKWARD_X1_EXPECTED_DATA);
    assert_eq!(gx2.to_vec(), MATMUL_BATCH_BACKWARD_X2_EXPECTED_DATA);
    Ok(())
}

define_test!(
    test_matmul_batch_backward,
    test_matmul_batch_backward_cpu,
    test_matmul_batch_backward_cuda
);

fn test_matmul_transpose_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[0.0, -2.0], [-4.0, 1.0], [2.0, 3.0]]
        .to_device(device)?
        .requires_grad();
    let x2 = ten![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let y = x1.reversed_axes().matmul(&x2);
    assert_tensor(&y, &ten![[-2.0, -4.0], [16.0, 18.0]]);
    let grads = y.backward();
    let gx1 = grads.get(&x1).unwrap();
    let gx2 = grads.get(&x2).unwrap();
    assert_tensor(&gx1, &ten![[3.0, 3.0], [7.0, 7.0], [11.0, 11.0]]);
    assert_tensor(&gx2, &ten![[-2.0, -2.0], [-3.0, -3.0], [5.0, 5.0]]);
    Ok(())
}

define_test!(
    test_matmul_transpose_backward,
    test_matmul_transpose_backward_cpu,
    test_matmul_transpose_backward_cuda
);

fn test_broadcast_op2<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![-1.0f32].to_device(device)?;
    let x2 = ten![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]].to_device(device)?;
    let y = x1 + x2;
    assert_tensor(&y, &ten![[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    Ok(())
}

define_test!(
    test_broadcast_op2,
    test_broadcast_op2_cpu,
    test_broadcast_op2_cuda,
    test_broadcast_op2_wgpu
);

fn test_broadcast_op2_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![-1.0].to_device(device)?.requires_grad();
    let x2 = ten![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]
        .to_device(device)?
        .requires_grad();
    let x3 = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let y = (&x1 + &x2) * x3;
    assert_tensor(&y, &ten![[4.0, 10.0, 18.0], [28.0, 40.0, 54.0]]);
    let grads = y.backward();
    let gx1 = grads.get(&x1).unwrap();
    assert_tensor(gx1, &ten![21.0]);
    let gx2 = grads.get(&x2).unwrap();
    assert_tensor(gx2, &ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    Ok(())
}

define_test!(
    test_broadcast_op2_backward,
    test_broadcast_op2_backward_cpu,
    test_broadcast_op2_backward_cuda
);

fn test_pow<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[0.0f32, -2.0, -4.0], [1.0, 2.0, 3.0]].to_device(device)?;
    let x2 = ten![3.0].to_device(device)?;
    let y = x1.pow(&x2);
    assert_tensor(&y, &ten![[0.0, -8.0, -64.0], [1.0, 8.0, 27.0]]);
    Ok(())
}

define_test!(test_pow, test_pow_cpu, test_pow_cuda, test_pow_wgpu);

fn test_pow_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let x2 = ten![3.0].to_device(device)?.requires_grad();
    let y = x1.pow(&x2);
    assert_tensor(&y, &ten![[1.0, 8.0, 27.0], [64.0, 125.0, 216.0]]);
    let grads = y.backward();
    let gx1 = grads.get(&x1).unwrap();
    assert_tensor(&gx1, &ten![[3.0, 12.0, 27.0], [48.0, 75.0, 108.0]]);
    let gx2 = grads.get(&x2).unwrap();
    assert_tensor(&gx2, &ten![6.5792512120101]);
    Ok(())
}

define_test!(
    test_pow_backward,
    test_pow_backward_cpu,
    test_pow_backward_cuda
);

fn test_pow_scalar<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[0.0, -2.0, -4.0], [1.0, 2.0, 3.0]].to_device(device)?;
    let y = x.pow_scalar(3.0);
    assert_tensor(&y, &ten![[0.0, -8.0, -64.0], [1.0, 8.0, 27.0]]);
    Ok(())
}

define_test!(test_pow_scalar, test_pow_scalar_cpu, test_pow_scalar_cuda);

fn test_pow_scalar_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[0.0, -2.0, -4.0], [1.0, 2.0, 3.0]]
        .to_device(device)?
        .requires_grad();
    let y = x.pow_scalar(3.0);
    assert_tensor(&y, &ten![[0.0, -8.0, -64.0], [1.0, 8.0, 27.0]]);
    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(&gx, &ten![[0.0, 12.0, 48.0], [3.0, 12.0, 27.0]]);
    Ok(())
}

define_test!(
    test_pow_scalar_backward,
    test_pow_scalar_backward_cpu,
    test_pow_scalar_backward_cuda
);

fn test_exp<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![2.0f32].to_device(device)?;
    let y = x.exp();
    assert_tensor(&y, &ten![7.38905609893065]);
    Ok(())
}

define_test!(test_exp, test_exp_cpu, test_exp_cuda, test_exp_wgpu);

fn test_exp_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![2.0].to_device(device)?.requires_grad();
    let y = x.exp() * ten![2.0].to_device(device)?;
    assert_tensor(&y, &ten![14.7781121978613]);
    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(&gx, &ten![14.7781121978613]);
    Ok(())
}

define_test!(
    test_exp_backward,
    test_exp_backward_cpu,
    test_exp_backward_cuda
);

fn test_sqrt<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![4.0].to_device(device)?;
    let y = x.sqrt();
    assert_tensor(&y, &ten![2.0]);
    Ok(())
}

define_test!(test_sqrt, test_sqrt_cpu, test_sqrt_cuda);

fn test_sqrt_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![4.0].to_device(device)?.requires_grad();
    let y = x.sqrt() * ten![2.0].to_device(device)?;
    assert_tensor(&y, &ten![4.0]);
    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(&gx, &ten![0.5]);
    Ok(())
}

define_test!(
    test_sqrt_backward,
    test_sqrt_backward_cpu,
    test_sqrt_backward_cuda
);

fn test_ln<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![4.0].to_device(device)?;
    let y = x.ln();
    assert_tensor(&y, &ten![1.3862944]);
    Ok(())
}

define_test!(test_ln, test_ln_cpu, test_ln_cuda);

fn test_ln_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![4.0].to_device(device)?.requires_grad();
    let y = x.ln() * ten![2.0].to_device(device)?;
    assert_tensor(&y, &ten![2.772588722239781]);
    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(&gx, &ten![0.5]);
    Ok(())
}

define_test!(
    test_ln_backward,
    test_ln_backward_cpu,
    test_ln_backward_cuda
);

fn test_sin<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![0.5].to_device(device)?;
    let y = x.sin();
    assert_tensor(&y, &ten![0.479425538604203]);
    Ok(())
}

define_test!(test_sin, test_sin_cpu, test_sin_cuda);

fn test_sin_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![0.5].to_device(device)?.requires_grad();
    let y = x.sin() * ten![2.0].to_device(device)?;
    assert_tensor(&y, &ten![0.958851077208406]);
    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(&gx, &ten![1.7551651237807455]);
    Ok(())
}

define_test!(
    test_sin_backward,
    test_sin_backward_cpu,
    test_sin_backward_cuda
);

fn test_cos<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![0.5].to_device(device)?;
    let y = x.cos();
    assert_tensor(&y, &ten![0.8775825618903728]);
    Ok(())
}

define_test!(test_cos, test_cos_cpu, test_cos_cuda);

fn test_cos_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![0.5].to_device(device)?.requires_grad();
    let y = x.cos() * ten![2.0].to_device(device)?;
    assert_tensor(&y, &ten![1.7551651237807455]);
    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(&gx, &ten![-0.958851077208406]);
    Ok(())
}

define_test!(
    test_cos_backward,
    test_cos_backward_cpu,
    test_cos_backward_cuda
);

fn test_tanh<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![0.5].to_device(device)?;
    let y = x.tanh();
    assert_tensor(&y, &ten![0.46211715726000974]);
    Ok(())
}

define_test!(test_tanh, test_tanh_cpu, test_tanh_cuda);

fn test_tanh_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![0.5].to_device(device)?.requires_grad();
    let y = x.tanh() * ten![2.0].to_device(device)?;
    assert_tensor(&y, &ten![0.9242343145200195]);
    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(&gx, &ten![1.5728954659318548]);
    Ok(())
}

define_test!(
    test_tanh_backward,
    test_tanh_backward_cpu,
    test_tanh_backward_cuda
);

fn test_tril<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::fill(vec![3, 3], 2.0, device);
    let y = x.tril();
    #[rustfmt::skip]
    let y_expected = ten![
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [2.0, 2.0, 2.0],
    ].to_device(device)?;
    assert_tensor(&y, &y_expected);
    Ok(())
}

define_test!(test_tril, test_tril_cpu, test_tril_cuda);

fn test_tril2<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::fill(vec![4, 3], 2.0, device);
    let y = x.tril();
    #[rustfmt::skip]
    let y_expected = ten![
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [2.0, 2.0, 2.0],
        [2.0, 2.0, 2.0],
    ].to_device(device)?;
    assert_tensor(&y, &y_expected);
    Ok(())
}

define_test!(test_tril2, test_tril2_cpu, test_tril2_cuda);

fn test_tril3<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::fill(vec![3, 4], 2.0, device);
    let y = x.tril();
    #[rustfmt::skip]
    let y_expected = ten![
        [2.0, 0.0, 0.0, 0.0],
        [2.0, 2.0, 0.0, 0.0],
        [2.0, 2.0, 2.0, 0.0],
    ].to_device(device)?;
    assert_tensor(&y, &y_expected);
    Ok(())
}

define_test!(test_tril3, test_tril3_cpu, test_tril3_cuda);

fn test_gather<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ]
    .to_device(device)?;
    let index = ten![[0, 1], [2, 3], [1, 3]].to_device(device)?;
    let y = x.gather(&index, 1);
    assert_tensor(&y, &ten![[1.0, 2.0], [7.0, 8.0], [10.0, 12.0]]);
    Ok(())
}

define_test!(test_gather, test_gather_cpu, test_gather_cuda);

fn test_gather_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ]
    .to_device(device)?
    .requires_grad();
    let index = ten![[0, 1], [2, 3], [1, 3]].to_device(device)?;
    let y = x.gather(&index, 1) * 2.0;
    assert_tensor(&y, &ten![[2.0, 4.0], [14.0, 16.0], [20.0, 24.0]]);
    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(
        &gx,
        &ten![
            [2.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 2.0],
            [0.0, 2.0, 0.0, 2.0]
        ],
    );
    Ok(())
}

define_test!(
    test_gather_backward,
    test_gather_backward_cpu,
    test_gather_backward_cuda
);

fn test_scatter<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ]
    .to_device(device)?;
    let index = ten![[0, 1], [2, 3], [1, 3]].to_device(device)?;
    let src = ten![[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]].to_device(device)?;
    x.scatter(&index, &src, 1);
    assert_tensor(
        &x,
        &ten![
            [10.0, 20.0, 3.0, 4.0],
            [5.0, 6.0, 30.0, 40.0],
            [9.0, 50.0, 11.0, 60.0]
        ],
    );
    Ok(())
}

define_test!(test_scatter, test_scatter_cpu, test_scatter_cuda);

fn test_scatter_add<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ]
    .to_device(device)?;
    let index = ten![[0, 1], [2, 3], [1, 3]].to_device(device)?;
    let src = ten![[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]].to_device(device)?;
    x.scatter_add(&index, &src, 1);
    assert_tensor(
        &x,
        &ten![
            [11.0, 22.0, 3.0, 4.0],
            [5.0, 6.0, 37.0, 48.0],
            [9.0, 60.0, 11.0, 72.0]
        ],
    );
    Ok(())
}

define_test!(
    test_scatter_add,
    test_scatter_add_cpu,
    test_scatter_add_cuda
);

fn test_index_select<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ]
    .to_device(device)?;
    let index = ten![1, 3].to_device(device)?;
    let y = x.index_select(1, &index);
    assert_tensor(&y, &ten![[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]);
    Ok(())
}

define_test!(
    test_index_select,
    test_index_select_cpu,
    test_index_select_cuda
);

fn test_index_select_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ]
    .to_device(device)?
    .requires_grad();
    let index = ten![1, 3].to_device(device)?;
    let y = x.index_select(1, &index) * ten![2.0].to_device(device)?;
    assert_tensor(&y, &ten![[4.0, 8.0], [12.0, 16.0], [20.0, 24.0]]);
    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(
        &gx,
        &ten![
            [0.0, 2.0, 0.0, 2.0],
            [0.0, 2.0, 0.0, 2.0],
            [0.0, 2.0, 0.0, 2.0]
        ],
    );
    Ok(())
}

define_test!(
    test_index_select_backward,
    test_index_select_backward_cpu,
    test_index_select_backward_cuda
);

fn test_index_copy<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ]
    .to_device(device)?;
    let index = ten![1, 3].to_device(device)?;
    let src = ten![[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]].to_device(device)?;
    x.index_copy(1, &index, &src);
    assert_tensor(
        &x,
        &ten![
            [1.0, 10.0, 3.0, 20.0],
            [5.0, 30.0, 7.0, 40.0],
            [9.0, 50.0, 11.0, 60.0]
        ],
    );
    Ok(())
}

define_test!(test_index_copy, test_index_copy_cpu, test_index_copy_cuda);

fn test_index_add<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ]
    .to_device(device)?;
    let index = ten![1, 3].to_device(device)?;
    let src = ten![[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]].to_device(device)?;
    x.index_add(1, &index, &src);
    assert_tensor(
        &x,
        &ten![
            [1.0, 12.0, 3.0, 24.0],
            [5.0, 36.0, 7.0, 48.0],
            [9.0, 60.0, 11.0, 72.0]
        ],
    );
    Ok(())
}

define_test!(test_index_add, test_index_add_cpu, test_index_add_cuda);

fn test_reshape<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?;
    let y = x.reshape(vec![6]);
    assert_tensor(&y, &ten![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    Ok(())
}

define_test!(test_reshape, test_reshape_cpu, test_reshape_cuda);

fn test_reshape2<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![1.0, 2.0, 3.0].to_device(device)?;
    let x = x.broadcast_to(vec![2, 3]);
    let y = x.reshape(vec![6]);
    assert_tensor(&y, &ten![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    Ok(())
}

define_test!(test_reshape2, test_reshape2_cpu, test_reshape2_cuda);

fn test_reshape_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let x2 = ten![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].to_device(device)?;
    let y = x1.reshape(vec![6]) * x2;
    assert_tensor(&y, &ten![1.0, 4.0, 9.0, 16.0, 25.0, 36.0]);
    let grads = y.backward();
    let gx = grads.get(&x1).unwrap();
    assert_tensor(gx, &ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    Ok(())
}

define_test!(
    test_reshape_backward,
    test_reshape_backward_cpu,
    test_reshape_backward_cuda
);

fn test_reshape_backward2<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![1.0, 2.0, 3.0].to_device(device)?.requires_grad();
    let x2 = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?;
    let y = x1.broadcast_to(vec![2, 3]) * x2;
    assert_tensor(&y, &ten![[1.0, 4.0, 9.0], [4.0, 10.0, 18.0]]);
    let grads = y.backward();
    let gx1 = grads.get(&x1).unwrap();
    assert_tensor(gx1, &ten![5.0, 7.0, 9.0]);
    Ok(())
}

define_test!(
    test_reshape_backward2,
    test_reshape_backward2_cpu,
    test_reshape_backward2_cuda
);

fn test_squeeze<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]]].to_device(device)?;
    let y = x.squeeze();
    assert_tensor(&y, &ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    Ok(())
}

define_test!(test_squeeze, test_squeeze_cpu, test_squeeze_cuda);

fn test_squeeze_axes<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]]].to_device(device)?;
    let y = x.squeeze_axes(&[0, 3]);
    assert_tensor(&y, &ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    Ok(())
}

define_test!(
    test_squeeze_axes,
    test_squeeze_axes_cpu,
    test_squeeze_axes_cuda
);

fn test_unsqueeze<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?;
    let y = x.unsqueeze(2);
    assert_tensor(&y, &ten![[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]]);
    Ok(())
}

define_test!(test_unsqueeze, test_unsqueeze_cpu, test_unsqueeze_cuda);

fn test_permuted_axes<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?;
    let y = x.permuted_axes(&[1, 0]);
    assert_tensor(&y, &ten![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
    Ok(())
}

define_test!(
    test_permuted_axes,
    test_permuted_axes_cpu,
    test_permuted_axes_cuda
);

fn test_permuted_axes_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let y = x.permuted_axes(&[1, 0]);
    assert_tensor(&y, &ten![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(gx, &ten![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);
    Ok(())
}

define_test!(
    test_permuted_axes_backward,
    test_permuted_axes_backward_cpu,
    test_permuted_axes_backward_cuda
);

fn test_reversed_axes<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?;
    let y = x.reversed_axes();
    assert_tensor(&y, &ten![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
    Ok(())
}

define_test!(
    test_reversed_axes,
    test_reversed_axes_cpu,
    test_reversed_axes_cuda
);

fn test_cat<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[1.0, 2.0]].to_device(device)?;
    let x2 = ten![[3.0, 4.0]].to_device(device)?;
    let x3 = ten![[5.0, 6.0]].to_device(device)?;
    let y = Tensor::cat(&vec![x1, x2, x3], 0);
    assert_tensor(&y, &ten![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    Ok(())
}

define_test!(test_cat, test_cat_cpu, test_cat_cuda);

fn test_cat_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[1.0, 2.0]].to_device(device)?.requires_grad();
    let x2 = ten![[3.0, 4.0]].to_device(device)?.requires_grad();
    let x3 = ten![[5.0, 6.0]].to_device(device)?.requires_grad();
    let y = Tensor::cat(&vec![x1.clone(), x2.clone(), x3.clone()], 0);
    let y = y * ten![[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]].to_device(device)?;
    assert_tensor(&y, &ten![[10.0, 40.0], [90.0, 160.0], [250.0, 360.0]]);
    let grads = y.backward();
    let gx1 = grads.get(&x1).unwrap();
    let gx2 = grads.get(&x2).unwrap();
    let gx3 = grads.get(&x3).unwrap();
    assert_tensor(&gx1, &ten![[10.0, 20.0]]);
    assert_tensor(&gx2, &ten![[30.0, 40.0]]);
    assert_tensor(&gx3, &ten![[50.0, 60.0]]);
    Ok(())
}

define_test!(
    test_cat_backward,
    test_cat_backward_cpu,
    test_cat_backward_cuda
);

fn test_stack<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![1.0, 2.0].to_device(device)?;
    let x2 = ten![3.0, 4.0].to_device(device)?;
    let x3 = ten![5.0, 6.0].to_device(device)?;
    let y = Tensor::stack(&vec![x1, x2, x3], 0);
    assert_tensor(&y, &ten![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    Ok(())
}

define_test!(test_stack, test_stack_cpu, test_stack_cuda);

fn test_split<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let ys = x.split(1, &vec![2, 1]);
    assert_tensor(&ys[0], &ten![[1.0, 2.0], [4.0, 5.0]]);
    assert_tensor(&ys[1], &ten![[3.0], [6.0]]);
    Ok(())
}

define_test!(test_split, test_split_cpu, test_split_cuda);

fn test_repeat_interleave<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![1.0, 2.0, 3.0].to_device(device)?.requires_grad();
    let y = x.repeat_interleave(0, 2);
    assert_tensor(&y, &ten![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    Ok(())
}

define_test!(
    test_repeat_interleave,
    test_repeat_interleave_cpu,
    test_repeat_interleave_cuda
);

fn test_repeat_interleave2<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let y = x.repeat_interleave(0, 2);
    assert_tensor(
        &y,
        &ten![
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0]
        ],
    );
    Ok(())
}

define_test!(
    test_repeat_interleave2,
    test_repeat_interleave2_cpu,
    test_repeat_interleave2_cuda
);

fn test_get_item<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ]
    .to_device(device)?;
    let y = x.get_item(vec![(1, 3), (1, 4)]);
    assert_tensor(&y, &ten![[6.0, 7.0, 8.0], [10.0, 11.0, 12.0]]);
    Ok(())
}

define_test!(test_get_item, test_get_item_cpu, test_get_item_cuda);

fn test_get_item2<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        [
            [13.0, 14.0, 15.0, 16.0],
            [17.0, 18.0, 19.0, 20.0],
            [21.0, 22.0, 23.0, 24.0],
        ],
    ]
    .to_device(device)?;
    let h = x.get_item(vec![(1, 2), (0, 3), (0, 4)]);
    let y = h.get_item(vec![(0, 1), (1, 3), (1, 4)]);
    assert_tensor(&y, &ten![[[18.0, 19.0, 20.0], [22.0, 23.0, 24.0]]]);
    Ok(())
}

define_test!(test_get_item2, test_get_item2_cpu, test_get_item2_cuda);

fn test_get_item3<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ]
    .to_device(device)?;
    let h = x.get_item(vec![(1, 3), (1, 4)]);
    let y = -h;
    assert_tensor(&y, &ten![[-6.0, -7.0, -8.0], [-10.0, -11.0, -12.0]]);
    Ok(())
}

define_test!(test_get_item3, test_get_item3_cpu, test_get_item3_cuda);

fn test_get_item_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ]
    .to_device(device)?
    .requires_grad();
    let h = x.get_item(vec![(1, 3), (1, 4)]);
    let y = h * ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?;
    assert_tensor(&y, &ten![[6.0, 14.0, 24.0], [40.0, 55.0, 72.0]]);
    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(
        &gx,
        &ten![
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 4.0, 5.0, 6.0],
        ],
    );
    Ok(())
}

define_test!(
    test_get_item_backward,
    test_get_item_backward_cpu,
    test_get_item_backward_cuda
);

fn test_set_item<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::zeros(vec![3, 4], device);
    x.set_item(
        &vec![(1, 3), (1, 4)],
        &ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?,
    );
    assert_tensor(
        &x,
        &ten![
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 4.0, 5.0, 6.0]
        ],
    );
    Ok(())
}

define_test!(test_set_item, test_set_item_cpu, test_set_item_cuda);

fn test_select<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ]
    .to_device(device)?;
    let y = x.select(1, 2);
    assert_tensor(&y, &ten![3.0, 7.0, 11.0]);
    Ok(())
}

define_test!(test_select, test_select_cpu, test_select_cuda);

fn test_narrow<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ]
    .to_device(device)?;
    let y = x.narrow(1, 1, 2);
    assert_tensor(&y, &ten![[2.0, 3.0], [6.0, 7.0], [10.0, 11.0]]);
    Ok(())
}

define_test!(test_narrow, test_narrow_cpu, test_narrow_cuda);

fn test_copy<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::zeros(vec![2, 3], device);
    x.copy(&ten![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]].to_device(device)?);
    assert_tensor(&x, &ten![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
    Ok(())
}

define_test!(test_copy, test_copy_cpu, test_copy_cuda);

fn test_copy2<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::zeros(vec![2, 3], device);
    x.copy(&ten![1.0, 2.0, 3.0].to_device(device)?);
    assert_tensor(&x, &ten![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
    Ok(())
}

define_test!(test_copy2, test_copy2_cpu, test_copy2_cuda);

fn test_copy3<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::zeros(vec![3, 4], device);
    let x2 = x.get_item(vec![(1, 3), (1, 4)]);
    x2.copy(&ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?);
    assert_tensor(
        &x,
        &ten![
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 4.0, 5.0, 6.0]
        ],
    );
    Ok(())
}

define_test!(test_copy3, test_copy3_cpu, test_copy3_cuda);

fn test_broadcast_to<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![1.0, 2.0, 3.0].to_device(device)?;
    let y = x.broadcast_to(vec![2, 3]);
    assert_tensor(&y, &ten![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
    Ok(())
}

define_test!(
    test_broadcast_to,
    test_broadcast_to_cpu,
    test_broadcast_to_cuda
);

fn test_broadcast_to2<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?;
    let y = x.broadcast_to(vec![2, 3]);
    assert_tensor(&y, &ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    Ok(())
}

define_test!(
    test_broadcast_to2,
    test_broadcast_to2_cpu,
    test_broadcast_to2_cuda
);

fn test_broadcast_to3<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![1.0, 2.0, 3.0].to_device(device)?;
    let x = x.broadcast_to(vec![2, 3]);
    let y = x.broadcast_to(vec![2, 3]);
    assert_tensor(&y, &ten![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
    Ok(())
}

define_test!(
    test_broadcast_to3,
    test_broadcast_to3_cpu,
    test_broadcast_to3_cuda
);

fn test_broadcast_to4<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0], [2.0], [3.0]].to_device(device)?;
    let y = x.broadcast_to(vec![3, 2]);
    assert_tensor(&y, &ten![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]);
    Ok(())
}

define_test!(
    test_broadcast_to4,
    test_broadcast_to4_cpu,
    test_broadcast_to4_cuda
);

fn test_broadcast_to5<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0], [2.0], [3.0]].to_device(device)?;
    let y = x.broadcast_to(vec![2, 3, 4]);
    assert_tensor(
        &y,
        &ten![
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0]
            ],
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0]
            ]
        ],
    );
    Ok(())
}

define_test!(
    test_broadcast_to5,
    test_broadcast_to5_cpu,
    test_broadcast_to5_cuda
);

fn test_broadcast_to_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![1.0, 2.0, 3.0].to_device(device)?.requires_grad();
    let x2 = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?;
    let y = x1.broadcast_to(vec![2, 3]) * x2;
    assert_tensor(&y, &ten![[1.0, 4.0, 9.0], [4.0, 10.0, 18.0]]);
    let grads = y.backward();
    let gx = grads.get(&x1).unwrap();
    assert_tensor(gx, &ten![5.0, 7.0, 9.0]);
    Ok(())
}

define_test!(
    test_broadcast_to_backward,
    test_broadcast_to_backward_cpu,
    test_broadcast_to_backward_cuda
);

fn test_masked_fill<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::fill(vec![4], 1.0, device);
    let mask = Tensor::<B, u32>::from_vec(vec![0, 1, 0, 1], vec![4], device);
    let y = x.masked_fill(&mask, 2.0);
    assert_tensor(&y, &ten![1.0, 2.0, 1.0, 2.0].to_device(device)?);
    Ok(())
}

define_test!(
    test_masked_fill,
    test_masked_fill_cpu,
    test_masked_fill_cuda
);

fn test_masked_fill_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::fill(vec![4], 1.0, device).requires_grad();
    let mask = Tensor::<B, u32>::from_vec(vec![0, 1, 0, 1], vec![4], device);
    let y = x.masked_fill(&mask, 2.0);
    assert_tensor(&y, &ten![1.0, 2.0, 1.0, 2.0].to_device(device)?);

    let y2 = y * ten![3.0].to_device(device)?;
    let grads = y2.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(&gx, &ten![3.0, 0.0, 3.0, 0.0]);
    Ok(())
}

define_test!(
    test_masked_fill_backward,
    test_masked_fill_backward_cpu,
    test_masked_fill_backward_cuda
);

fn test_sum_to<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?;
    let y = x.sum_to(&vec![1, 3]);
    assert_tensor(&y, &ten![[5.0, 7.0, 9.0]]);
    Ok(())
}

define_test!(test_sum_to, test_sum_to_cpu, test_sum_to_cuda);

fn test_sum_to2<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?;
    let y = x.sum_to(&vec![3]);
    assert_tensor(&y, &ten![5.0, 7.0, 9.0]);
    Ok(())
}

define_test!(test_sum_to2, test_sum_to2_cpu, test_sum_to2_cuda);

fn test_sum_to3<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0]].to_device(device)?;
    let x = x.broadcast_to(vec![4, 3]);
    let y = x.sum_to(&vec![1, 3]);
    assert_tensor(&y, &ten![[4.0, 8.0, 12.0]]);
    Ok(())
}

define_test!(test_sum_to3, test_sum_to3_cpu, test_sum_to3_cuda);

fn test_sum_to4<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0]].to_device(device)?;
    let x = x.broadcast_to(vec![4, 3]);
    let y = x.sum_to(&vec![4, 1]);
    assert_tensor(&y, &ten![[6.0], [6.0], [6.0], [6.0]]);
    Ok(())
}

define_test!(test_sum_to4, test_sum_to4_cpu, test_sum_to4_cuda);

fn test_sum_to5<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?;
    let y = x.sum_to(&vec![1]);
    assert_tensor(&y, &ten![21.0]);
    Ok(())
}

define_test!(test_sum_to5, test_sum_to5_cpu, test_sum_to5_cuda);

fn test_sum_to6<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?;
    let y = x.sum_to(&vec![2, 1]);
    assert_tensor(&y, &ten![[6.0], [15.0]]);
    Ok(())
}

define_test!(test_sum_to6, test_sum_to6_cpu, test_sum_to6_cuda);

fn test_sum_to_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let y = x.sum_to(&vec![3]);
    assert_tensor(&y, &ten![5.0, 7.0, 9.0]);
    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(gx, &ten![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);
    Ok(())
}

define_test!(
    test_sum_to_backward,
    test_sum_to_backward_cpu,
    test_sum_to_backward_cuda
);

fn test_sum_axis<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?;
    let y = x.sum_axis(0, false);
    assert_tensor(&y, &ten![5.0, 7.0, 9.0]);
    Ok(())
}

define_test!(test_sum_axis, test_sum_axis_cpu, test_sum_axis_cuda);

fn test_sum_axis_keepdims_true<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?;
    let y = x.sum_axis(0, true);
    assert_tensor(&y, &ten![[5.0, 7.0, 9.0]]);
    Ok(())
}

define_test!(
    test_sum_axis_keepdims_true,
    test_sum_axis_keepdims_true_cpu,
    test_sum_axis_keepdims_true_cuda
);

fn test_sum_axis_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let y = x.sum_axis(0, false);
    assert_tensor(&y, &ten![5.0, 7.0, 9.0]);
    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(gx, &ten![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);
    Ok(())
}

define_test!(
    test_sum_axis_backward,
    test_sum_axis_backward_cpu,
    test_sum_axis_backward_cuda
);

fn test_sum_axis_backward_keepdims_true<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let y = x.sum_axis(0, true);
    assert_tensor(&y, &ten![[5.0, 7.0, 9.0]]);
    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(gx, &ten![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);
    Ok(())
}

define_test!(
    test_sum_axis_backward_keepdims_true,
    test_sum_axis_backward_keepdims_true_cpu,
    test_sum_axis_backward_keepdims_true_cuda
);

fn test_max<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 4.0, -2.0], [2.0, 3.0, -1.0]].to_device(device)?;
    let y = x.max();
    assert_tensor(&y, &ten![4.0]);
    Ok(())
}

define_test!(test_max, test_max_cpu, test_max_cuda);

fn test_max_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 4.0, -2.0], [2.0, 3.0, -1.0]]
        .to_device(device)?
        .requires_grad();
    let y = x.max() * 2.0;
    assert_tensor(&y, &ten![8.0]);
    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(&gx, &ten![[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]);
    Ok(())
}

define_test!(
    test_max_backward,
    test_max_backward_cpu,
    test_max_backward_cuda
);

fn test_max_axis<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 4.0, -2.0], [2.0, 3.0, -1.0]].to_device(device)?;
    let y = x.max_axis(0, false);
    assert_tensor(&y, &ten![2.0, 4.0, -1.0]);
    Ok(())
}

define_test!(test_max_axis, test_max_axis_cpu, test_max_axis_cuda);

fn test_max_axis_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 4.0, -2.0], [2.0, 3.0, -1.0]]
        .to_device(device)?
        .requires_grad();
    let y = x.max_axis(0, false) * 2.0;
    assert_tensor(&y, &ten![4.0, 8.0, -2.0]);
    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(&gx, &ten![[0.0, 2.0, 0.0], [2.0, 0.0, 2.0]]);
    Ok(())
}

define_test!(
    test_max_axis_backward,
    test_max_axis_backward_cpu,
    test_max_axis_backward_cuda
);

fn test_max_axis_keepdims_true<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 4.0, -2.0], [2.0, 3.0, -1.0]].to_device(device)?;
    let y = x.max_axis(0, true);
    assert_tensor(&y, &ten![[2.0, 4.0, -1.0]]);
    Ok(())
}

define_test!(
    test_max_axis_keepdims_true,
    test_max_axis_keepdims_true_cpu,
    test_max_axis_keepdims_true_cuda
);

fn test_max_axis_backward_keepdims_true<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 4.0, -2.0], [2.0, 3.0, -1.0]]
        .to_device(device)?
        .requires_grad();
    let y = x.max_axis(0, true) * 2.0;
    assert_tensor(&y, &ten![[4.0, 8.0, -2.0]]);
    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(&gx, &ten![[0.0, 2.0, 0.0], [2.0, 0.0, 2.0]]);
    Ok(())
}

define_test!(
    test_max_axis_backward_keepdims_true,
    test_max_axis_backward_keepdims_true_cpu,
    test_max_axis_backward_keepdims_true_cuda
);

fn test_argmax_axis<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 4.0, -2.0], [2.0, 3.0, -1.0]].to_device(device)?;
    let y = x.argmax_axis(0, false);
    assert_tensor(&y, &ten![1, 0, 1]);
    Ok(())
}

define_test!(
    test_argmax_axis,
    test_argmax_axis_cpu,
    test_argmax_axis_cuda
);

fn test_argmax_axis_keepdims_true<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 4.0, -2.0], [2.0, 3.0, -1.0]].to_device(device)?;
    let y = x.argmax_axis(0, true);
    assert_tensor(&y, &ten![[1, 0, 1]]);
    Ok(())
}

define_test!(
    test_argmax_axis_keepdims_true,
    test_argmax_axis_keepdims_true_cpu,
    test_argmax_axis_keepdims_true_cuda
);

fn test_multinomial<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![0.1, 0.2, 0.7].to_device(device)?;
    let y = x.multinomial(10, Some(838861));
    assert_tensor(&y, &ten![1, 2, 2, 2, 2, 2, 2, 0, 2, 2]);
    Ok(())
}

define_test!(
    test_multinomial,
    test_multinomial_cpu,
    test_multinomial_cuda
);

fn test_contiguous<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![1.0, 2.0, 3.0].to_device(device)?;
    let x = x.broadcast_to(vec![2, 3]);
    assert!(!x.is_contiguous());
    let y = x.contiguous();
    assert!(y.is_contiguous());
    assert_tensor(&y, &ten![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
    Ok(())
}

define_test!(test_contiguous, test_contiguous_cpu, test_contiguous_cuda);

fn test_contiguous_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![1.0, 2.0, 3.0].to_device(device)?.requires_grad();
    let h = x.broadcast_to(vec![2, 3]);
    assert!(!h.is_contiguous());
    let y = h.contiguous();
    assert!(y.is_contiguous());
    assert_tensor(&y, &ten![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(gx, &ten![2.0, 2.0, 2.0]);
    Ok(())
}

define_test!(
    test_contiguous_backward,
    test_contiguous_backward_cpu,
    test_contiguous_backward_cuda
);

fn test_sigmoid<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![-1.0, 0.0, 1.0].to_device(device)?;
    let y = x.sigmoid();
    assert_tensor(&y, &ten![0.2689414213699951, 0.5, 0.7310585786300049]);
    Ok(())
}

define_test!(test_sigmoid, test_sigmoid_cpu, test_sigmoid_cuda);

fn test_sigmoid_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![-1.0, 0.0, 1.0].to_device(device)?.requires_grad();
    let y = x.sigmoid();
    assert_tensor(&y, &ten![0.2689414213699951, 0.5, 0.7310585786300049]);
    assert!(x.is_requires_grad());
    assert!(y.is_requires_grad());

    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(&gx, &ten![0.19661193324148185, 0.25, 0.19661193324148188]);

    Ok(())
}

define_test!(
    test_sigmoid_backward,
    test_sigmoid_backward_cpu,
    test_sigmoid_backward_cuda
);

fn test_relu<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![-1.0, 0.0, 1.0].to_device(device)?;
    let y = x.relu();
    assert_tensor(&y, &ten![0.0, 0.0, 1.0]);
    Ok(())
}

define_test!(test_relu, test_relu_cpu, test_relu_cuda);

fn test_relu_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![-1.0, 0.0, 1.0].to_device(device)?.requires_grad();
    let y = x.relu() * ten![2.0].to_device(device)?;
    assert_tensor(&y, &ten![0.0, 0.0, 2.0]);
    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(&gx, &ten![0.0, 0.0, 2.0]);
    Ok(())
}

define_test!(
    test_relu_backward,
    test_relu_backward_cpu,
    test_relu_backward_cuda
);

fn test_leaky_relu<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![-2.0, 0.0, 2.0].to_device(device)?.requires_grad();
    let y = x.leaky_relu(0.2);
    assert_tensor(&y, &ten![-0.4, 0.0, 2.0]);
    Ok(())
}

define_test!(test_leaky_relu, test_leaky_relu_cpu, test_leaky_relu_cuda);

fn test_leaky_relu_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![-2.0, 0.0, 2.0].to_device(device)?.requires_grad();
    let y = x.leaky_relu(0.2);
    assert_tensor(&y, &ten![-0.4, 0.0, 2.0]);
    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(gx, &ten![0.2, 0.2, 1.0]);
    Ok(())
}

define_test!(
    test_leaky_relu_backward,
    test_leaky_relu_backward_cpu,
    test_leaky_relu_backward_cuda
);

fn test_silu<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![-2.0, 0.0, 2.0].to_device(device)?.requires_grad();
    let y = x.silu();
    assert_tensor(&y, &ten![-0.2384058386, 0.0000000000, 1.7615940571]);
    Ok(())
}

define_test!(test_silu, test_silu_cpu, test_silu_cuda);

fn test_silu_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![-2.0, 0.0, 2.0].to_device(device)?.requires_grad();
    let y = x.silu();
    assert_tensor(&y, &ten![-0.2384058386, 0.0000000000, 1.7615940571]);
    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(gx, &ten![-0.0907842517, 0.5000000000, 1.0907843113]);
    Ok(())
}

define_test!(
    test_silu_backward,
    test_silu_backward_cpu,
    test_silu_backward_cuda
);

fn test_gelu<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![-2.0, 0.0, 2.0].to_device(device)?.requires_grad();
    let y = x.gelu();
    assert_tensor(&y, &ten![-0.0455000997, 0.0000000000, 1.9544999599]);
    Ok(())
}

define_test!(test_gelu, test_gelu_cpu, test_gelu_cuda);

fn test_gelu_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![-2.0, 0.0, 2.0].to_device(device)?.requires_grad();
    let y = x.gelu();
    assert_tensor(&y, &ten![-0.0455000997, 0.0000000000, 1.9544999599]);
    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor_with_eps(gx, &ten![-0.0852318704, 0.5000000000, 1.0852319002], 1e-3);
    Ok(())
}

define_test!(
    test_gelu_backward,
    test_gelu_backward_cpu,
    test_gelu_backward_cuda
);

fn test_dropout_train<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].to_device(device)?;
    let y = x.dropout(0.5, true, Some(838861));
    assert_tensor(&y, &ten![2.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    Ok(())
}

define_test!(
    test_dropout_train,
    test_dropout_train_cpu,
    test_dropout_train_cuda
);

fn test_dropout_test<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].to_device(device)?;
    let y = x.dropout(0.5, false, Some(838861));
    assert_tensor(&y, &ten![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    Ok(())
}

define_test!(
    test_dropout_test,
    test_dropout_test_cpu,
    test_dropout_test_cuda
);

fn test_dropout_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        .to_device(device)?
        .requires_grad();
    let y = x.dropout(0.5, true, Some(838861));
    assert_tensor(&y, &ten![2.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(&gx, &ten![2.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    Ok(())
}

define_test!(
    test_dropout_backward,
    test_dropout_backward_cpu,
    test_dropout_backward_cuda
);

fn test_softmax<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![0.0, 1.0, 2.0].to_device(device)?;
    let y = x.softmax(0);
    assert_tensor(
        &y,
        &ten![
            0.09003057317038044848889200011399,
            0.24472847105479761387059056687576,
            0.66524095577482178498485154705122
        ],
    );
    Ok(())
}

define_test!(test_softmax, test_softmax_cpu, test_softmax_cuda);

fn test_softmax_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![0.0, 1.0, 2.0].to_device(device)?.requires_grad();
    let y = x.softmax(0) * ten![2.0].to_device(device)?;
    assert_tensor(
        &y,
        &ten![
            0.18006114634076089697778400022798,
            0.48945694210959522774118113375152,
            1.33048191154964356996970309410244
        ],
    );
    let grads = y.get_item(vec![(0, 1)]).backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(
        &gx,
        &ten![
            0.16385013812998644455731778180052,
            -0.04406608904034858137377383968669,
            -0.11978404908963782848907442257769
        ],
    );
    Ok(())
}

define_test!(
    test_softmax_backward,
    test_softmax_backward_cpu,
    test_softmax_backward_cuda
);

fn test_im2col<B: Backend>(device: Device<B>) -> Result<()> {
    let batch_size = 2usize;
    let img_ch = 3;
    let img_h = 4;
    let img_w = 5;
    let fil_h = 2;
    let fil_w = 3;
    let out_h = 3;
    let out_w = 3;
    let x = Tensor::<_, f64>::arange(0..((batch_size * img_ch * img_h * img_w) as isize), device)
        .reshape(vec![batch_size, img_ch, img_h, img_w]);
    let y = x.im2col(out_h, out_w, fil_h, fil_w, 1, 1);
    assert_eq!(
        y.shape(),
        &vec![batch_size, img_ch * fil_h * fil_w, out_h * out_w]
    );
    assert_eq!(
        y.to_vec(),
        vec![
            0., 1., 2., 5., 6., 7., 10., 11., 12., 1., 2., 3., 6., 7., 8., 11., 12., 13., 2., 3.,
            4., 7., 8., 9., 12., 13., 14., 5., 6., 7., 10., 11., 12., 15., 16., 17., 6., 7., 8.,
            11., 12., 13., 16., 17., 18., 7., 8., 9., 12., 13., 14., 17., 18., 19., 20., 21., 22.,
            25., 26., 27., 30., 31., 32., 21., 22., 23., 26., 27., 28., 31., 32., 33., 22., 23.,
            24., 27., 28., 29., 32., 33., 34., 25., 26., 27., 30., 31., 32., 35., 36., 37., 26.,
            27., 28., 31., 32., 33., 36., 37., 38., 27., 28., 29., 32., 33., 34., 37., 38., 39.,
            40., 41., 42., 45., 46., 47., 50., 51., 52., 41., 42., 43., 46., 47., 48., 51., 52.,
            53., 42., 43., 44., 47., 48., 49., 52., 53., 54., 45., 46., 47., 50., 51., 52., 55.,
            56., 57., 46., 47., 48., 51., 52., 53., 56., 57., 58., 47., 48., 49., 52., 53., 54.,
            57., 58., 59., 60., 61., 62., 65., 66., 67., 70., 71., 72., 61., 62., 63., 66., 67.,
            68., 71., 72., 73., 62., 63., 64., 67., 68., 69., 72., 73., 74., 65., 66., 67., 70.,
            71., 72., 75., 76., 77., 66., 67., 68., 71., 72., 73., 76., 77., 78., 67., 68., 69.,
            72., 73., 74., 77., 78., 79., 80., 81., 82., 85., 86., 87., 90., 91., 92., 81., 82.,
            83., 86., 87., 88., 91., 92., 93., 82., 83., 84., 87., 88., 89., 92., 93., 94., 85.,
            86., 87., 90., 91., 92., 95., 96., 97., 86., 87., 88., 91., 92., 93., 96., 97., 98.,
            87., 88., 89., 92., 93., 94., 97., 98., 99., 100., 101., 102., 105., 106., 107., 110.,
            111., 112., 101., 102., 103., 106., 107., 108., 111., 112., 113., 102., 103., 104.,
            107., 108., 109., 112., 113., 114., 105., 106., 107., 110., 111., 112., 115., 116.,
            117., 106., 107., 108., 111., 112., 113., 116., 117., 118., 107., 108., 109., 112.,
            113., 114., 117., 118., 119.
        ]
    );
    Ok(())
}

define_test!(test_im2col, test_im2col_cpu, test_im2col_cuda);

fn test_col2im<B: Backend>(device: Device<B>) -> Result<()> {
    let batch_size = 2;
    let img_ch = 3;
    let img_h = 4;
    let img_w = 5;
    let fil_h = 2;
    let fil_w = 3;
    let out_h = 3;
    let out_w = 3;
    let x = Tensor::<_, f64>::arange(
        0..((batch_size * img_ch * fil_h * fil_w * out_h * out_w) as isize),
        device,
    )
    .reshape(vec![batch_size, img_ch * fil_h * fil_w, out_h * out_w]);
    let img_shape = vec![batch_size, img_ch, img_h, img_w];
    let y = x.col2im(img_shape.clone(), out_h, out_w, fil_h, fil_w, 1, 1);
    assert_eq!(y.shape(), &img_shape);
    assert_eq!(
        y.to_vec(),
        vec![
            0., 10., 30., 30., 20., 30., 80., 150., 120., 70., 36., 92., 168., 132., 76., 33., 76.,
            129., 96., 53., 54., 118., 192., 138., 74., 138., 296., 474., 336., 178., 144., 308.,
            492., 348., 184., 87., 184., 291., 204., 107., 108., 226., 354., 246., 128., 246.,
            512., 798., 552., 286., 252., 524., 816., 564., 292., 141., 292., 453., 312., 161.,
            162., 334., 516., 354., 182., 354., 728., 1122., 768., 394., 360., 740., 1140., 780.,
            400., 195., 400., 615., 420., 215., 216., 442., 678., 462., 236., 462., 944., 1446.,
            984., 502., 468., 956., 1464., 996., 508., 249., 508., 777., 528., 269., 270., 550.,
            840., 570., 290., 570., 1160., 1770., 1200., 610., 576., 1172., 1788., 1212., 616.,
            303., 616., 939., 636., 323.
        ]
    );
    Ok(())
}

define_test!(test_col2im, test_col2im_cpu, test_col2im_cuda);

fn test_zero_padding2d<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 1, 2, 3], device);
    let y = x.zero_padding2d(2, 3);
    assert_eq!(y.shape(), &vec![1, 1, 4, 6]);
    assert_eq!(
        y.to_vec(),
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
    );
    Ok(())
}

define_test!(
    test_zero_padding2d,
    test_zero_padding2d_cpu,
    test_zero_padding2d_cuda
);

fn test_zero_padding2d_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 1, 2, 3], device)
        .requires_grad();
    let y = x.zero_padding2d(2, 3);
    assert_eq!(y.shape(), &vec![1, 1, 4, 6]);
    assert_eq!(
        y.to_vec(),
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
    );
    let x2 = Tensor::from_vec(
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0, 12.0, 13.0, 0.0, 0.0, 0.0, 14.0, 15.0, 16.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        vec![1, 1, 4, 6],
        device,
    );
    let y2 = y * x2;
    let grads = y2.backward();
    let gx = grads.get(&x).unwrap();
    assert_eq!(gx.shape(), &vec![1, 1, 2, 3]);
    assert_eq!(gx.to_vec(), vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
    Ok(())
}

define_test!(
    test_zero_padding2d_backward,
    test_zero_padding2d_backward_cpu,
    test_zero_padding2d_backward_cuda
);

fn test_cropping2d<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::from_vec(
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        vec![1, 1, 4, 6],
        device,
    );
    let y = x.cropping2d(2, 3);
    assert_eq!(y.shape(), &vec![1, 1, 2, 3]);
    assert_eq!(y.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    Ok(())
}

define_test!(test_cropping2d, test_cropping2d_cpu, test_cropping2d_cuda);

fn test_cropping2d_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::from_vec(
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        vec![1, 1, 4, 6],
        device,
    )
    .requires_grad();
    let y = x.cropping2d(2, 3);
    assert_eq!(y.shape(), &vec![1, 1, 2, 3]);
    assert_eq!(y.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let x2 = Tensor::from_vec(
        vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        vec![1, 1, 2, 3],
        device,
    );
    let y2 = y * x2;
    let grads = y2.backward();
    let gx = grads.get(&x).unwrap();
    assert_eq!(gx.shape(), &vec![1, 1, 4, 6]);
    assert_eq!(
        gx.to_vec(),
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0, 12.0, 13.0, 0.0, 0.0, 0.0, 14.0, 15.0, 16.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
    );
    Ok(())
}

define_test!(
    test_cropping2d_backward,
    test_cropping2d_backward_cpu,
    test_cropping2d_backward_cuda
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
    let y = x.conv2d(
        &w,
        Some(&b),
        in_filters,
        out_filters,
        fil_h,
        fil_w,
        1,
        1,
        None,
        false,
    );
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

fn test_conv2d_auto_padding<B: Backend>(device: Device<B>) -> Result<()> {
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
    let y = x.conv2d(
        &w,
        Some(&b),
        in_filters,
        out_filters,
        fil_h,
        fil_w,
        1,
        1,
        None,
        true,
    );
    assert_eq!(y.shape(), &vec![batch_size, out_filters, img_h, img_w]);
    assert_eq!(
        y.to_vec(),
        vec![
            3492., 5115., 5268., 5421., 3504., 4032., 5880., 6033., 6186., 3984., 4572., 6645.,
            6798., 6951., 4464., 2079., 2994., 3057., 3120., 1983., 8461., 12730., 13207., 13684.,
            9121., 10081., 15115., 15592., 16069., 10681., 11701., 17500., 17977., 18454., 12241.,
            5914., 8827., 9052., 9277., 6142., 9972., 14295., 14448., 14601., 9264., 10512.,
            15060., 15213., 15366., 9744., 11052., 15825., 15978., 16131., 10224., 4779., 6774.,
            6837., 6900., 4323., 27901., 41350., 41827., 42304., 27841., 29521., 43735., 44212.,
            44689., 29401., 31141., 46120., 46597., 47074., 30961., 15094., 22327., 22552., 22777.,
            14962.,
        ]
    );

    Ok(())
}

define_test!(
    test_conv2d_auto_padding,
    test_conv2d_auto_padding_cpu,
    test_conv2d_auto_padding_cuda
);

fn test_conv2d_strides<B: Backend>(device: Device<B>) -> Result<()> {
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
    let y = x.conv2d(
        &w,
        Some(&b),
        in_filters,
        out_filters,
        fil_h,
        fil_w,
        2,
        3,
        None,
        false,
    );
    assert_eq!(y.shape(), &vec![batch_size, out_filters, 2, 1]);
    assert_eq!(
        y.to_vec(),
        vec![5115., 6645., 12730., 17500., 14295., 15825., 41350., 46120.]
    );
    Ok(())
}

define_test!(
    test_conv2d_strides,
    test_conv2d_strides_cpu,
    test_conv2d_strides_cuda
);

// TODO: auto_paddingをtrueにするとエラーになる。
fn test_conv2d_strides_auto_padding<B: Backend>(device: Device<B>) -> Result<()> {
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
    let y = x.conv2d(
        &w,
        Some(&b),
        in_filters,
        out_filters,
        fil_h,
        fil_w,
        2,
        3,
        Some((0, 2)),
        false,
    );
    assert_eq!(y.shape(), &vec![batch_size, out_filters, 2, 2]);
    assert_eq!(
        y.to_vec(),
        vec![
            3492., 5421., 4572., 6951., 8461., 13684., 11701., 18454., 9972., 14601., 11052.,
            16131., 27901., 42304., 31141., 47074.
        ]
    );
    Ok(())
}

define_test!(
    test_conv2d_strides_auto_padding,
    test_conv2d_strides_auto_padding_cpu,
    test_conv2d_strides_auto_padding_cuda
);

fn test_conv2d_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let batch_size = 2;
    let in_filters = 3;
    let out_filters = 2;
    let img_h = 4;
    let img_w = 5;
    let fil_h = 2;
    let fil_w = 3;
    let x = arange_with_shape::<_, f64>(&[batch_size, in_filters, img_h, img_w], device)
        .requires_grad();
    let w = arange_with_shape(&[out_filters, in_filters, fil_h, fil_w], device).requires_grad();
    let b = arange_with_shape(&[out_filters], device).requires_grad();
    let y = x.conv2d(
        &w,
        Some(&b),
        in_filters,
        out_filters,
        fil_h,
        fil_w,
        1,
        1,
        None,
        false,
    );
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

    let grads = y.backward();

    let gx = grads.get(&x).unwrap();
    assert_eq!(gx.shape(), &vec![batch_size, in_filters, img_h, img_w]);
    assert_eq!(
        gx.to_vec(),
        vec![
            18., 38., 60., 42., 22., 42., 88., 138., 96., 50., 42., 88., 138., 96., 50., 24., 50.,
            78., 54., 28., 30., 62., 96., 66., 34., 66., 136., 210., 144., 74., 66., 136., 210.,
            144., 74., 36., 74., 114., 78., 40., 42., 86., 132., 90., 46., 90., 184., 282., 192.,
            98., 90., 184., 282., 192., 98., 48., 98., 150., 102., 52., 18., 38., 60., 42., 22.,
            42., 88., 138., 96., 50., 42., 88., 138., 96., 50., 24., 50., 78., 54., 28., 30., 62.,
            96., 66., 34., 66., 136., 210., 144., 74., 66., 136., 210., 144., 74., 36., 74., 114.,
            78., 40., 42., 86., 132., 90., 46., 90., 184., 282., 192., 98., 90., 184., 282., 192.,
            98., 48., 98., 150., 102., 52.
        ]
    );

    let gw = grads.get(&w).unwrap();
    assert_eq!(gw.shape(), &vec![out_filters, in_filters, fil_h, fil_w]);
    assert_eq!(
        gw.to_vec(),
        vec![
            648., 666., 684., 738., 756., 774., 1008., 1026., 1044., 1098., 1116., 1134., 1368.,
            1386., 1404., 1458., 1476., 1494., 648., 666., 684., 738., 756., 774., 1008., 1026.,
            1044., 1098., 1116., 1134., 1368., 1386., 1404., 1458., 1476., 1494.
        ]
    );

    let gb = grads.get(&b).unwrap();
    assert_eq!(gb.shape(), &vec![out_filters]);
    assert_eq!(gb.to_vec(), vec![18., 18.]);
    Ok(())
}

define_test!(
    test_conv2d_backward,
    test_conv2d_backward_cpu,
    test_conv2d_backward_cuda
);

fn test_conv2d_backward_auto_padding<B: Backend>(device: Device<B>) -> Result<()> {
    let batch_size = 2;
    let in_filters = 3;
    let out_filters = 2;
    let img_h = 4;
    let img_w = 5;
    let fil_h = 2;
    let fil_w = 3;
    let x = arange_with_shape::<_, f64>(&[batch_size, in_filters, img_h, img_w], device)
        .requires_grad();
    let w = arange_with_shape(&[out_filters, in_filters, fil_h, fil_w], device).requires_grad();
    let b = arange_with_shape(&[out_filters], device).requires_grad();
    let y = x.conv2d(
        &w,
        Some(&b),
        in_filters,
        out_filters,
        fil_h,
        fil_w,
        1,
        1,
        None,
        true,
    );
    assert_eq!(y.shape(), &vec![batch_size, out_filters, img_h, img_w]);
    assert_eq!(
        y.to_vec(),
        vec![
            3492., 5115., 5268., 5421., 3504., 4032., 5880., 6033., 6186., 3984., 4572., 6645.,
            6798., 6951., 4464., 2079., 2994., 3057., 3120., 1983., 8461., 12730., 13207., 13684.,
            9121., 10081., 15115., 15592., 16069., 10681., 11701., 17500., 17977., 18454., 12241.,
            5914., 8827., 9052., 9277., 6142., 9972., 14295., 14448., 14601., 9264., 10512.,
            15060., 15213., 15366., 9744., 11052., 15825., 15978., 16131., 10224., 4779., 6774.,
            6837., 6900., 4323., 27901., 41350., 41827., 42304., 27841., 29521., 43735., 44212.,
            44689., 29401., 31141., 46120., 46597., 47074., 30961., 15094., 22327., 22552., 22777.,
            14962.,
        ]
    );

    let grads = y.backward();

    let gx = grads.get(&x).unwrap();
    assert_eq!(gx.shape(), &vec![batch_size, in_filters, img_h, img_w]);
    assert_eq!(
        gx.to_vec(),
        vec![
            38., 60., 60., 60., 42., 88., 138., 138., 138., 96., 88., 138., 138., 138., 96., 88.,
            138., 138., 138., 96., 62., 96., 96., 96., 66., 136., 210., 210., 210., 144., 136.,
            210., 210., 210., 144., 136., 210., 210., 210., 144., 86., 132., 132., 132., 90., 184.,
            282., 282., 282., 192., 184., 282., 282., 282., 192., 184., 282., 282., 282., 192.,
            38., 60., 60., 60., 42., 88., 138., 138., 138., 96., 88., 138., 138., 138., 96., 88.,
            138., 138., 138., 96., 62., 96., 96., 96., 66., 136., 210., 210., 210., 144., 136.,
            210., 210., 210., 144., 136., 210., 210., 210., 144., 86., 132., 132., 132., 90., 184.,
            282., 282., 282., 192., 184., 282., 282., 282., 192., 184., 282., 282., 282., 192.
        ]
    );

    let gw = grads.get(&w).unwrap();
    assert_eq!(gw.shape(), &vec![out_filters, in_filters, fil_h, fil_w]);
    assert_eq!(
        gw.to_vec(),
        vec![
            1248., 1580., 1280., 996., 1260., 1020., 1888., 2380., 1920., 1476., 1860., 1500.,
            2528., 3180., 2560., 1956., 2460., 1980., 1248., 1580., 1280., 996., 1260., 1020.,
            1888., 2380., 1920., 1476., 1860., 1500., 2528., 3180., 2560., 1956., 2460., 1980.
        ]
    );

    let gb = grads.get(&b).unwrap();
    assert_eq!(gb.shape(), &vec![out_filters]);
    assert_eq!(gb.to_vec(), vec![40., 40.]);

    Ok(())
}

define_test!(
    test_conv2d_backward_auto_padding,
    test_conv2d_backward_auto_padding_cpu,
    test_conv2d_backward_auto_padding_cuda
);

fn test_conv2d_backward_strides<B: Backend>(device: Device<B>) -> Result<()> {
    let batch_size = 2;
    let in_filters = 3;
    let out_filters = 2;
    let img_h = 4;
    let img_w = 5;
    let fil_h = 2;
    let fil_w = 3;
    let x = arange_with_shape::<_, f64>(&[batch_size, in_filters, img_h, img_w], device)
        .requires_grad();
    let w = arange_with_shape(&[out_filters, in_filters, fil_h, fil_w], device).requires_grad();
    let b = arange_with_shape(&[out_filters], device).requires_grad();
    let y = x.conv2d(
        &w,
        Some(&b),
        in_filters,
        out_filters,
        fil_h,
        fil_w,
        2,
        3,
        None,
        false,
    );
    assert_eq!(y.shape(), &vec![batch_size, out_filters, 2, 1]);
    assert_eq!(
        y.to_vec(),
        vec![5115., 6645., 12730., 17500., 14295., 15825., 41350., 46120.]
    );

    let grads = y.backward();

    let gx = grads.get(&x).unwrap();
    assert_eq!(gx.shape(), &vec![batch_size, in_filters, img_h, img_w]);
    assert_eq!(
        gx.to_vec(),
        vec![
            18., 20., 22., 0., 0., 24., 26., 28., 0., 0., 18., 20., 22., 0., 0., 24., 26., 28., 0.,
            0., 30., 32., 34., 0., 0., 36., 38., 40., 0., 0., 30., 32., 34., 0., 0., 36., 38., 40.,
            0., 0., 42., 44., 46., 0., 0., 48., 50., 52., 0., 0., 42., 44., 46., 0., 0., 48., 50.,
            52., 0., 0., 18., 20., 22., 0., 0., 24., 26., 28., 0., 0., 18., 20., 22., 0., 0., 24.,
            26., 28., 0., 0., 30., 32., 34., 0., 0., 36., 38., 40., 0., 0., 30., 32., 34., 0., 0.,
            36., 38., 40., 0., 0., 42., 44., 46., 0., 0., 48., 50., 52., 0., 0., 42., 44., 46., 0.,
            0., 48., 50., 52., 0., 0.
        ]
    );

    let gw = grads.get(&w).unwrap();
    assert_eq!(gw.shape(), &vec![out_filters, in_filters, fil_h, fil_w]);
    assert_eq!(
        gw.to_vec(),
        vec![
            140., 144., 148., 160., 164., 168., 220., 224., 228., 240., 244., 248., 300., 304.,
            308., 320., 324., 328., 140., 144., 148., 160., 164., 168., 220., 224., 228., 240.,
            244., 248., 300., 304., 308., 320., 324., 328.
        ]
    );

    let gb = grads.get(&b).unwrap();
    assert_eq!(gb.shape(), &vec![out_filters]);
    assert_eq!(gb.to_vec(), vec![4., 4.]);

    Ok(())
}

define_test!(
    test_conv2d_backward_strides,
    test_conv2d_backward_strides_cpu,
    test_conv2d_backward_strides_cuda
);

fn test_conv2d_backward_strides_auto_padding<B: Backend>(device: Device<B>) -> Result<()> {
    let batch_size = 2;
    let in_filters = 3;
    let out_filters = 2;
    let img_h = 4;
    let img_w = 5;
    let fil_h = 2;
    let fil_w = 3;
    let x = arange_with_shape::<_, f64>(&[batch_size, in_filters, img_h, img_w], device)
        .requires_grad();
    let w = arange_with_shape(&[out_filters, in_filters, fil_h, fil_w], device).requires_grad();
    let b = arange_with_shape(&[out_filters], device).requires_grad();
    let y = x.conv2d(
        &w,
        Some(&b),
        in_filters,
        out_filters,
        fil_h,
        fil_w,
        2,
        3,
        Some((0, 2)),
        false,
    );
    assert_eq!(y.shape(), &vec![batch_size, out_filters, 2, 2]);
    assert_eq!(
        y.to_vec(),
        vec![
            3492., 5421., 4572., 6951., 8461., 13684., 11701., 18454., 9972., 14601., 11052.,
            16131., 27901., 42304., 31141., 47074.
        ]
    );

    let grads = y.backward();

    let gx = grads.get(&x).unwrap();
    assert_eq!(gx.shape(), &vec![batch_size, in_filters, img_h, img_w]);
    assert_eq!(
        gx.to_vec(),
        vec![
            20., 22., 18., 20., 22., 26., 28., 24., 26., 28., 20., 22., 18., 20., 22., 26., 28.,
            24., 26., 28., 32., 34., 30., 32., 34., 38., 40., 36., 38., 40., 32., 34., 30., 32.,
            34., 38., 40., 36., 38., 40., 44., 46., 42., 44., 46., 50., 52., 48., 50., 52., 44.,
            46., 42., 44., 46., 50., 52., 48., 50., 52., 20., 22., 18., 20., 22., 26., 28., 24.,
            26., 28., 20., 22., 18., 20., 22., 26., 28., 24., 26., 28., 32., 34., 30., 32., 34.,
            38., 40., 36., 38., 40., 32., 34., 30., 32., 34., 38., 40., 36., 38., 40., 44., 46.,
            42., 44., 46., 50., 52., 48., 50., 52., 44., 46., 42., 44., 46., 50., 52., 48., 50.,
            52.
        ]
    );

    let gw = grads.get(&w).unwrap();
    assert_eq!(gw.shape(), &vec![out_filters, in_filters, fil_h, fil_w]);
    assert_eq!(
        gw.to_vec(),
        vec![
            148., 292., 300., 168., 332., 340., 228., 452., 460., 248., 492., 500., 308., 612.,
            620., 328., 652., 660., 148., 292., 300., 168., 332., 340., 228., 452., 460., 248.,
            492., 500., 308., 612., 620., 328., 652., 660.
        ]
    );

    let gb = grads.get(&b).unwrap();
    assert_eq!(gb.shape(), &vec![out_filters]);
    assert_eq!(gb.to_vec(), vec![8., 8.]);

    Ok(())
}

define_test!(
    test_conv2d_backward_strides_auto_padding,
    test_conv2d_backward_strides_auto_padding_cpu,
    test_conv2d_backward_strides_auto_padding_cuda
);

fn test_deconv2d<B: Backend>(device: Device<B>) -> Result<()> {
    let batch_size = 2;
    let in_filters = 2;
    let out_filters = 3;
    let img_h = 3;
    let img_w = 3;
    let fil_h = 2;
    let fil_w = 3;
    let x = arange_with_shape::<_, f64>(&[batch_size, in_filters, img_h, img_w], device);
    let w = arange_with_shape(&[in_filters, out_filters, fil_h, fil_w], device);
    let b = arange_with_shape(&[out_filters], device);
    let y = x.deconv2d(
        &w,
        Some(&b),
        in_filters,
        out_filters,
        fil_h,
        fil_w,
        1,
        1,
        None,
        false,
    );
    assert_eq!(y.shape(), &vec![batch_size, out_filters, 4, 5]);
    assert_eq!(
        y.to_vec(),
        vec![
            162., 351., 569., 413., 224., 405., 876., 1417., 1024., 553., 531., 1140., 1831.,
            1312., 703., 333., 711., 1136., 809., 431., 217., 472., 768., 558., 303., 550., 1189.,
            1922., 1385., 746., 748., 1597., 2552., 1817., 968., 460., 976., 1551., 1098., 582.,
            272., 593., 967., 703., 382., 695., 1502., 2427., 1746., 939., 965., 2054., 3273.,
            2322., 1233., 587., 1241., 1966., 1387., 733., 486., 1035., 1649., 1169., 620., 1161.,
            2460., 3901., 2752., 1453., 1287., 2724., 4315., 3040., 1603., 765., 1611., 2540.,
            1781., 935., 757., 1588., 2496., 1746., 915., 1738., 3637., 5702., 3977., 2078., 1936.,
            4045., 6332., 4409., 2300., 1108., 2308., 3603., 2502., 1302., 1028., 2141., 3343.,
            2323., 1210., 2315., 4814., 7503., 5202., 2703., 2585., 5366., 8349., 5778., 2997.,
            1451., 3005., 4666., 3223., 1669.,
        ]
    );
    Ok(())
}

define_test!(test_deconv2d, test_deconv2d_cpu, test_deconv2d_cuda);

fn test_deconv2d_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let batch_size = 2;
    let in_filters = 2;
    let out_filters = 3;
    let img_h = 3;
    let img_w = 3;
    let fil_h = 2;
    let fil_w = 3;
    let x = arange_with_shape::<_, f64>(&[batch_size, in_filters, img_h, img_w], device)
        .requires_grad();
    let w = arange_with_shape(&[in_filters, out_filters, fil_h, fil_w], device).requires_grad();
    let b = arange_with_shape(&[out_filters], device).requires_grad();
    let y = x.deconv2d(
        &w,
        Some(&b),
        in_filters,
        out_filters,
        fil_h,
        fil_w,
        1,
        1,
        None,
        false,
    );
    assert_eq!(y.shape(), &vec![batch_size, out_filters, 4, 5]);
    assert_eq!(
        y.to_vec(),
        vec![
            162., 351., 569., 413., 224., 405., 876., 1417., 1024., 553., 531., 1140., 1831.,
            1312., 703., 333., 711., 1136., 809., 431., 217., 472., 768., 558., 303., 550., 1189.,
            1922., 1385., 746., 748., 1597., 2552., 1817., 968., 460., 976., 1551., 1098., 582.,
            272., 593., 967., 703., 382., 695., 1502., 2427., 1746., 939., 965., 2054., 3273.,
            2322., 1233., 587., 1241., 1966., 1387., 733., 486., 1035., 1649., 1169., 620., 1161.,
            2460., 3901., 2752., 1453., 1287., 2724., 4315., 3040., 1603., 765., 1611., 2540.,
            1781., 935., 757., 1588., 2496., 1746., 915., 1738., 3637., 5702., 3977., 2078., 1936.,
            4045., 6332., 4409., 2300., 1108., 2308., 3603., 2502., 1302., 1028., 2141., 3343.,
            2323., 1210., 2315., 4814., 7503., 5202., 2703., 2585., 5366., 8349., 5778., 2997.,
            1451., 3005., 4666., 3223., 1669.,
        ]
    );

    let grads = y.backward();

    let gx = grads.get(&x).unwrap();
    assert_eq!(gx.shape(), &vec![batch_size, in_filters, img_h, img_w]);
    assert_eq!(
        gx.to_vec(),
        vec![
            153., 153., 153., 153., 153., 153., 153., 153., 153., 477., 477., 477., 477., 477.,
            477., 477., 477., 477., 153., 153., 153., 153., 153., 153., 153., 153., 153., 477.,
            477., 477., 477., 477., 477., 477., 477., 477.
        ]
    );

    let gw = grads.get(&w).unwrap();
    assert_eq!(gw.shape(), &vec![in_filters, out_filters, fil_h, fil_w]);
    assert_eq!(
        gw.to_vec(),
        vec![
            234., 234., 234., 234., 234., 234., 234., 234., 234., 234., 234., 234., 234., 234.,
            234., 234., 234., 234., 396., 396., 396., 396., 396., 396., 396., 396., 396., 396.,
            396., 396., 396., 396., 396., 396., 396., 396.
        ]
    );

    let gb = grads.get(&b).unwrap();
    assert_eq!(gb.shape(), &vec![out_filters]);
    assert_eq!(gb.to_vec(), vec![40., 40., 40.]);
    Ok(())
}

define_test!(
    test_deconv2d_backward,
    test_deconv2d_backward_cpu,
    test_deconv2d_backward_cuda
);

fn test_max_pool2d<B: Backend>(device: Device<B>) -> Result<()> {
    let batch_size = 2;
    let img_ch = 3;
    let img_h = 4;
    let img_w = 5;
    let fil_h = 2;
    let fil_w = 3;
    let x = arange_with_shape::<B, f64>(&[batch_size, img_ch, img_h, img_w], device);
    let y = x.max_pool2d(fil_h, fil_w, None, None);
    assert_eq!(y.shape(), &vec![batch_size, img_ch, 2, 1]);
    assert_eq!(
        y.to_vec(),
        vec![7., 17., 27., 37., 47., 57., 67., 77., 87., 97., 107., 117.,]
    );
    Ok(())
}

define_test!(test_max_pool2d, test_max_pool2d_cpu, test_max_pool2d_cuda);

fn test_max_pool2d_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let batch_size = 2;
    let img_ch = 3;
    let img_h = 4;
    let img_w = 5;
    let fil_h = 2;
    let fil_w = 3;
    let x =
        arange_with_shape::<B, f64>(&[batch_size, img_ch, img_h, img_w], device).requires_grad();
    let y = x.max_pool2d(fil_h, fil_w, None, None);
    assert_eq!(y.shape(), &vec![batch_size, img_ch, 2, 1]);
    assert_eq!(
        y.to_vec(),
        vec![7., 17., 27., 37., 47., 57., 67., 77., 87., 97., 107., 117.,]
    );

    let grads = y.backward();

    let gx = grads.get(&x).unwrap();
    assert_eq!(gx.shape(), &vec![batch_size, img_ch, img_h, img_w]);
    assert_eq!(
        gx.to_vec(),
        vec![
            0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 1., 0., 0.
        ]
    );
    Ok(())
}

define_test!(
    test_max_pool2d_backward,
    test_max_pool2d_backward_cpu,
    test_max_pool2d_backward_cuda
);
