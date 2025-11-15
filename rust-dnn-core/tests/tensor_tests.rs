mod test_utils;
use core::panic;

use rust_dnn_core::{backend::Backend, device::Device, error::Result, ten, tensor::Tensor};

use crate::test_utils::assert_tensor;

fn test_to_vec<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![0.0, 1.0, 2.0].to_device(device)?;
    assert_eq!(x.to_vec(), vec![0.0, 1.0, 2.0]);
    Ok(())
}

define_test!(test_to_vec, test_to_vec_cpu, test_to_vec_cuda);

fn test_to_dtype<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![0.0, 1.0, 2.0].to_device(device)?;
    let y = x.to_dtype::<u32>()?;
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
    let x1 = ten![[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]].to_device(device)?;
    let x2 = ten![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]].to_device(device)?;
    let y = (x1 + x2)?;
    assert_tensor(&y, &ten![[4.0, 6.0, 8.0], [10.0, 12.0, 14.0]]);
    Ok(())
}

define_test!(test_add, test_add_cpu, test_add_cuda);

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
    let y = ((&x1 + &x2) * x3)?;
    assert_tensor(&y, &ten![[4.0, 12.0, 24.0], [40.0, 60.0, 84.0]]);
    let grads = y.backward()?;
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
    let x1 = ten![[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]].to_device(device)?;
    let x2 = ten![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]].to_device(device)?;
    let y = (x1 - x2)?;
    assert_tensor(&y, &ten![[-6.0, -6.0, -6.0], [-6.0, -6.0, -6.0]]);
    Ok(())
}

define_test!(test_sub, test_sub_cpu, test_sub_cuda);

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
    let y = ((&x1 - &x2)? * x3)?;
    assert_tensor(&y, &ten![[-6.0, -12.0, -18.0], [-24.0, -30.0, -36.0]]);
    let grads = y.backward()?;
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
    let x1 = ten![[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]].to_device(device)?;
    let x2 = ten![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]].to_device(device)?;
    let y = (x1 * x2)?;
    assert_tensor(&y, &ten![[-5.0, 0.0, 7.0], [16.0, 27.0, 40.0]]);
    Ok(())
}

define_test!(test_mul, test_mul_cpu, test_mul_cuda);

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
    let y = ((&x1 * &x2) * x3)?;
    assert_tensor(&y, &ten![[-5.0, 0.0, 21.0], [64.0, 135.0, 240.0]]);
    let grads = y.backward()?;
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
    let x1 = ten![[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]].to_device(device)?;
    let x2 = ten![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]].to_device(device)?;
    let y = (x1 / x2)?;
    assert_tensor(&y, &ten![[-0.2, 0.0, 0.14285715], [0.25, 0.33333334, 0.4]]);
    Ok(())
}

define_test!(test_div, test_div_cpu, test_div_cuda);

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
    let y = ((&x1 / &x2) * x3)?;
    assert_tensor(&y, &ten![[-0.2, 0.0, 0.42857146], [1.0, 1.6666667, 2.4]]);
    let grads = y.backward()?;
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
    let x = ten![[0.0, -2.0, -4.0], [1.0, 2.0, 3.0]].to_device(device)?;
    let y = -x;
    assert_tensor(&y, &ten![[0.0, 2.0, 4.0], [-1.0, -2.0, -3.0]]);
    Ok(())
}

define_test!(test_neg, test_neg_cpu, test_neg_cuda);

fn test_neg_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[0.0, -2.0, -4.0], [1.0, 2.0, 3.0]]
        .to_device(device)?
        .requires_grad();
    let y = -&x;
    assert_tensor(&y, &ten![[0.0, 2.0, 4.0], [-1.0, -2.0, -3.0]]);
    let grads = y.backward()?;
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
    let x1 = ten![-1.0, 0.0, 1.0].to_device(device)?;
    let x2 = ten![-1.0, 0.0, 2.0].to_device(device)?;
    let y = x1.eq(&x2)?;
    assert_tensor(&y, &ten![1, 1, 0]);
    Ok(())
}

define_test!(test_eq, test_eq_cpu, test_eq_cuda);

fn test_lt<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![-1.0, 0.0, 1.0].to_device(device)?;
    let x2 = ten![0.0, -1.0, 1.0].to_device(device)?;
    let y = x1.lt(&x2)?;
    assert_tensor(&y, &ten![1, 0, 0]);
    Ok(())
}

define_test!(test_lt, test_lt_cpu, test_lt_cuda);

fn test_le<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![-1.0, 0.0, 1.0].to_device(device)?;
    let x2 = ten![0.0, -1.0, 1.0].to_device(device)?;
    let y = x1.le(&x2)?;
    assert_tensor(&y, &ten![1, 0, 1]);
    Ok(())
}

define_test!(test_le, test_le_cpu, test_le_cuda);

fn test_gt<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![0.0, -1.0, 1.0].to_device(device)?;
    let x2 = ten![-1.0, 0.0, 1.0].to_device(device)?;
    let y = x1.gt(&x2)?;
    assert_tensor(&y, &ten![1, 0, 0]);
    Ok(())
}

define_test!(test_gt, test_gt_cpu, test_gt_cuda);

fn test_ge<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![0.0, -1.0, 1.0].to_device(device)?;
    let x2 = ten![-1.0, 0.0, 1.0].to_device(device)?;
    let y = x1.ge(&x2)?;
    assert_tensor(&y, &ten![1, 0, 1]);
    Ok(())
}

define_test!(test_ge, test_ge_cpu, test_ge_cuda);

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
    let y = x1.matmul(&x2)?;
    assert_tensor(&y, &ten![[-26.0, -32.0], [22.0, 28.0]]);
    Ok(())
}

define_test!(test_matmul, test_matmul_cpu, test_matmul_cuda);

fn test_matmul_batch<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = Tensor::<_, f64>::arange(0..(2 * 3 * 4 * 5), device).reshape(vec![2, 3, 4, 5])?;
    let x2 = Tensor::<_, f64>::arange(0..(2 * 3 * 5 * 6), device).reshape(vec![2, 3, 5, 6])?;
    let y = x1.matmul(&x2)?;
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
    let x1 = Tensor::<_, f64>::arange(0..(2 * 3 * 4 * 5), device).reshape(vec![2, 3, 4, 5])?;
    let x2 = Tensor::<_, f64>::arange(0..(5 * 6), device).reshape(vec![5, 6])?;
    let y = x1.matmul(&x2)?;
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
    let y = x1.reversed_axes()?.matmul(&x2)?;
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
    let y = (x1.matmul(&x2)? * ten![[1.0, 2.0], [3.0, 4.0]].to_device(device)?)?;
    assert_tensor(&y, &ten![[-26.0, -64.0], [66.0, 112.0]]);
    let grads = y.backward()?;
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
    let x1 = Tensor::<_, f64>::arange(0..(2 * 3 * 4 * 5), device).reshape(vec![2, 3, 4, 5])?;
    let x2 = Tensor::<_, f64>::arange(0..(2 * 3 * 5 * 6), device).reshape(vec![2, 3, 5, 6])?;
    let y = x1.matmul(&x2)?;
    assert_eq!(y.shape(), &vec![2, 3, 4, 6]);
    assert_eq!(y.to_vec(), MATMUL_BATCH_FORWARD_EXPECTED_DATA);
    let grads = y.backward()?;
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
    let y = x1.reversed_axes()?.matmul(&x2)?;
    assert_tensor(&y, &ten![[-2.0, -4.0], [16.0, 18.0]]);
    let grads = y.backward()?;
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
    let x1 = ten![-1.0].to_device(device)?;
    let x2 = ten![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]].to_device(device)?;
    let y = (x1 + x2)?;
    assert_tensor(&y, &ten![[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    Ok(())
}

define_test!(
    test_broadcast_op2,
    test_broadcast_op2_cpu,
    test_broadcast_op2_cuda
);

fn test_broadcast_op2_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![-1.0].to_device(device)?.requires_grad();
    let x2 = ten![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]
        .to_device(device)?
        .requires_grad();
    let x3 = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let y = ((&x1 + &x2) * x3)?;
    assert_tensor(&y, &ten![[4.0, 10.0, 18.0], [28.0, 40.0, 54.0]]);
    let grads = y.backward()?;
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
    let x1 = ten![[0.0, -2.0, -4.0], [1.0, 2.0, 3.0]].to_device(device)?;
    let x2 = ten![3.0].to_device(device)?;
    let y = x1.pow(&x2)?;
    assert_tensor(&y, &ten![[0.0, -8.0, -64.0], [1.0, 8.0, 27.0]]);
    Ok(())
}

define_test!(test_pow, test_pow_cpu, test_pow_cuda);

fn test_pow_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let x2 = ten![3.0].to_device(device)?.requires_grad();
    let y = x1.pow(&x2)?;
    assert_tensor(&y, &ten![[1.0, 8.0, 27.0], [64.0, 125.0, 216.0]]);
    let grads = y.backward()?;
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
    let y = x.pow_scalar(3.0)?;
    assert_tensor(&y, &ten![[0.0, -8.0, -64.0], [1.0, 8.0, 27.0]]);
    Ok(())
}

define_test!(test_pow_scalar, test_pow_scalar_cpu, test_pow_scalar_cuda);

fn test_pow_scalar_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[0.0, -2.0, -4.0], [1.0, 2.0, 3.0]]
        .to_device(device)?
        .requires_grad();
    let y = x.pow_scalar(3.0)?;
    assert_tensor(&y, &ten![[0.0, -8.0, -64.0], [1.0, 8.0, 27.0]]);
    let grads = y.backward()?;
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
    let x = ten![2.0].to_device(device)?;
    let y = x.exp();
    assert_tensor(&y, &ten![7.38905609893065]);
    Ok(())
}

define_test!(test_exp, test_exp_cpu, test_exp_cuda);

fn test_exp_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![2.0].to_device(device)?.requires_grad();
    let y = (x.exp() * ten![2.0].to_device(device)?)?;
    assert_tensor(&y, &ten![14.7781121978613]);
    let grads = y.backward()?;
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
    let y = (x.sqrt() * ten![2.0].to_device(device)?)?;
    assert_tensor(&y, &ten![4.0]);
    let grads = y.backward()?;
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
    let y = (x.ln() * ten![2.0].to_device(device)?)?;
    assert_tensor(&y, &ten![2.772588722239781]);
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap();
    assert_tensor(&gx, &ten![0.5]);
    Ok(())
}

define_test!(
    test_ln_backward,
    test_ln_backward_cpu,
    test_ln_backward_cuda
);

fn test_gather<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ]
    .to_device(device)?;
    let index = ten![[0, 1], [2, 3], [1, 3]].to_device(device)?;
    let y = x.gather(&index, 1)?;
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
    let y = (x.gather(&index, 1)? * Tensor::from_scalar(2.0, device))?;
    assert_tensor(&y, &ten![[2.0, 4.0], [14.0, 16.0], [20.0, 24.0]]);
    let grads = y.backward()?;
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
    x.scatter(&index, &src, 1)?;
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
    x.scatter_add(&index, &src, 1)?;
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
    let y = x.index_select(1, &index)?;
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
    let y = (x.index_select(1, &index)? * ten![2.0].to_device(device)?)?;
    assert_tensor(&y, &ten![[4.0, 8.0], [12.0, 16.0], [20.0, 24.0]]);
    let grads = y.backward()?;
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
    let src = ten![10.0, 20.0].to_device(device)?;
    x.index_copy(1, &index, &src)?;
    assert_tensor(
        &x,
        &ten![
            [1.0, 10.0, 3.0, 20.0],
            [5.0, 10.0, 7.0, 20.0],
            [9.0, 10.0, 11.0, 20.0]
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
    let src = ten![10.0, 20.0].to_device(device)?;
    x.index_add(1, &index, &src)?;
    assert_tensor(
        &x,
        &ten![
            [1.0, 12.0, 3.0, 24.0],
            [5.0, 16.0, 7.0, 28.0],
            [9.0, 20.0, 11.0, 32.0]
        ],
    );
    Ok(())
}

define_test!(test_index_add, test_index_add_cpu, test_index_add_cuda);

fn test_reshape<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?;
    let y = x.reshape(vec![6])?;
    assert_tensor(&y, &ten![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    Ok(())
}

define_test!(test_reshape, test_reshape_cpu, test_reshape_cuda);

fn test_reshape2<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![1.0, 2.0, 3.0].to_device(device)?;
    let x = x.broadcast_to(vec![2, 3])?;
    let y = x.reshape(vec![6])?;
    assert_tensor(&y, &ten![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    Ok(())
}

define_test!(test_reshape2, test_reshape2_cpu, test_reshape2_cuda);

fn test_reshape_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let x2 = ten![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].to_device(device)?;
    let y = (x1.reshape(vec![6])? * x2)?;
    assert_tensor(&y, &ten![1.0, 4.0, 9.0, 16.0, 25.0, 36.0]);
    let grads = y.backward()?;
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
    let y = (x1.broadcast_to(vec![2, 3])? * x2)?;
    assert_tensor(&y, &ten![[1.0, 4.0, 9.0], [4.0, 10.0, 18.0]]);
    let grads = y.backward()?;
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
    let y = x.squeeze_axes(&[0, 3])?;
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
    let y = x.unsqueeze(2)?;
    assert_tensor(&y, &ten![[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]]);
    Ok(())
}

define_test!(test_unsqueeze, test_unsqueeze_cpu, test_unsqueeze_cuda);

fn test_permuted_axes<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?;
    let y = x.permuted_axes(&[1, 0])?;
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
    let y = x.permuted_axes(&[1, 0])?;
    assert_tensor(&y, &ten![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
    let grads = y.backward()?;
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
    let y = x.reversed_axes()?;
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
    let y = Tensor::cat(&vec![x1, x2, x3], 0)?;
    assert_tensor(&y, &ten![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    Ok(())
}

define_test!(test_cat, test_cat_cpu, test_cat_cuda);

fn test_cat_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x1 = ten![[1.0, 2.0]].to_device(device)?.requires_grad();
    let x2 = ten![[3.0, 4.0]].to_device(device)?.requires_grad();
    let x3 = ten![[5.0, 6.0]].to_device(device)?.requires_grad();
    let y = Tensor::cat(&vec![x1.clone(), x2.clone(), x3.clone()], 0)?;
    let y = (y * ten![[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]].to_device(device)?)?;
    assert_tensor(&y, &ten![[10.0, 40.0], [90.0, 160.0], [250.0, 360.0]]);
    let grads = y.backward()?;
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

fn test_split<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let ys = x.split(1, &vec![2, 1])?;
    assert_tensor(&ys[0], &ten![[1.0, 2.0], [4.0, 5.0]]);
    assert_tensor(&ys[1], &ten![[3.0], [6.0]]);
    Ok(())
}

define_test!(test_split, test_split_cpu, test_split_cuda);

fn test_get_item<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ]
    .to_device(device)?;
    let y = x.get_item(vec![(1, 3), (1, 4)])?;
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
    let h = x.get_item(vec![(1, 2), (0, 3), (0, 4)])?;
    let y = h.get_item(vec![(0, 1), (1, 3), (1, 4)])?;
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
    let h = x.get_item(vec![(1, 3), (1, 4)])?;
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
    let h = x.get_item(vec![(1, 3), (1, 4)])?;
    let y = (h * ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?)?;
    assert_tensor(&y, &ten![[6.0, 14.0, 24.0], [40.0, 55.0, 72.0]]);
    let grads = y.backward()?;
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
    )?;
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
    let y = x.select(1, 2)?;
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
    let y = x.narrow(1, 1, 2)?;
    assert_tensor(&y, &ten![[2.0, 3.0], [6.0, 7.0], [10.0, 11.0]]);
    Ok(())
}

define_test!(test_narrow, test_narrow_cpu, test_narrow_cuda);

fn test_copy<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::zeros(vec![2, 3], device);
    x.copy(&ten![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]].to_device(device)?)?;
    assert_tensor(&x, &ten![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
    Ok(())
}

define_test!(test_copy, test_copy_cpu, test_copy_cuda);

fn test_copy2<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::zeros(vec![2, 3], device);
    x.copy(&ten![1.0, 2.0, 3.0].to_device(device)?)?;
    assert_tensor(&x, &ten![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
    Ok(())
}

define_test!(test_copy2, test_copy2_cpu, test_copy2_cuda);

fn test_copy3<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::zeros(vec![3, 4], device);
    let x2 = x.get_item(vec![(1, 3), (1, 4)])?;
    x2.copy(&ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?)?;
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
    let y = x.broadcast_to(vec![2, 3])?;
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
    let y = x.broadcast_to(vec![2, 3])?;
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
    let x = x.broadcast_to(vec![2, 3])?;
    let y = x.broadcast_to(vec![2, 3])?;
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
    let y = x.broadcast_to(vec![3, 2])?;
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
    let y = x.broadcast_to(vec![2, 3, 4])?;
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
    let y = (x1.broadcast_to(vec![2, 3])? * x2)?;
    assert_tensor(&y, &ten![[1.0, 4.0, 9.0], [4.0, 10.0, 18.0]]);
    let grads = y.backward()?;
    let gx = grads.get(&x1).unwrap();
    assert_tensor(gx, &ten![5.0, 7.0, 9.0]);
    Ok(())
}

define_test!(
    test_broadcast_to_backward,
    test_broadcast_to_backward_cpu,
    test_broadcast_to_backward_cuda
);

fn test_sum_to<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?;
    let y = x.sum_to(&vec![1, 3])?;
    assert_tensor(&y, &ten![[5.0, 7.0, 9.0]]);
    Ok(())
}

define_test!(test_sum_to, test_sum_to_cpu, test_sum_to_cuda);

fn test_sum_to2<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?;
    let y = x.sum_to(&vec![3])?;
    assert_tensor(&y, &ten![5.0, 7.0, 9.0]);
    Ok(())
}

define_test!(test_sum_to2, test_sum_to2_cpu, test_sum_to2_cuda);

fn test_sum_to3<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0]].to_device(device)?;
    let x = x.broadcast_to(vec![4, 3])?;
    let y = x.sum_to(&vec![1, 3])?;
    assert_tensor(&y, &ten![[4.0, 8.0, 12.0]]);
    Ok(())
}

define_test!(test_sum_to3, test_sum_to3_cpu, test_sum_to3_cuda);

fn test_sum_to4<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0]].to_device(device)?;
    let x = x.broadcast_to(vec![4, 3])?;
    let y = x.sum_to(&vec![4, 1])?;
    assert_tensor(&y, &ten![[6.0], [6.0], [6.0], [6.0]]);
    Ok(())
}

define_test!(test_sum_to4, test_sum_to4_cpu, test_sum_to4_cuda);

fn test_sum_to5<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?;
    let y = x.sum_to(&vec![1])?;
    assert_tensor(&y, &ten![21.0]);
    Ok(())
}

define_test!(test_sum_to5, test_sum_to5_cpu, test_sum_to5_cuda);

fn test_sum_to6<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?;
    let y = x.sum_to(&vec![2, 1])?;
    assert_tensor(&y, &ten![[6.0], [15.0]]);
    Ok(())
}

define_test!(test_sum_to6, test_sum_to6_cpu, test_sum_to6_cuda);

fn test_sum_to_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let y = x.sum_to(&vec![3])?;
    assert_tensor(&y, &ten![5.0, 7.0, 9.0]);
    let grads = y.backward()?;
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
    let y = x.sum_axis(0, false)?;
    assert_tensor(&y, &ten![5.0, 7.0, 9.0]);
    Ok(())
}

define_test!(test_sum_axis, test_sum_axis_cpu, test_sum_axis_cuda);

fn test_sum_axis_keepdims_true<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_device(device)?;
    let y = x.sum_axis(0, true)?;
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
    let y = x.sum_axis(0, false)?;
    assert_tensor(&y, &ten![5.0, 7.0, 9.0]);
    let grads = y.backward()?;
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
    let y = x.sum_axis(0, true)?;
    assert_tensor(&y, &ten![[5.0, 7.0, 9.0]]);
    let grads = y.backward()?;
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
    let y = x.max()?;
    assert_tensor(&y, &ten![4.0]);
    Ok(())
}

define_test!(test_max, test_max_cpu, test_max_cuda);

fn test_max_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 4.0, -2.0], [2.0, 3.0, -1.0]]
        .to_device(device)?
        .requires_grad();
    let y = (x.max()? * Tensor::from_scalar(2.0, device))?;
    assert_tensor(&y, &ten![8.0]);
    let grads = y.backward()?;
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
    let y = x.max_axis(0, false)?;
    assert_tensor(&y, &ten![2.0, 4.0, -1.0]);
    Ok(())
}

define_test!(test_max_axis, test_max_axis_cpu, test_max_axis_cuda);

fn test_max_axis_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 4.0, -2.0], [2.0, 3.0, -1.0]]
        .to_device(device)?
        .requires_grad();
    let y = (x.max_axis(0, false)? * Tensor::from_scalar(2.0, device))?;
    assert_tensor(&y, &ten![4.0, 8.0, -2.0]);
    let grads = y.backward()?;
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
    let y = x.max_axis(0, true)?;
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
    let y = (x.max_axis(0, true)? * Tensor::from_scalar(2.0, device))?;
    assert_tensor(&y, &ten![[4.0, 8.0, -2.0]]);
    let grads = y.backward()?;
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
    let y = x.argmax_axis(0, false)?;
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
    let y = x.argmax_axis(0, true)?;
    assert_tensor(&y, &ten![[1, 0, 1]]);
    Ok(())
}

define_test!(
    test_argmax_axis_keepdims_true,
    test_argmax_axis_keepdims_true_cpu,
    test_argmax_axis_keepdims_true_cuda
);

fn test_contiguous<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![1.0, 2.0, 3.0].to_device(device)?;
    let x = x.broadcast_to(vec![2, 3])?;
    assert!(!x.is_contiguous());
    let y = x.contiguous();
    assert!(y.is_contiguous());
    assert_tensor(&y, &ten![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
    Ok(())
}

define_test!(test_contiguous, test_contiguous_cpu, test_contiguous_cuda);

fn test_contiguous_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![1.0, 2.0, 3.0].to_device(device)?.requires_grad();
    let h = x.broadcast_to(vec![2, 3])?;
    assert!(!h.is_contiguous());
    let y = h.contiguous();
    assert!(y.is_contiguous());
    assert_tensor(&y, &ten![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap();
    assert_tensor(gx, &ten![2.0, 2.0, 2.0]);
    Ok(())
}

define_test!(
    test_contiguous_backward,
    test_contiguous_backward_cpu,
    test_contiguous_backward_cuda
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
    let y = (x.relu() * ten![2.0].to_device(device)?)?;
    assert_tensor(&y, &ten![0.0, 0.0, 2.0]);
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap();
    assert_tensor(&gx, &ten![0.0, 0.0, 2.0]);
    Ok(())
}

define_test!(
    test_relu_backward,
    test_relu_backward_cpu,
    test_relu_backward_cuda
);

fn test_softmax<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![0.0, 1.0, 2.0].to_device(device)?;
    let y = x.softmax(0)?;
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
    let y = (x.softmax(0)? * ten![2.0].to_device(device)?)?;
    assert_tensor(
        &y,
        &ten![
            0.18006114634076089697778400022798,
            0.48945694210959522774118113375152,
            1.33048191154964356996970309410244
        ],
    );
    let grads = y.get_item(vec![(0, 1)])?.backward()?;
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
