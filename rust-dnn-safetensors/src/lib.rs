use rust_dnn_core::{
    backend::Backend,
    cpu_backend::CpuBackend,
    device::Device,
    dtype::DType,
    error::{Error, Result},
    num::Num,
    tensor::Tensor,
};
use serde::{Deserialize, Serialize, de};
use serde_json::Value;
use std::{
    any::Any,
    collections::HashMap,
    hash::Hash,
    io::{Cursor, Read},
};

#[derive(Serialize, Deserialize)]
struct TensorData {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

pub fn serialize<B: Backend, T: Num>(tensors: HashMap<String, Tensor<B, T>>) -> Result<Vec<u8>> {
    let mut data: Vec<u8> = Vec::new();

    let mut tensor_datas = HashMap::new();
    for (name, tensor) in tensors {
        let tensor_data = add_buffer(tensor, &mut data)?;
        tensor_datas.insert(name, tensor_data);
    }
    let header_json = generate_header_json(tensor_datas)?;
    let mut result = Vec::new();
    result.extend((header_json.len() as u64).to_le_bytes());
    result.extend(header_json.as_bytes());
    result.extend(data);
    Ok(result)
}

fn generate_header_json(tensor_datas: HashMap<String, TensorData>) -> Result<String> {
    let Ok(header_json) = serde_json::to_string(&tensor_datas) else {
        return Err(Error::ArgumentsError {
            msg: "Invalid header".to_string(),
        });
    };

    Ok(header_json)
}

fn add_buffer<B: Backend, T: Num>(
    tensor: Tensor<B, T>,
    buffer: &mut Vec<u8>,
) -> Result<TensorData> {
    let tensor_data = match tensor.dtype() {
        DType::F32 => {
            let data_offset_begin = buffer.len();
            for value in tensor.to_vec() {
                buffer.extend(value.as_f32().to_le_bytes());
            }
            let data_offset_end = buffer.len();

            let tensor_data = TensorData {
                dtype: "F32".to_string(),
                shape: tensor.shape().to_vec(),
                data_offsets: [data_offset_begin, data_offset_end],
            };
            tensor_data
        }
        DType::F64 => {
            let data_offset_begin = buffer.len();
            for value in tensor.to_vec() {
                buffer.extend(value.as_f64().to_le_bytes());
            }
            let data_offset_end = buffer.len();

            let tensor_data = TensorData {
                dtype: "F64".to_string(),
                shape: tensor.shape().to_vec(),
                data_offsets: [data_offset_begin, data_offset_end],
            };
            tensor_data
        }
        _ => {
            return Err(Error::ArgumentsError {
                msg: format!("Not supported dtype({:?})", tensor.dtype()),
            });
        }
    };
    Ok(tensor_data)
}

pub fn deserialize<B: Backend, T: Num>(
    bytes: Vec<u8>,
    device: Device<B>,
) -> Result<HashMap<String, Tensor<B, T>>> {
    let (header_len, header_map) = parse_header(&bytes)?;
    let mut tensors = HashMap::<String, Tensor<B, T>>::new();
    let data_offset = 8 + header_len;
    for (name, tensor_data) in header_map {
        let tensor = parse_tensor(tensor_data, data_offset, &bytes, device)?;
        tensors.insert(name, tensor);
    }
    Ok(tensors)
}

fn parse_header(bytes: &Vec<u8>) -> Result<(usize, HashMap<String, TensorData>)> {
    let mut cursor = Cursor::new(&bytes);

    let mut len_buf = [0u8; 8];
    cursor.read_exact(&mut len_buf).unwrap();
    let header_len = u64::from_le_bytes(len_buf) as usize;

    let mut header_buf = vec![0u8; header_len];
    cursor.read_exact(&mut header_buf).unwrap();
    let Ok(header_json) = String::from_utf8(header_buf) else {
        return Err(Error::ArgumentsError {
            msg: "Invalid header".to_string(),
        });
    };

    let Ok(raw_header) = serde_json::from_str(&header_json) else {
        return Err(Error::ArgumentsError {
            msg: "Invalid header".to_string(),
        });
    };
    let mut header_map: HashMap<String, TensorData> = HashMap::new();
    if let Value::Object(map) = raw_header {
        for (key, value) in map {
            if key == "__metadata__" || key == "metadata" {
                // metadataはスキップ
                continue;
            }
            // tensor情報だけTensorDataに変換
            let Ok(td) = serde_json::from_value(value) else {
                return Err(Error::ArgumentsError {
                    msg: "Invalid header".to_string(),
                });
            };
            header_map.insert(key, td);
        }
    }

    Ok((header_len, header_map))
}

fn parse_tensor<B: Backend, T: Num>(
    tensor_data: TensorData,
    data_offset: usize,
    bytes: &Vec<u8>,
    device: Device<B>,
) -> Result<Tensor<B, T>> {
    let tensor = match tensor_data.dtype.as_str() {
        "F32" => {
            if T::dtype() != DType::F32 {
                return Err(Error::ArgumentsError {
                    msg: format!("Mismatch dtype({:?}, {:?})", tensor_data.dtype, T::dtype()),
                });
            }

            let start = tensor_data.data_offsets[0];
            let end = tensor_data.data_offsets[1];
            let bytes = &bytes[data_offset + start..data_offset + end];

            let mut values = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks(4) {
                let mut f32_buf = [0u8; 4];
                f32_buf.copy_from_slice(chunk);
                values.push(f32::from_le_bytes(f32_buf));
            }

            let tensor = Tensor::from_vec(values, tensor_data.shape, device)?;
            let tensor = unsafe { tensor.reinterpret_cast_dtype() };
            tensor
        }
        "F64" => {
            if T::dtype() != DType::F64 {
                return Err(Error::ArgumentsError {
                    msg: format!("Mismatch dtype({:?}, {:?})", tensor_data.dtype, T::dtype()),
                });
            }

            let start = tensor_data.data_offsets[0];
            let end = tensor_data.data_offsets[1];
            let bytes = &bytes[data_offset + start..data_offset + end];

            let mut values = Vec::with_capacity(bytes.len() / 8);
            for chunk in bytes.chunks(8) {
                let mut f64_buf = [0u8; 8];
                f64_buf.copy_from_slice(chunk);
                values.push(f64::from_le_bytes(f64_buf));
            }

            let tensor = Tensor::from_vec(values, tensor_data.shape, device)?;
            let tensor = unsafe { tensor.reinterpret_cast_dtype() };
            tensor
        }
        _ => {
            return Err(Error::ArgumentsError {
                msg: format!("Not supported dtype({:?})", tensor_data.dtype),
            });
        }
    };
    Ok(tensor)
}
