#![allow(dead_code)]

use crate::gguf::{GGUFError, GGUFFile, parse};
use crate::quant;
use half::f16;
use memmap2::{Mmap, MmapOptions};
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Cursor;
use std::ops::{Index, IndexMut};
use std::{fs::File, path::Path};

pub struct WeightStore {
    pub mmap: Mmap,
    pub gguf: GGUFFile,
    cache: RefCell<HashMap<String, Tensor>>,
}

#[derive(Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl WeightStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GGUFError> {
        let file = File::open(path)?;

        let mmap = unsafe { MmapOptions::new().map(&file)? };

        let mut cursor = Cursor::new(&mmap[..]);
        let gguf = parse(&mut cursor)?;

        Ok(Self {
            mmap,
            gguf,
            cache: RefCell::new(HashMap::new()),
        })
    }

    pub fn get_bytes(&self, name: &str) -> Result<&[u8], GGUFError> {
        let info = self
            .gguf
            .get_tensor(name)
            .ok_or(GGUFError::TensorNotFound)?;

        let start = info.file_offset as usize;
        let n = info.shape.iter().product::<u64>() as usize;

        let byte_size: usize = match info.ggml_type {
            0 => n * 4,                    // F32
            1 => n * 2,                    // F16
            8 => ((n + 31) / 32) * 34,     // Q8_0
            10 => ((n + 255) / 256) * 84,  // Q2_K
            12 => ((n + 255) / 256) * 144, // Q4_K
            13 => ((n + 255) / 256) * 176, // Q5_K
            14 => ((n + 255) / 256) * 210, // Q6_K
            16 => ((n + 255) / 256) * 66,  // IQ2_XXS
            18 => ((n + 255) / 256) * 98,  // IQ3_XXS
            21 => ((n + 255) / 256) * 110, // IQ3_S
            22 => ((n + 255) / 256) * 82,  // IQ2_S
            23 => ((n + 255) / 256) * 136, // IQ4_XS
            24 => n,                       // I8
            25 => n * 2,                   // I16
            26 => n * 4,                   // I32
            27 => n * 8,                   // I64
            28 => n * 8,                   // F64
            30 => n * 2,                   // BF16
            _ => return Err(GGUFError::UnknownMetadataType(info.ggml_type)),
        };

        let end = start + byte_size;
        Ok(&self.mmap[start..end])
    }

    pub fn load(&self, name: &str) -> Result<Tensor, GGUFError> {
        if let Some(cached) = self.cache.borrow().get(name) {
            return Ok(cached.clone());
        }

        let info = self
            .gguf
            .get_tensor(name)
            .ok_or(GGUFError::TensorNotFound)?;

        let bytes = self.get_bytes(name)?;

        let shape: Vec<usize> = info.shape.iter().map(|&x| x as usize).collect();

        let tensor = match info.ggml_type {
            0 => unsafe { Tensor::load_f32(bytes, shape) },

            1 => Tensor::load_f16(bytes, shape),

            8 => quant::dequant_q8_0(bytes, shape),
            10 => quant::dequant_q2_k(bytes, shape),
            12 => quant::dequant_q4_k(bytes, shape),

            13 => quant::dequant_q5_k(bytes, shape),
            14 => quant::dequant_q6_k(bytes, shape),
            16 => quant::dequant_iq2_xxs(bytes, shape),
            18 => quant::dequant_iq3_xxs(bytes, shape),
            21 => quant::dequant_iq3_s(bytes, shape),
            22 => quant::dequant_iq2_s(bytes, shape),
            23 => quant::dequant_iq4_xs(bytes, shape),
            30 => Tensor::load_bf16(bytes, shape),
            t => todo!("ggml type {} not yet implemented", t),
        };

        self.cache
            .borrow_mut()
            .insert(name.to_string(), tensor.clone());

        Ok(tensor)
    }
}

impl Tensor {
    pub fn zeros(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();

        let data = vec![0.0f32; numel];

        Self { data, shape }
    }

    pub fn clamp(&self, min: f32, max: f32) -> Tensor {
        let new_data: Vec<f32> = self.data.iter().map(|&x| x.clamp(min, max)).collect();
        Self {
            data: new_data,
            shape: self.shape.clone(),
        }
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "shape mismatch for subtraction");
        let new_data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Self {
            data: new_data,
            shape: self.shape.clone(),
        }
    }

    pub fn outer(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.rank(), 1, "outer requires rank-1 self");
        assert_eq!(other.rank(), 1, "outer requires rank-1 other");

        let n = self.shape()[0];
        let m = other.shape()[0];

        let mut out = Tensor::zeros(vec![n, m]);

        for i in 0..n {
            for j in 0..m {
                out[(i, j)] = self.data()[i] * other.data()[j];
            }
        }

        out
    }

    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();

        assert_eq!(
            data.len(),
            numel,
            "Shape mismatch: data length {}, but expected {} based on shape {:?}",
            data.len(),
            numel,
            shape
        );

        Self { data, shape }
    }

    pub fn numel(&self) -> usize {
        let numel: usize = self.shape.iter().product();
        numel
    }

    pub fn rank(&self) -> usize {
        let rank: usize = self.shape.len();
        rank
    }

    pub fn reshape(mut self, new_shape: Vec<usize>) -> Self {
        assert_eq!(
            self.numel(),
            new_shape.iter().product::<usize>(), // ← compare to NEW shape
            "Cannot reshape: {} elements into shape {:?}",
            self.numel(),
            new_shape
        );

        self.shape = new_shape;
        self
    }

    pub fn transpose(&self) -> Self {
        assert_eq!(
            self.rank(),
            2,
            "Transpose is only valid for rank-2 tensors. Current rank: {}",
            self.rank()
        );

        let rows = self.shape[0];
        let cols = self.shape[1];

        let new_shape = vec![cols, rows];

        let mut out_data = vec![0.0f32; self.numel()];

        for i in 0..rows {
            for j in 0..cols {
                out_data[j * rows + i] = self.data[i * cols + j];
            }
        }

        Self {
            data: out_data,
            shape: new_shape,
        }
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let (m, k) = (self.shape[0], self.shape[1]);
        let n = other.shape[1];

        assert_eq!(k, other.shape[0], "matmul shape mismatch");

        let mut out = Tensor::zeros(vec![m, n]);

        for i in 0..m {
            for j in 0..n {
                for k in 0..k {
                    out[(i, j)] += self[(i, k)] * other[(k, j)];
                }
            }
        }
        out
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.shape, other.shape,
            "Shape mismatch for element-wise addition: {:?} vs {:?}",
            self.shape, other.shape
        );

        // self.data.iter().zip(other.data.iter()).map(|(a,b)| a+b).collect()
        let new_data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        // Return new Tensor with same shape
        Self {
            data: new_data,
            shape: self.shape.clone(),
        }
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.shape, other.shape,
            "Shape mismatch for element-wise multiplication: {:?} vs {:?}",
            self.shape, other.shape
        );

        // self.data.iter().zip(other.data.iter()).map(|(a,b)| a+b).collect()
        let new_data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();

        // Return new Tensor with same shape
        Self {
            data: new_data,
            shape: self.shape.clone(),
        }
    }

    pub fn scale(&self, s: f32) -> Tensor {
        let new_data: Vec<f32> = self.data.iter().map(|x| x * s).collect();

        Self {
            data: new_data,
            shape: self.shape.clone(),
        }
    }

    pub fn softmax(&self) -> Tensor {
        // find the maximum value for numerical stability
        let max_val = self.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // compute exp(x - max)
        let exps: Vec<f32> = self.data.iter().map(|&x| (x - max_val).exp()).collect();

        // compute sum(exp(x - max))
        let sum_exps: f32 = exps.iter().sum();

        // normalise elements
        let new_data = exps.iter().map(|&exp_val| exp_val / sum_exps).collect();

        Self {
            data: new_data,
            shape: self.shape.clone(),
        }
    }

    pub fn rms_norm(&self, weight: &Tensor, eps: f32) -> Tensor {
        assert_eq!(
            self.shape, weight.shape,
            "Weight shape {:?} must match input shape {:?}",
            weight.shape, self.shape
        );

        // compute mean(x²)
        let sum_sq: f32 = self.data.iter().map(|&x| x * x).sum();
        let mean_sq = sum_sq / (self.numel() as f32);

        // rms = sqrt(mean(x²) + eps)
        let rms = (mean_sq + eps).sqrt();

        // divide each element by RMS, then multiply by corresponding weight element
        let new_data = self
            .data
            .iter()
            .zip(weight.data.iter())
            .map(|(&x, &w)| (x / rms) * w)
            .collect();

        Self {
            data: new_data,
            shape: self.shape.clone(),
        }
    }

    pub fn silu(&self) -> Tensor {
        // silu activation: x * sigmoid(x) = x / (1 + exp(-x))
        let new_data = self.data.iter().map(|&x| x / (1.0 + (-x).exp())).collect();

        Self {
            data: new_data,
            shape: self.shape.clone(),
        }
    }

    pub fn sigmoid(&self) -> Tensor {
        let new_data = self
            .data
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        Self {
            data: new_data,
            shape: self.shape.clone(),
        }
    }

    pub fn softplus(&self) -> Tensor {
        let new_data = self
            .data
            .iter()
            .map(|&x| if x > 20.0 { x } else { (1.0 + x.exp()).ln() })
            .collect();
        Self {
            data: new_data,
            shape: self.shape.clone(),
        }
    }

    pub unsafe fn load_f32(bytes: &[u8], shape: Vec<usize>) -> Tensor {
        let ptr = bytes.as_ptr();
        assert_eq!(
            ptr as usize % std::mem::align_of::<f32>(),
            0,
            "Memory slice is not aligned for f32 (4-byte alignment required)"
        );

        let total_elements = bytes.len() / 4;
        let f32_slice = unsafe { std::slice::from_raw_parts(ptr as *const f32, total_elements) };

        let data = f32_slice.to_vec();

        Self::from_vec(data, shape)
    }

    pub fn load_f16(bytes: &[u8], shape: Vec<usize>) -> Tensor {
        assert_eq!(
            bytes.len() % 2,
            0,
            "f16 data length must be a multiple of 2 bytes"
        );

        let data: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|chunk| {
                let array = [chunk[0], chunk[1]];
                f16::from_le_bytes(array).to_f32()
            })
            .collect();

        Self::from_vec(data, shape)
    }

    pub fn load_bf16(bytes: &[u8], shape: Vec<usize>) -> Tensor {
        assert_eq!(
            bytes.len() % 2,
            0,
            "bf16 data length must be a multiple of 2 bytes"
        );

        let data: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|chunk| {
                let bits = ((chunk[1] as u32) << 24) | ((chunk[0] as u32) << 16);
                f32::from_bits(bits)
            })
            .collect();

        Self::from_vec(data, shape)
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn matvec(&self, x: &Tensor) -> Tensor {
        assert_eq!(self.rank(), 2, "matvec requires a rank-2 weight matrix");
        assert_eq!(x.rank(), 1, "matvec requires a rank-1 input vector");
        assert_eq!(self.shape()[1], x.shape()[0], "matvec dimension mismatch");

        let out_dim = self.shape()[0];
        let in_dim = self.shape()[1];
        let mut out = vec![0.0f32; out_dim];

        for i in 0..out_dim {
            let mut sum = 0.0;
            for j in 0..in_dim {
                sum += self[(i, j)] * x.data()[j];
            }
            out[i] = sum;
        }

        Tensor::from_vec(out, vec![out_dim])
    }
}

impl Index<(usize, usize)> for Tensor {
    type Output = f32;

    fn index(&self, (i, j): (usize, usize)) -> &f32 {
        assert_eq!(self.shape.len(), 2, "Index works only for rank-2 tensors");
        assert!(
            i < self.shape[0],
            "Row index out of bounds: {} >= {}",
            i,
            self.shape[0]
        );
        assert!(
            j < self.shape[1],
            "Col index out of bounds: {} >= {}",
            j,
            self.shape[1]
        );

        &self.data[i * self.shape[1] + j]
    }
}

impl IndexMut<(usize, usize)> for Tensor {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f32 {
        assert_eq!(self.shape.len(), 2, "Index works only for rank-2 tensors");
        assert!(
            i < self.shape[0],
            "Row index out of bounds: {} >= {}",
            i,
            self.shape[0]
        );
        assert!(
            j < self.shape[1],
            "Col index out of bounds: {} >= {}",
            j,
            self.shape[1]
        );

        &mut self.data[i * self.shape[1] + j]
    }
}

// tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_zeros() {
        // Create Tensor::zeros(vec![3,4])
        let tensor = Tensor::zeros(vec![3, 4]);

        // Assert numel() == 12
        assert_eq!(tensor.numel(), 12);

        // Assert rank() == 2
        assert_eq!(tensor.rank(), 2);

        // Assert data.len() == 12
        assert_eq!(tensor.data.len(), 12);
    }
}

#[test]
fn test_tensor_transpose() {
    // Create Tensor::from_vec(vec!, vec!)
    let tensor = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);

    // Transpose it
    let transposed = tensor.transpose();

    // Assert shape is [3, 2]
    assert_eq!(transposed.shape, vec![3, 2]);

    // Assert data is [1., 4., 2., 5., 3., 6.]
    assert_eq!(transposed.data, vec![1., 4., 2., 5., 3., 6.]);
}

#[test]
fn test_tensor_matmul() {
    // Create matrix A [2,3]
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    // Create matrix B [3,2]
    let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);

    // Compute result (assumes you have a matmul method or Add/Mul implementation)
    let result = a.matmul(&b);

    // Expected output from NumPy: [[58.0, 64.0], [139.0, 154.0]]
    let expected_shape = vec![2, 2];
    let expected_data = vec![58.0, 64.0, 139.0, 154.0];

    // Assert shape matches
    assert_eq!(result.shape, expected_shape);

    // Assert every element matches to 4 decimal places
    for i in 0..result.data.len() {
        let diff = (result.data[i] - expected_data[i]).abs();
        assert!(
            diff < 1e-4,
            "Value mismatch at index {}: got {}, expected {} (diff: {})",
            i,
            result.data[i],
            expected_data[i],
            diff
        );
    }
}

#[test]
fn test_outer() {
    let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    let b = Tensor::from_vec(vec![3.0, 4.0, 5.0], vec![3]);
    let result = a.outer(&b);
    assert_eq!(result.shape(), &[2, 3]);
    // result[(0,0)] = 1*3=3, result[(1,2)] = 2*5=10
    assert_eq!(result.data(), &[3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
}
