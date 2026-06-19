#![allow(dead_code)]

use crate::gguf::{GGUFError, GGUFFile, parse};
use half::f16;
use memmap2::{Mmap, MmapOptions};
use std::io::Cursor;
use std::ops::{Index, IndexMut};
use std::{fs::File, path::Path};

pub struct WeightStore {
    pub mmap: Mmap,
    pub gguf: GGUFFile,
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

        Ok(Self { mmap, gguf })
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
            12 => ((n + 255) / 256) * 144, // Q4_K
            13 => ((n + 255) / 256) * 176, // Q5_K
            14 => ((n + 255) / 256) * 210, // Q6_K
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

    fn dequant_q8_0(bytes: &[u8], shape: Vec<usize>) -> Tensor {
        assert_eq!(
            bytes.len() % 34,
            0,
            "Q8_0 data size must be a multiple of 34 bytes"
        );

        let mut data = Vec::with_capacity(shape.iter().product());

        for block in bytes.chunks_exact(34) {
            let scale_bytes = [block[0], block[1]];
            let scale = f16::from_le_bytes(scale_bytes).to_f32();

            for i in 0..32 {
                let quant_val = block[2 + i] as i8;
                data.push(quant_val as f32 * scale);
            }
        }

        Tensor::from_vec(data, shape)
    }

    fn dequant_q4_k(bytes: &[u8], shape: Vec<usize>) -> Tensor {
        assert_eq!(
            bytes.len() % 144,
            0,
            "Q4_K data size must be a multiple of 144 bytes"
        );
        let mut data = Vec::with_capacity(shape.iter().product());

        for block in bytes.chunks_exact(144) {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();

            let scales = &block[4..16];
            let qs = &block[16..144];

            let mut q_idx = 0;
            let mut is = 0;

            for _j in (0..256).step_by(64) {
                let (sc1, m1) = if is < 4 {
                    let sc = scales[is] & 63;
                    let m = scales[is + 4] & 63;
                    (sc, m)
                } else {
                    let sc = scales[is] & 63;
                    let m = scales[is - 4] >> 6;
                    (sc, m)
                };
                let d1 = d * (sc1 as f32);
                let m1 = dmin * (m1 as f32);

                let (sc2, m2) = if (is + 1) < 4 {
                    let sc = scales[is + 1] & 63;
                    let m = scales[is + 1 + 4] & 63;
                    (sc, m)
                } else {
                    let sc = scales[is + 1] & 63;
                    let m = scales[is + 1 - 4] >> 6;
                    (sc, m)
                };
                let d2 = d * (sc2 as f32);
                let m2 = dmin * (m2 as f32);

                for l in 0..32 {
                    let q_val = (qs[q_idx + l] & 0x0F) as f32;
                    data.push(d1 * q_val - m1);
                }

                for l in 0..32 {
                    let q_val = (qs[q_idx + l] >> 4) as f32;
                    data.push(d2 * q_val - m2);
                }

                q_idx += 32;
                is += 2;
            }
        }
        Tensor::from_vec(data, shape)
    }

    fn dequant_q6_k(bytes: &[u8], shape: Vec<usize>) -> Tensor {
        assert_eq!(
            bytes.len() % 210,
            0,
            "Q6_K data size must be a multiple of 210 bytes"
        );
        let mut data = Vec::with_capacity(shape.iter().product());

        for block in bytes.chunks_exact(210) {
            let ql = &block[0..128];
            let qh = &block[128..192];
            let scales = &block[192..208];
            let d = f16::from_le_bytes([block[208], block[209]]).to_f32(); // ← last 2 bytes

            let mut sc = [0i8; 16];
            for i in 0..16 {
                sc[i] = scales[i] as i8;
            }

            for half in 0..2 {
                let ql_off = half * 64;
                let qh_off = half * 32;
                let sc_off = half * 8;

                for l in 0..32 {
                    let is = sc_off + (l / 16);

                    let q1 =
                        (((ql[ql_off + l] & 0x0F) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i8) - 32;
                    let q2 = (((ql[ql_off + l + 32] & 0x0F) | (((qh[qh_off + l] >> 2) & 3) << 4))
                        as i8)
                        - 32;
                    let q3 =
                        (((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i8) - 32;
                    let q4 = (((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4))
                        as i8)
                        - 32;

                    data.push(d * (sc[is] as f32) * (q1 as f32));
                    data.push(d * (sc[is + 2] as f32) * (q2 as f32));
                    data.push(d * (sc[is + 4] as f32) * (q3 as f32));
                    data.push(d * (sc[is + 6] as f32) * (q4 as f32));
                }
            }
        }

        Tensor::from_vec(data, shape)
    }

    fn dequant_q5_k(bytes: &[u8], shape: Vec<usize>) -> Tensor {
        assert_eq!(
            bytes.len() % 176,
            0,
            "Q5_K data size must be a multiple of 176 bytes"
        );
        let mut data = Vec::with_capacity(shape.iter().product());

        for block in bytes.chunks_exact(176) {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();

            let scales = &block[4..16]; // 12 bytes
            let qh = &block[16..48]; // 32 bytes — high bits
            let qs = &block[48..176]; // 128 bytes — low 4 bits

            // unpack 8 sub-block scales and mins from 12 bytes (6 bits each)
            let mut sc = [0u8; 8];
            let mut mn = [0u8; 8];
            sc[0] = scales[0] & 0x3F;
            sc[1] = scales[1] & 0x3F;
            sc[2] = scales[2] & 0x3F;
            sc[3] = scales[3] & 0x3F;
            sc[4] = (scales[0] >> 6) | ((scales[4] & 0x0F) << 2);
            sc[5] = (scales[1] >> 6) | ((scales[5] & 0x0F) << 2);
            sc[6] = (scales[2] >> 6) | ((scales[6] & 0x0F) << 2);
            sc[7] = (scales[3] >> 6) | ((scales[7] & 0x0F) << 2);
            mn[0] = scales[4] & 0x3F;
            mn[1] = scales[5] & 0x3F;
            mn[2] = scales[6] & 0x3F;
            mn[3] = scales[7] & 0x3F;
            mn[4] = (scales[4] >> 6) | ((scales[8] & 0x0F) << 2);
            mn[5] = (scales[5] >> 6) | ((scales[9] & 0x0F) << 2);
            mn[6] = (scales[6] >> 6) | ((scales[10] & 0x0F) << 2);
            mn[7] = (scales[7] >> 6) | ((scales[11] & 0x0F) << 2);

            // 256 values in 8 sub-blocks of 32
            for j in 0..8usize {
                let d_sc = d * sc[j] as f32;
                let d_mn = dmin * mn[j] as f32;
                let qs_off = j * 16; // 16 bytes of qs per sub-block (32 values × 4 bits / 8)
                let qh_off = j * 4; // 4 bytes of qh per sub-block  (32 values × 1 bit  / 8)

                for l in 0..16usize {
                    // each qs byte holds two 4-bit low values
                    let lo0 = qs[qs_off + l] & 0x0F;
                    let lo1 = qs[qs_off + l] >> 4;

                    // qh byte index: l + qh_off, but we need bit positions within the byte
                    // 32 values packed as 1 bit each across 4 bytes
                    // value index within sub-block: l*2 and l*2+1
                    let bit0 = (l * 2) % 8;
                    let bit1 = (l * 2 + 1) % 8;
                    let qh_byte0 = qh[qh_off + (l * 2) / 8];
                    let qh_byte1 = qh[qh_off + (l * 2 + 1) / 8];

                    let hi0 = (qh_byte0 >> bit0) & 1;
                    let hi1 = (qh_byte1 >> bit1) & 1;

                    let q0 = (lo0 | (hi0 << 4)) as f32;
                    let q1 = (lo1 | (hi1 << 4)) as f32;

                    data.push(d_sc * q0 - d_mn);
                    data.push(d_sc * q1 - d_mn);
                }
            }
        }

        Tensor::from_vec(data, shape)
    }

    pub fn load(&self, name: &str) -> Result<Tensor, GGUFError> {
        let info = self
            .gguf
            .get_tensor(name)
            .ok_or(GGUFError::TensorNotFound)?;

        let bytes = self.get_bytes(name)?;

        let shape: Vec<usize> = info.shape.iter().map(|&x| x as usize).collect();

        let tensor = match info.ggml_type {
            0 => unsafe { Tensor::load_f32(bytes, shape) },

            1 => Tensor::load_f16(bytes, shape),

            8 => Self::dequant_q8_0(bytes, shape),

            12 => Self::dequant_q4_k(bytes, shape),

            13 => Self::dequant_q5_k(bytes, shape),
            14 => Self::dequant_q6_k(bytes, shape),

            23 => todo!("ggml type 23 (IQ4_XS) not yet implemented"),
            t => todo!("ggml type {} not yet implemented", t),
        };

        Ok(tensor)
    }
}

impl Tensor {
    pub fn zeros(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();

        let data = vec![0.0f32; numel];

        Self { data, shape }
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

    pub fn data(&self) -> &[f32] {
        &self.data
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
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
