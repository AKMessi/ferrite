use half::f16;

use crate::tensor::Tensor;


pub fn dequant_q8_0(bytes: &[u8], shape: Vec<usize>) -> Tensor {
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

pub fn dequant_q4_k(bytes: &[u8], shape: Vec<usize>) -> Tensor {
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

pub fn dequant_q6_k(bytes: &[u8], shape: Vec<usize>) -> Tensor {
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
                let q3 = (((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i8) - 32;
                let q4 =
                    (((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i8) - 32;

                data.push(d * (sc[is] as f32) * (q1 as f32));
                data.push(d * (sc[is + 2] as f32) * (q2 as f32));
                data.push(d * (sc[is + 4] as f32) * (q3 as f32));
                data.push(d * (sc[is + 6] as f32) * (q4 as f32));
            }
        }
    }

    Tensor::from_vec(data, shape)
}

pub fn dequant_q5_k(bytes: &[u8], shape: Vec<usize>) -> Tensor {
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

pub fn dequant_iq4_xs(bytes: &[u8], shape: Vec<usize>) -> Tensor {
    assert_eq!(bytes.len() % 136, 0, "IQ4_XS data size must be multiple of 136 bytes");

    const KVALUES_IQ4NL: [i8; 16] = [
        -127, -104, -83, -65, -49, -35, -22, -10,
        1,   13,  25,  38,  53,  69,  89,  113
    ];

    let mut data = Vec::with_capacity(shape.iter().product());

    for block in bytes.chunks_exact(136) {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let scales_h = u16::from_le_bytes([block[2], block[3]]);
        let scales_l = &block[4..8];
        let qs = &block[8..136];

        let mut sub_scales = [0i32; 8];
                for j in 0..8 {
                    let low = if j % 2 == 0 {
                        scales_l[j / 2] & 0x0F
                    } else {
                        scales_l[j / 2] >> 4
                    };
                    let high = (scales_h >> (2 * j)) & 0x03;
                    let six_bit = (low as i32) | ((high as i32) << 4);
                    sub_scales[j] = six_bit - 32;
                }

                for j in 0..8 {
                    let scale = d * sub_scales[j] as f32;
                    let qs_off = j * 16;

                    for i in 0..16 {
                        let byte = qs[qs_off + i];
                        let lo = byte & 0x0F;
                        let hi = byte >> 4;

                        data.push(scale * KVALUES_IQ4NL[lo as usize] as f32);
                        data.push(scale * KVALUES_IQ4NL[hi as usize] as f32);
                    }
                }
            }

            Tensor::from_vec(data, shape)
}

