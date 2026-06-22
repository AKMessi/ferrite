#[allow(dead_code)]
use crate::cache::KVCache;
use crate::{
    cache::SSMState,
    gguf::{GGUFFile, MetadataValue},
    tensor::{Tensor, WeightStore},
};

pub struct ModelConfig {
    pub n_layers: usize,
    embed_dim: usize,
    n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    ffn_dim: usize,
    vocab_size: usize,
    context_len: usize,
    rope_theta: f32,
    rms_eps: f32,
    rope_dim: usize,
    pub ssm_group_count: usize,
    pub ssm_state_size: usize,
}

pub struct Model {
    pub config: ModelConfig,
    weights: WeightStore,
}

impl ModelConfig {
    pub fn from_gguf(gguf: &GGUFFile) -> Self {
        let n_layers = gguf
            .metadata
            .get("qwen35.block_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(24) as usize;

        let embed_dim = gguf
            .metadata
            .get("qwen35.embedding_length")
            .and_then(|v| v.as_u64())
            .unwrap_or(1024) as usize;

        let n_heads = gguf
            .metadata
            .get("qwen35.attention.head_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(8) as usize;

        let n_kv_heads = gguf
            .metadata
            .get("qwen35.attention.head_count_kv")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;

        let head_dim = gguf
            .metadata
            .get("qwen35.attention.value_length")
            .and_then(|v| v.as_u64())
            .unwrap_or(256) as usize;

        let ffn_dim = gguf
            .metadata
            .get("qwen35.feed_forward_length")
            .and_then(|v| v.as_u64())
            .unwrap_or(3584) as usize;

        let vocab_size = gguf
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                MetadataValue::Array(arr) => Some(arr.len()),
                _ => None,
            })
            .unwrap_or(248320);

        let context_len = gguf
            .metadata
            .get("qwen35.context_length")
            .and_then(|v| v.as_u64())
            .unwrap_or(262144) as usize;

        let rope_theta = gguf
            .metadata
            .get("qwen35.rope.freq_base")
            .and_then(|v| v.as_f32())
            .unwrap_or(10_000_000.0);

        let rms_eps = gguf
            .metadata
            .get("qwen35.attention.layer_norm_rms_epsilon")
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-6);

        let rope_dim = gguf
            .metadata
            .get("qwen35.rope.dimension_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(64) as usize;

        let ssm_group_count = gguf
            .metadata
            .get("qwen35.ssm.group_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(16) as usize;

        let ssm_state_size = gguf
            .metadata
            .get("qwen35.ssm.state_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize;

        Self {
            n_layers,
            embed_dim,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_dim,
            vocab_size,
            context_len,
            rope_theta,
            rms_eps,
            rope_dim,
            ssm_group_count,
            ssm_state_size,
        }
    }
}

pub fn rope(q: &Tensor, k: &Tensor, pos: usize, config: &ModelConfig) -> (Tensor, Tensor) {
    let rope_dim = config.rope_dim;
    let theta = config.rope_theta;
    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;

    let mut q_out = q.clone();
    let mut k_out = k.clone();

    for i in 0..(rope_dim / 2) {
        let idx_0 = 2 * i;
        let idx_1 = 2 * i + 1;
        let exponent = (idx_0 as f32) / (rope_dim as f32);
        let angle = (pos as f32) / theta.powf(exponent);
        let cos = angle.cos();
        let sin = angle.sin();

        // apply rotation to every Q head
        for h in 0..n_heads {
            let q0 = q[(h, idx_0)];
            let q1 = q[(h, idx_1)];
            q_out[(h, idx_0)] = q0 * cos - q1 * sin;
            q_out[(h, idx_1)] = q0 * sin + q1 * cos;
        }

        // apply rotation to every K head (fewer heads — GQA)
        for h in 0..n_kv_heads {
            let k0 = k[(h, idx_0)];
            let k1 = k[(h, idx_1)];
            k_out[(h, idx_0)] = k0 * cos - k1 * sin;
            k_out[(h, idx_1)] = k0 * sin + k1 * cos;
        }
    }

    (q_out, k_out)
}

impl Model {
    pub fn new(config: ModelConfig, weights: WeightStore) -> Self {
        Self { config, weights }
    }

    pub fn embed(&self, token_id: u32) -> Tensor {
        let embd_table = self.weights.load("token_embd.weight").unwrap();
        let embed_dim = self.config.embed_dim;

        let row_start = (token_id as usize) * embed_dim;
        let row_end = row_start + embed_dim;

        let row_data: Vec<f32> = embd_table.data()[row_start..row_end].to_vec();

        Tensor::from_vec(row_data, vec![embed_dim])
    }

    pub fn forward(
        &self,
        tokens: &[u32],
        pos: usize,
        cache: &mut KVCache,
        ssm_state: &mut SSMState,
    ) -> Tensor {
        // 1. embed the current token (you have self.embed() already)
        let mut x = self.embed(tokens[pos]);

        // 2. pass through all 24 blocks, routing by block index
        for layer in 0..self.config.n_layers {
            let is_attention = (layer + 1) % 4 == 0;
            x = if is_attention {
                transformer_block(&x, layer, pos, self, cache)
            } else {
                let ssm_out = ssm_block(&x, layer, self, ssm_state);
                x.add(&ssm_out)
            };
        }

        // 3. final RMSNorm
        let final_norm_w = self.weights.load("output_norm.weight").unwrap();
        let x = x.rms_norm(&final_norm_w, self.config.rms_eps);

        // 4. project to vocab size via lm_head
        let embd_table = self.weights.load("token_embd.weight").unwrap();
        let logits = embd_table.matvec(&x);

        logits
    }
}

pub fn attention(
    x: &Tensor,
    layer: usize,
    pos: usize,
    model: &Model,
    cache: &mut KVCache,
) -> Tensor {
    let cfg = &model.config;
    let head_dim = model.config.head_dim;
    let n_heads = model.config.n_heads;
    let n_kv_heads = model.config.n_kv_heads;

    let q_w = model
        .weights
        .load(&format!("blk.{}.attn_q.weight", layer))
        .unwrap();
    let k_w = model
        .weights
        .load(&format!("blk.{}.attn_k.weight", layer))
        .unwrap();
    let v_w = model
        .weights
        .load(&format!("blk.{}.attn_v.weight", layer))
        .unwrap();
    let o_w = model
        .weights
        .load(&format!("blk.{}.attn_output.weight", layer))
        .unwrap();

    let q_proj_out = q_w.matvec(x);
    let k_proj_out = k_w.matvec(x);
    let v_proj_out = v_w.matvec(x);

    let q_data = q_proj_out.data();
    let q_raw_vec: Vec<f32> = q_data[0..2048].to_vec();
    let gate_raw_vec: Vec<f32> = q_data[2048..4096].to_vec();

    let q_norm_w = model
        .weights
        .load(&format!("blk.{}.attn_q_norm.weight", layer))
        .unwrap();
    let k_norm_w = model
        .weights
        .load(&format!("blk.{}.attn_k_norm.weight", layer))
        .unwrap();

    let mut q_heads: Vec<Tensor> = Vec::new();
    for h in 0..n_heads {
        let start = h * head_dim;
        let end = start + head_dim;
        let head_vec = Tensor::from_vec(q_raw_vec[start..end].to_vec(), vec![head_dim]);
        let normed = head_vec.rms_norm(&q_norm_w, cfg.rms_eps);
        q_heads.push(normed);
    }

    let k_data = k_proj_out.data();
    let mut k_heads: Vec<Tensor> = Vec::new();
    for h in 0..n_kv_heads {
        let start = h * head_dim;
        let end = start + head_dim;
        let head_vec = Tensor::from_vec(k_data[start..end].to_vec(), vec![head_dim]);
        let normed = head_vec.rms_norm(&k_norm_w, cfg.rms_eps);
        k_heads.push(normed);
    }
    // 1. Split V into heads (same pattern as K — no norm applied to V though)
    let v_data = v_proj_out.data();
    let mut v_heads: Vec<Tensor> = Vec::new();
    for h in 0..n_kv_heads {
        let start = h * head_dim;
        let end = start + head_dim;
        v_heads.push(Tensor::from_vec(
            v_data[start..end].to_vec(),
            vec![head_dim],
        ));
    }

    // 2. Apply RoPE to each Q head and each K head
    let mut q_stacked_data = Vec::new();
    for qh in &q_heads {
        q_stacked_data.extend_from_slice(qh.data());
    }
    let q_stacked = Tensor::from_vec(q_stacked_data, vec![n_heads, head_dim]);

    let mut k_stacked_data = Vec::new();
    for kh in &k_heads {
        k_stacked_data.extend_from_slice(kh.data());
    }
    let k_stacked = Tensor::from_vec(k_stacked_data, vec![n_kv_heads, head_dim]);

    let (q_rot, k_rot) = rope(&q_stacked, &k_stacked, pos, cfg);

    let mut v_stacked_data = Vec::new();
    for vh in &v_heads {
        v_stacked_data.extend_from_slice(vh.data())
    }
    let v_stacked = Tensor::from_vec(v_stacked_data, vec![n_kv_heads, head_dim]);

    cache.update(&model.config, layer, pos, &k_rot, &v_stacked);

    let all_keys = cache.get_keys_up_to(layer, pos + 1, &model.config);
    let all_values = cache.get_values_up_to(layer, pos + 1, &model.config);

    let group_size = n_heads / n_kv_heads;
    let mut head_outputs: Vec<Tensor> = Vec::new();

    for h in 0..n_heads {
        let kv_h = h / group_size;
        let q_h: Vec<f32> = (0..head_dim).map(|d| q_rot[(h, d)]).collect();

        // step 1
        let mut scores = vec![0.0f32; pos + 1];
        for t in 0..=pos {
            let mut dot = 0.0;
            for d in 0..head_dim {
                let flat_idx = t * (n_kv_heads * head_dim) + kv_h * head_dim + d;
                dot += q_h[d] * all_keys.data()[flat_idx];
            }
            scores[t] = dot / (head_dim as f32).sqrt();
        }
        // step 2
        let scores_tensor = Tensor::from_vec(scores, vec![pos + 1]);
        let weights = scores_tensor.softmax();

        // step 3
        let mut out_h = vec![0.0f32; head_dim];
        for t in 0..=pos {
            let w = weights.data()[t];
            for d in 0..head_dim {
                let flat_idx = t * (n_kv_heads * head_dim) + kv_h * head_dim + d;
                out_h[d] += w * all_values.data()[flat_idx];
            }
        }
        head_outputs.push(Tensor::from_vec(out_h, vec![head_dim]));
    }

    // 4. Concatenate head outputs -> [n_heads * head_dim] = [2048]
    let mut concat_data = Vec::new();
    for ho in &head_outputs {
        concat_data.extend_from_slice(ho.data());
    }
    let concat = Tensor::from_vec(concat_data, vec![n_heads * head_dim]);

    // 5. Apply the gate: sigmoid(gate_raw) elementwise on the concatenated output
    let gated_data: Vec<f32> = concat
        .data()
        .iter()
        .zip(gate_raw_vec.iter())
        .map(|(&c, &g)| c * (1.0 / (1.0 + (-g).exp())))
        .collect();
    let gated = Tensor::from_vec(gated_data, vec![n_heads * head_dim]);

    // 6. Output projection: [2048] -> [1024]
    let output = o_w.matvec(&gated);

    output
}

fn ffn(x: &Tensor, layer: usize, model: &Model) -> Tensor {
    let gate_w = model
        .weights
        .load(&format!("blk.{}.ffn_gate.weight", layer))
        .unwrap();
    let up_w = model
        .weights
        .load(&format!("blk.{}.ffn_up.weight", layer))
        .unwrap();
    let down_w = model
        .weights
        .load(&format!("blk.{}.ffn_down.weight", layer))
        .unwrap();

    // your turn — three calls:
    let silu_out = gate_w.matvec(x).silu();
    let up_out = up_w.matvec(x);
    let gated = silu_out.mul(&up_out);

    let down_out = down_w.matvec(&gated);

    down_out
}

pub fn transformer_block(
    x: &Tensor,
    layer: usize,
    pos: usize,
    model: &Model,
    cache: &mut KVCache,
) -> Tensor {
    let norm1_w = model
        .weights
        .load(&format!("blk.{}.attn_norm.weight", layer))
        .unwrap();
    let normed = x.rms_norm(&norm1_w, model.config.rms_eps);
    let attn_out = attention(&normed, layer, pos, model, cache);
    let x2 = x.add(&attn_out);

    let norm2_w = model
        .weights
        .load(&format!("blk.{}.post_attention_norm.weight", layer))
        .unwrap();
    let normed2 = x2.rms_norm(&norm2_w, model.config.rms_eps);
    let ffn_out = ffn(&normed2, layer, model);
    let x3 = x2.add(&ffn_out);

    x3
}

pub fn ssm_block(x: &Tensor, layer: usize, model: &Model, ssm_state: &mut SSMState) -> Tensor {
    let qkv_w = model
        .weights
        .load(&format!("blk.{}.attn_qkv.weight", layer))
        .unwrap();

    let gate_w = model
        .weights
        .load(&format!("blk.{}.attn_gate.weight", layer))
        .unwrap();

    // TODO Layer 5: conv1d needs a sliding window over past tokens (kernel size 4).
    // Skipped for now — single-token forward pass has no history to convolve over.
    let _conv_w = model
        .weights
        .load(&format!("blk.{}.ssm_conv1d.weight", layer))
        .unwrap();

    let a_param = model.weights.load(&format!("blk.{}.ssm_a", layer)).unwrap();

    let alpha_w = model
        .weights
        .load(&format!("blk.{}.ssm_alpha.weight", layer))
        .unwrap();

    let beta_w = model
        .weights
        .load(&format!("blk.{}.ssm_beta.weight", layer))
        .unwrap();

    let dt_bias = model
        .weights
        .load(&format!("blk.{}.ssm_dt.bias", layer))
        .unwrap();

    let norm_w = model
        .weights
        .load(&format!("blk.{}.ssm_norm.weight", layer))
        .unwrap();

    let out_w = model
        .weights
        .load(&format!("blk.{}.ssm_out.weight", layer))
        .unwrap();

    let ssm_group_count = model.config.ssm_group_count;
    let ssm_state_size = model.config.ssm_state_size;

    let mixed_qkv = qkv_w.matvec(x);
    let z = gate_w.matvec(x);

    let mixed_qkv = mixed_qkv.silu();

    let key_dim = ssm_group_count * ssm_state_size;
    let value_dim = key_dim;

    let mixed_data = mixed_qkv.data();

    let query: Vec<f32> = mixed_data[0..key_dim].to_vec();
    let key: Vec<f32> = mixed_data[key_dim..(2 * key_dim)].to_vec();
    let value: Vec<f32> = mixed_data[(2 * key_dim)..(2 * key_dim + value_dim)].to_vec();

    let query = Tensor::from_vec(query, vec![key_dim]);
    let key = Tensor::from_vec(key, vec![key_dim]);
    let value = Tensor::from_vec(value, vec![value_dim]);

    let alpha_out = alpha_w.matvec(x);
    let beta_out = beta_w.matvec(x);
    let beta = beta_out.sigmoid();

    let alpha_plus_dt = alpha_out.add(&dt_bias);
    let softplus_part = alpha_plus_dt.softplus();
    let exp_a = Tensor::from_vec(
        a_param.data().iter().map(|&v| v.exp()).collect(),
        a_param.shape().to_vec(),
    );
    let _g = exp_a.mul(&softplus_part).scale(-1.0);

    let mut output_data = vec![0.0f32; ssm_group_count * ssm_state_size];

    for grp in 0..ssm_group_count {
        let start = grp * ssm_state_size;
        let end = start + ssm_state_size;

        let q_g = Tensor::from_vec(query.data()[start..end].to_vec(), vec![ssm_state_size]);
        let k_g = Tensor::from_vec(key.data()[start..end].to_vec(), vec![ssm_state_size]);
        let v_g = Tensor::from_vec(value.data()[start..end].to_vec(), vec![ssm_state_size]);
        let beta_g = beta.data()[grp];
        let g_g = _g.data()[grp];

        let mut state = ssm_state.get_group_state(layer, grp, &model.config);
        let decay = g_g.exp();
        state = state.scale(decay);

        let predicted = state.matvec(&k_g);

        let delta = v_g.sub(&predicted);

        let correction = delta.outer(&k_g).scale(beta_g);

        state = state.add(&correction);

        ssm_state.set_group_state(layer, grp, &state, &model.config);

        let out_g = state.matvec(&q_g);

        output_data[start..end].copy_from_slice(out_g.data());
    }

    let output = Tensor::from_vec(output_data, vec![ssm_group_count * ssm_state_size]);

    let mut normed_data = vec![0.0f32; ssm_group_count * ssm_state_size];
    for grp in 0..ssm_group_count {
        let start = grp * ssm_state_size;
        let end = start + ssm_state_size;

        let group_slice =
            Tensor::from_vec(output.data()[start..end].to_vec(), vec![ssm_state_size]);
        let group_normed = group_slice.rms_norm(&norm_w, model.config.rms_eps);

        normed_data[start..end].copy_from_slice(group_normed.data());
    }
    let normed = Tensor::from_vec(normed_data, vec![ssm_group_count * ssm_state_size]);

    let gated = normed.mul(&z.silu());
    out_w.matvec(&gated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_defaults() {
        // sanity check that defaults are sane even without a real GGUF
        let cfg = ModelConfig {
            n_layers: 24,
            embed_dim: 1024,
            n_heads: 8,
            n_kv_heads: 2,
            head_dim: 256,
            ffn_dim: 3584,
            vocab_size: 248320,
            context_len: 262144,
            rope_theta: 10_000_000.0,
            rms_eps: 1e-6,
            rope_dim: 64,
            ssm_group_count: 16,
            ssm_state_size: 128,
        };
        assert_eq!(cfg.n_heads / cfg.n_kv_heads, 4); // GQA group size
        assert_eq!(cfg.ssm_group_count * cfg.ssm_state_size, 2048); // matches attn_qkv split
    }
}
