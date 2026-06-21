use crate::{model::ModelConfig, tensor::Tensor};

pub struct KVCache {
    keys: Vec<Tensor>,
    values: Vec<Tensor>,
}

impl KVCache {
    pub fn new(config: &ModelConfig, max_seq_len: usize) -> Self {
        let n_layers = config.n_layers;
        let n_kv_heads = config.n_kv_heads;
        let head_dim = config.head_dim;
        
        let mut keys = Vec::with_capacity(n_layers);
        let mut values = Vec::with_capacity(n_layers);

        for _ in 0..n_layers {
            keys.push(Tensor::zeros(vec![max_seq_len, n_kv_heads, head_dim]));
            values.push(Tensor::zeros(vec![max_seq_len, n_kv_heads, head_dim]));
        }
        Self { keys, values }
    }

    pub fn update(&mut self, config: &ModelConfig, layer: usize, pos: usize, k:&Tensor, v: &Tensor) {
        let n_kv_heads = config.n_kv_heads;
        let head_dim = config.head_dim;

        for h in 0..n_kv_heads {
            for d in 0..head_dim {
                let flat_idx = pos * (n_kv_heads * head_dim) + h * head_dim + d;
                self.keys[layer].data_mut()[flat_idx] = k[(h, d)];
                self.values[layer].data_mut()[flat_idx] = v[(h, d)];
            }
        }
    }

    pub fn get_keys_up_to(&self, layer: usize, seq_len: usize, config: &ModelConfig) -> Tensor {
        let n_kv_heads = config.n_kv_heads;
        let head_dim = config.head_dim;

        let elements_per_position = n_kv_heads * head_dim;
        let total_elements = seq_len * elements_per_position;

        let sliced_data: Vec<f32> = self.keys[layer].data()[0..total_elements].to_vec();

        Tensor::from_vec(sliced_data, vec![seq_len, n_kv_heads, head_dim])
    }

    pub fn get_values_up_to(&self, layer: usize, seq_len: usize, config: &ModelConfig) -> Tensor {
        let n_kv_heads = config.n_kv_heads;
        let head_dim = config.head_dim;

        let elements_per_position = n_kv_heads * head_dim;
        let total_elements = seq_len * elements_per_position;

        let sliced_data: Vec<f32> = self.values[layer].data()[0..total_elements].to_vec();

        Tensor::from_vec(sliced_data, vec![seq_len, n_kv_heads, head_dim])
    }
}