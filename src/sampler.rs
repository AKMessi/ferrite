use rand::distr::{Distribution, weighted::WeightedIndex};

use crate::tensor::Tensor;

pub fn greedy(logits: &Tensor) -> u32 {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;

    for (i, &val) in logits.data().iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = i;
        }
    }

    best_idx as u32
}

pub fn apply_temperature(logits: &Tensor, temp: f32) -> Tensor {
    logits.scale(1.0 / temp)
}

pub fn top_k_filter(probs: Vec<(u32, f32)>, k: usize) -> Vec<(u32, f32)> {
    probs.into_iter().take(k).collect()
}

pub fn top_p_filter(probs: Vec<(u32, f32)>, p: f32) -> Vec<(u32, f32)> {
    let mut result = Vec::new();
    let mut cumulative = 0.0;

    for (id, prob) in probs {
        result.push((id, prob));
        cumulative += prob;
        if cumulative >= p {
            break;
        }
    }
    result
}

pub fn sample(logits: &Tensor, temp: f32, top_k: usize, top_p: f32) -> u32 {
    let scaled = apply_temperature(logits, temp);
    let probs_tensor = scaled.softmax();

    let mut probs: Vec<(u32, f32)> = probs_tensor
        .data()
        .iter()
        .enumerate()
        .map(|(i, &p)| (i as u32, p))
        .collect();

    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let probs = top_k_filter(probs, top_k);
    let probs = top_p_filter(probs, top_p);

    let total: f32 = probs.iter().map(|(_, p)| p).sum();

    let normalized: Vec<(u32, f32)> = probs.into_iter().map(|(id, p)| (id, p / total)).collect();

    let weights: Vec<f32> = normalized.iter().map(|(_, p)| *p).collect();
    let dist = WeightedIndex::new(weights).unwrap();
    let mut rng = rand::rng();
    let chosen_idx = dist.sample(&mut rng);

    normalized[chosen_idx].0
}

// tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_picks_max() {
        let logits = Tensor::from_vec(vec![0.1, 0.5, 3.0, -1.0, 0.2], vec![5]);
        assert_eq!(greedy(&logits), 2); // index 2 has the highest value, 3.0
    }

    #[test]
    fn test_temperature_scaling() {
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let scaled = apply_temperature(&logits, 2.0);
        assert_eq!(scaled.data(), &[0.5, 1.0, 1.5]);
    }

    #[test]
    fn test_top_k_filter() {
        let probs = vec![(0u32, 0.5), (1u32, 0.3), (2u32, 0.2)];
        let filtered = top_k_filter(probs, 2);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].0, 0);
        assert_eq!(filtered[1].0, 1);
    }

    #[test]
    fn test_top_p_filter() {
        let probs = vec![(0u32, 0.5), (1u32, 0.3), (2u32, 0.15), (3u32, 0.05)];
        let filtered = top_p_filter(probs, 0.8);
        // 0.5 + 0.3 = 0.8, crosses threshold exactly at index 1
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_sample_with_greedy_equivalent() {
        // top_k=1 should always behave like greedy
        let logits = Tensor::from_vec(vec![0.1, 0.5, 3.0, -1.0, 0.2], vec![5]);
        let sampled = sample(&logits, 1.0, 1, 1.0);
        assert_eq!(sampled, 2);
    }
}
