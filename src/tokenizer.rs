use std::collections::HashMap;

use crate::gguf::{GGUFFile, MetadataValue};

pub struct Tokenizer {
    vocab: Vec<String>,
    token_to_id: HashMap<String, u32>,
    merges: HashMap<(String, String), u32>,
}

impl Tokenizer {
    pub fn from_gguf(gguf: &GGUFFile) -> Self {
        let mut vocab = Vec::with_capacity(248_320);
        let mut token_to_id = HashMap::with_capacity(248_320);
        let mut merges = HashMap::new();

        if let Some(MetadataValue::Array(tokens_array)) = gguf.metadata.get("tokenizer.ggml.tokens")
        {
            for (id, item) in tokens_array.iter().enumerate() {
                if let MetadataValue::Str(token) = item {
                    vocab.push(token.clone());
                    token_to_id.insert(token.clone(), id as u32);
                }
            }
        }

        if let Some(MetadataValue::Array(merges_array)) = gguf.metadata.get("tokenizer.ggml.merges")
        {
            for (priority, item) in merges_array.iter().enumerate() {
                if let MetadataValue::Str(merge_str) = item {
                    let mut parts = merge_str.splitn(2, ' ');
                    if let (Some(left), Some(right)) = (parts.next(), parts.next()) {
                        let pair = (left.to_string(), right.to_string());
                        merges.insert(pair, priority as u32);
                    }
                }
            }
        }

        Tokenizer {
            vocab,
            token_to_id,
            merges,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // pre-tokenize: convert spaces to Ġ prefix on following word
        let pre = text.replace(' ', "Ġ");

        let mut tokens: Vec<u32> = pre
            .chars()
            .map(|c| {
                let s = c.to_string();
                self.token_to_id
                    .get(&s)
                    .copied()
                    .unwrap_or_else(|| *self.token_to_id.get(&format!("{}", c)).unwrap_or(&0))
            })
            .collect();

        // BPE loop
        loop {
            let mut best_priority = usize::MAX;
            let mut best_idx = None;

            for i in 0..tokens.len().saturating_sub(1) {
                let left = &self.vocab[tokens[i] as usize];
                let right = &self.vocab[tokens[i + 1] as usize];
                let key = (left.clone(), right.clone());
                if let Some(&p) = self.merges.get(&key) {
                    if (p as usize) < best_priority {
                        best_priority = p as usize;
                        best_idx = Some(i);
                    }
                }
            }

            match best_idx {
                None => break,
                Some(i) => {
                    let left = &self.vocab[tokens[i] as usize];
                    let right = &self.vocab[tokens[i + 1] as usize];
                    let merged = format!("{}{}", left, right);
                    match self.token_to_id.get(&merged) {
                        None => break,
                        Some(&id) => {
                            tokens[i] = id;
                            tokens.remove(i + 1);
                        }
                    }
                }
            }
        }

        tokens
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut raw_decoded = String::new();
        for &id in ids {
            if let Some(token_str) = self.vocab.get(id as usize) {
                raw_decoded.push_str(token_str);
            }
        }

        raw_decoded.replace("Ġ", " ")
    }
}

// tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        // minimal fake vocab + merges to test the logic without a real GGUF
        let vocab = vec![
            "h".to_string(),
            "e".to_string(),
            "l".to_string(),
            "o".to_string(),
            "he".to_string(),
            "hel".to_string(),
            "hell".to_string(),
            "hello".to_string(),
        ];
        let mut token_to_id = HashMap::new();
        for (i, t) in vocab.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }
        let mut merges = HashMap::new();
        merges.insert(("h".to_string(), "e".to_string()), 0u32);
        merges.insert(("he".to_string(), "l".to_string()), 1u32);
        merges.insert(("hel".to_string(), "l".to_string()), 2u32);
        merges.insert(("hell".to_string(), "o".to_string()), 3u32);

        let tokenizer = Tokenizer {
            vocab,
            token_to_id,
            merges,
        };

        let ids = tokenizer.encode("hello");
        assert_eq!(
            ids,
            vec![7u32],
            "expected 'hello' to encode to a single token [7]"
        );

        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_empty_string() {
        let tokenizer = Tokenizer {
            vocab: vec![],
            token_to_id: HashMap::new(),
            merges: HashMap::new(),
        };
        assert_eq!(tokenizer.encode(""), Vec::<u32>::new());
        assert_eq!(tokenizer.decode(&[]), "");
    }

    #[test]
    fn test_decode_replaces_glyph() {
        let vocab = vec!["Ġworld".to_string()];
        let mut token_to_id = HashMap::new();
        token_to_id.insert("Ġworld".to_string(), 0u32);
        let tokenizer = Tokenizer {
            vocab,
            token_to_id,
            merges: HashMap::new(),
        };

        let decoded = tokenizer.decode(&[0]);
        assert_eq!(decoded, " world"); // Ġ replaced with space
    }
}
