mod cache;
mod gguf;
mod model;
mod quant;
mod sampler;
mod tensor;
mod tokenizer;

use cache::KVCache;
use gguf::{GGUFFile, ggml_type_name};
use model::{Model, ModelConfig, ssm_block, transformer_block};
use tensor::WeightStore;
use tokenizer::Tokenizer;

use crate::{cache::SSMState, sampler::sample, tensor::Tensor};

fn generate(
    model: &Model,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    temp: f32,
    top_p: f32,
    top_k: usize,
) {
    let prompt_ids = tokenizer.encode(prompt);
    print!("{}", prompt);

    let max_seq_len = 2048;
    let mut cache = KVCache::new(&model.config, max_seq_len);
    let mut ssm_state = SSMState::new(&model.config);

    let mut all_ids = prompt_ids.clone();
    let mut logits = Tensor::zeros(vec![1]);

    for pos in 0..all_ids.len() {
        logits = model.forward(&all_ids, pos, &mut cache, &mut ssm_state);
    }

    for _ in 0..max_tokens {
        let next_id = sample(&logits, temp, top_k, top_p);

        if next_id == model.config.eos_token_id {
            break;
        }

        let token_str = tokenizer.decode(&[next_id]);
        print!("{}", token_str);

        use std::io::Write;

        std::io::stdout().flush().unwrap();

        all_ids.push(next_id);

        let new_pos = all_ids.len() - 1;

        logits = model.forward(&all_ids, new_pos, &mut cache, &mut ssm_state)
    }

    println!();
}

fn main() {
    let path = "/run/media/akmessi/2EE4240DE423D63D/coding/ferrite/model/qwen.gguf";
    println!("ferrite v0.1.0 - loading {path}\n");

    // ============================================================
    // LAYER 1 — GGUF parsing
    // ============================================================
    let gguf = match GGUFFile::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    println!("=== Layer 1: GGUF Header ===");
    println!("  version:        {}", gguf.version);
    println!("  tensor count:   {}", gguf.tensors.len());
    println!("  metadata keys:  {}", gguf.metadata.len());
    println!("  alignment:      {} bytes", gguf.alignment);
    println!("  tensor data at: offset {}", gguf.tensor_data_start);

    let demo_tensor = "token_embd.weight";
    if let Some(t) = gguf.get_tensor(demo_tensor) {
        println!(
            "  demo tensor '{demo_tensor}': shape {:?}, type {}",
            t.shape,
            ggml_type_name(t.ggml_type)
        );
    }
    println!();

    // ============================================================
    // LAYER 2 — Tensor engine + dequantization
    // ============================================================
    println!("=== Layer 2: Tensor Engine ===");
    let store = WeightStore::open(path).unwrap();

    let norm = store.load("blk.0.attn_norm.weight").unwrap();
    print!("  attn_norm.weight (F32) first 4: ");
    norm.data().iter().take(4).for_each(|x| print!("{:.6} ", x));
    println!();

    let ssm_out_w = store.load("blk.0.ssm_out.weight").unwrap();
    print!("  ssm_out.weight (Q4_K) first 4: ");
    ssm_out_w
        .data()
        .iter()
        .take(4)
        .for_each(|x| print!("{:.6} ", x));
    println!();

    let embd = store.load("token_embd.weight").unwrap();
    print!("  token_embd.weight (Q5_K) first 4: ");
    embd.data().iter().take(4).for_each(|x| print!("{:.6} ", x));
    println!();
    println!("  Layer 2: OK\n");

    // ============================================================
    // LAYER 3 — Tokenizer
    // ============================================================
    println!("=== Layer 3: Tokenizer ===");
    let tokenizer = Tokenizer::from_gguf(&store.gguf);
    let test_prompt = "Hello, world!";
    let ids = tokenizer.encode(test_prompt);
    let decoded = tokenizer.decode(&ids);
    println!("  encode({:?}) = {:?}", test_prompt, ids);
    println!("  decode -> {:?}", decoded);
    println!("  roundtrip OK: {}", decoded == test_prompt);
    println!();

    // ============================================================
    // LAYER 4 — Model construction + forward pass
    // ============================================================
    println!("=== Layer 4: Model + Forward Pass ===");
    let config = ModelConfig::from_gguf(&store.gguf);
    let model = Model::new(config, store);

    let test_token = *ids.get(0).unwrap_or(&9419u32);
    let embedding = model.embed(test_token);
    println!(
        "  embed(token {}) shape: {:?}",
        test_token,
        embedding.shape()
    );

    // standalone block checks (use their own throwaway caches —
    // these don't need to share state with the real generation cache below)
    let mut scratch_cache = KVCache::new(&model.config, 2048);
    let mut scratch_ssm_state = SSMState::new(&model.config);

    let ssm_check = ssm_block(&embedding, 0, &model, &mut scratch_ssm_state);
    println!(
        "  ssm_block(layer 0) shape: {:?}, first 4: {:?}",
        ssm_check.shape(),
        &ssm_check.data()[..4]
    );

    let attn_check = transformer_block(&embedding, 3, 0, &model, &mut scratch_cache);
    println!(
        "  transformer_block(layer 3) shape: {:?}, first 4: {:?}",
        attn_check.shape(),
        &attn_check.data()[..4]
    );
    println!();

    // ============================================================
    // LAYER 5 — KV Cache, full forward pass
    // ============================================================
    println!("=== Layer 5: Full Forward Pass with KV Cache ===");
    let mut cache = KVCache::new(&model.config, 2048);
    let mut ssm_state = SSMState::new(&model.config);

    let logits = model.forward(&ids, 0, &mut cache, &mut ssm_state);
    println!("  forward(pos=0) shape: {:?}", logits.shape());
    println!("  first 8 logits: {:?}", &logits.data()[..8]);

    // run a second position on the SAME cache and SAME ssm_state to confirm state persists
    if ids.len() > 1 {
        let logits2 = model.forward(&ids, 1, &mut cache, &mut ssm_state);
        println!("  forward(pos=1) shape: {:?}", logits2.shape());
        println!("  first 8 logits: {:?}", &logits2.data()[..8]);
    }

    println!("\nAll layers checked.");

    println!("=== Quant format audit ===");
    use std::collections::HashSet;

    let mut types_seen: HashSet<u32> = HashSet::new();
    for t in &model.weights().gguf.tensors {
        types_seen.insert(t.ggml_type);
    }

    let mut sorted_types: Vec<u32> = types_seen.into_iter().collect();
    sorted_types.sort();

    println!(
        "  unique ggml_type values found in this model: {:?}",
        sorted_types
    );
    for t in &sorted_types {
        println!("    type {} = {}", t, ggml_type_name(*t));
    }

    println!("\n  load() success per type:");
    for target_type in &sorted_types {
        if let Some(tensor_info) = model
            .weights()
            .gguf
            .tensors
            .iter()
            .find(|t| t.ggml_type == *target_type)
        {
            let name = &tensor_info.name;
            match model.weights().load(name) {
                Ok(_) => println!(
                    "    type {} ({}) — OK  [example: {}]",
                    target_type,
                    ggml_type_name(*target_type),
                    name
                ),
                Err(e) => println!(
                    "    type {} ({}) — FAILS: {}  [example: {}]",
                    target_type,
                    ggml_type_name(*target_type),
                    e,
                    name
                ),
            }
        }
    }
    println!();

    let iq4_test = model.weights().load("blk.8.ffn_gate.weight").unwrap();
    print!("IQ4_XS tensor first 8: ");
    iq4_test
        .data()
        .iter()
        .take(8)
        .for_each(|x| print!("{:.6} ", x));
    println!();

    println!("=== Layer 7: Sampler ===");
    use sampler::{greedy, sample};

    let greedy_token = greedy(&logits);
    println!("  greedy(logits) -> token {}", greedy_token);
    println!("  decoded: {:?}", tokenizer.decode(&[greedy_token]));

    let sampled_token = sample(&logits, 0.8, 40, 0.9);
    println!(
        "  sample(temp=0.8, top_k=40, top_p=0.9) -> token {}",
        sampled_token
    );
    println!("  decoded: {:?}", tokenizer.decode(&[sampled_token]));

    println!("\n=== Layer 8: Generation ===");
    let prompt = "The meaning of life is";
    generate(&model, &tokenizer, prompt, 10, 0.8, 0.9, 40);
}
