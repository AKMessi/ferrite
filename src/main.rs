mod cache;
mod gguf;
mod model;
mod tensor;
mod tokenizer;

use cache::KVCache;
use gguf::{GGUFFile, ggml_type_name};
use model::{Model, ModelConfig, ssm_block, transformer_block};
use tensor::WeightStore;
use tokenizer::Tokenizer;

fn main() {
    let path = "../model/qwen.gguf";
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

    let ssm_check = ssm_block(&embedding, 0, &model);
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

    let logits = model.forward(&ids, 0, &mut cache);
    println!("  forward(pos=0) shape: {:?}", logits.shape());
    println!("  first 8 logits: {:?}", &logits.data()[..8]);

    // run a second position on the SAME cache to confirm state persists
    if ids.len() > 1 {
        let logits2 = model.forward(&ids, 1, &mut cache);
        println!("  forward(pos=1) shape: {:?}", logits2.shape());
        println!("  first 8 logits: {:?}", &logits2.data()[..8]);
    }

    println!("\nAll layers checked.");
}
