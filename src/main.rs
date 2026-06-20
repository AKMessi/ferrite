mod gguf;
mod model;
mod tensor;
mod tokenizer;

use gguf::{GGUFFile, ggml_type_name};
use tensor::WeightStore;
use tokenizer::Tokenizer;

use crate::model::{Model, ModelConfig, ssm_block, transformer_block};

fn main() {
    let path = "../model/qwen.gguf";

    println!("ferrite v0.1.0 - loading {path}\n");

    let gguf = match GGUFFile::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    // header
    println!("=== Header ===");
    println!("  version:        {}", gguf.version);
    println!("  tensor count:   {}", gguf.tensors.len());
    println!("  metadata keys:  {}", gguf.metadata.len());
    println!("  alignment:      {} bytes", gguf.alignment);
    println!("  tensor data at: offset {}", gguf.tensor_data_start);
    println!();

    // key metadata
    println!("=== Key metadata ===");
    let interesting = [
        "general.architecture",
        "general.name",
        "qwen2.context_length",
        "qwen2.embedding_length",
        "qwen2.block_count",
        "qwen2.attention.head_count",
        "qwen2.attention.head_count_kv",
        "tokenizer.ggml.model",
    ];
    for key in &interesting {
        if let Some(val) = gguf.metadata.get(*key) {
            println!("  {key:<40} = {val:?}");
        }
    }
    println!();

    // all metadata
    println!("=== All metadata ({} keys) ===", gguf.metadata.len());
    let mut keys: Vec<&String> = gguf.metadata.keys().collect();
    keys.sort();
    for key in keys {
        let val = &gguf.metadata[key];
        // Skip printing long arrays (tokenizer vocab etc) — just show length
        match val {
            gguf::MetadataValue::Array(arr) => {
                println!("  {key:<45} = [array, {} items]", arr.len());
            }
            _ => println!("  {key:<45} = {val:?}"),
        }
    }
    println!();

    // tensor
    println!("=== Tensors ({} total) ===", gguf.tensors.len());
    println!("  {:<45} {:>20}  {}", "name", "shape", "type");
    println!("  {}", "-".repeat(75));
    for t in &gguf.tensors {
        let shape_str = format!("{:?}", t.shape);
        println!(
            "  {:<45} {:>20}  {}",
            t.name,
            shape_str,
            ggml_type_name(t.ggml_type)
        );
    }
    println!();

    // weight access demo
    println!("=== Weight access demo ===");
    let demo_tensor = "token_embd.weight";
    match gguf.get_tensor(demo_tensor) {
        Some(t) => {
            println!("  {demo_tensor}");
            println!("    shape:       {:?}", t.shape);
            println!("    type:        {}", ggml_type_name(t.ggml_type));
            println!("    file offset: {} bytes", t.file_offset);
            println!("    data offset: {} bytes (relative)", t.data_offset);
        }
        None => println!("  tensor '{demo_tensor}' not found"),
    }

    let store = WeightStore::open("../model/qwen.gguf").unwrap();

    // T3 checkpoint — F32 tensor, should be small floats near 1.0
    let norm = store.load("blk.0.attn_norm.weight").unwrap();
    print!("attn_norm.weight (F32) first 8: ");
    norm.data().iter().take(8).for_each(|x| print!("{:.6} ", x));
    println!();

    // T4 checkpoint — Q8_0 tensor
    let ssm = store.load("blk.0.ssm_out.weight").unwrap();
    print!("ssm_out.weight (Q8_0) first 8: ");
    ssm.data().iter().take(8).for_each(|x| print!("{:.6} ", x));
    println!();

    // T4 checkpoint — Q6_K tensor (the big one)
    let embd = store.load("token_embd.weight").unwrap();
    print!("token_embd.weight (Q6_K) first 8: ");
    embd.data().iter().take(8).for_each(|x| print!("{:.6} ", x));
    println!();

    println!("Layer 2 complete.");

    let tokenizer = Tokenizer::from_gguf(&store.gguf);
    let ids = tokenizer.encode("Hello, world!");
    println!("encoded: {:?}", ids);
    println!("decoded: {}", tokenizer.decode(&ids));

    let config = ModelConfig::from_gguf(&store.gguf);
    let model = Model::new(config, store);
    let embedding = model.embed(9419);
    println!(
        "embedding shape: {:?}, first 4: {:?}",
        embedding.shape(),
        &embedding.data()[..4]
    );

    // checkpoint: run one SSM block (block 0) and one attention block (block 3) standalone
    let test_token = 9419u32; // "Hello" from your tokenizer test
    let embedding = model.embed(test_token);

    println!("embedding shape: {:?}", embedding.shape());

    let ssm_out = ssm_block(&embedding, 0, &model);
    println!(
        "ssm_block(layer 0) output shape: {:?}, first 4: {:?}",
        ssm_out.shape(),
        &ssm_out.data()[..4]
    );

    let attn_out = transformer_block(&embedding, 3, 0, &model);
    println!(
        "transformer_block(layer 3) output shape: {:?}, first 4: {:?}",
        attn_out.shape(),
        &attn_out.data()[..4]
    );

    println!("\n=== Layer 4 checkpoint: full forward pass ===");
    let logits = model.forward(&[test_token], 0);
    println!("forward() output shape: {:?}", logits.shape());
    println!("first 8 logits: {:?}", &logits.data()[..8]);
}
