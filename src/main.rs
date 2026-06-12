mod gguf;

use gguf::{GGUFFile, ggml_type_name};

fn main() {
    let path = "../model/qwen-3.5.gguf";

    println!("ferrite v0.1.0 - loading {path}\n");

    let gguf = match GGUFFile::open(path) {
        Ok(f) => f,
        Err(e) => { eprintln!("error: {e}"); std::process::exit(1); } 
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
}
