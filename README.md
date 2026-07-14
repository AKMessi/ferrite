# ferrite

A from-scratch LLM inference engine for **Qwen3.5-0.8B** (a hybrid Mamba-Transformer / Gated DeltaNet architecture), written in Rust with zero ML framework dependencies — no PyTorch, no ggml bindings, no candle. Every piece — the GGUF parser, tensor engine, quantization formats, tokenizer, attention, SSM recurrence, KV cache, and sampler — is implemented by hand.

This was a personal learning project, not a production inference engine. It prioritizes understanding every layer of a modern LLM's internals over speed or correctness guarantees. See [Known Issues](#known-issues) below for an honest account of where it currently falls short.

## What it does

- Parses raw GGUF files directly from bytes (header, metadata KV pairs, tensor index)
- Dequantizes 8 weight formats from scratch: F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS
- Implements a BPE tokenizer (with special-token handling) reading vocab/merges straight from GGUF metadata
- Runs a full 24-block hybrid forward pass:
  - **Attention blocks** (every 4th layer): grouped-query attention (8 query heads / 2 KV heads) with gated attention output, partial RoPE (rotate-half convention, 64 of 256 head dims), and a real KV cache for multi-token context
  - **SSM blocks** (the rest): Gated DeltaNet — causal depthwise conv1d, input-dependent decay/beta gates, and a persistent per-group recurrent state carried across generation steps
- Implements greedy, temperature, top-k, and top-p (nucleus) sampling
- Runs a full prefill + autoregressive decode generation loop with streaming output

## Architecture

- src/
- gguf.rs      — GGUF binary format parser
- tensor.rs    — Tensor struct, matvec/matmul/rms_norm/softmax/etc., WeightStore
- quant.rs     — Dequantization kernels for all 8 supported formats
- tokenizer.rs — BPE tokenizer with special-token support
- model.rs     — ModelConfig, attention(), ssm_block(), transformer_block(), forward()
- cache.rs     — KVCache (attention) and SSMState (recurrent state) structs
- sampler.rs   — greedy / temperature / top-k / top-p sampling
- main.rs      — layer-by-layer checkpoints + generation entrypoint

## Running it

```bash
cargo run --release
```

**Always use `--release`.** Debug builds are ~15-20x slower for the dense numeric loops in this codebase — the difference between ~1.4s/token and ~24s/token on the same hardware.

Point `main.rs`'s `path` variable at a local Qwen3.5-0.8B GGUF file (not included in this repo).

## What's verified

- Every dequantization format's output values were checked against the reference `gguf` Python library and/or the real HuggingFace `transformers` implementation
- The KV cache and SSM recurrent state were confirmed to genuinely persist and affect output across multiple forward passes (verified by running `forward()` at consecutive positions on the same cache/state and observing distinct, context-dependent outputs)
- Two real architectural bugs were found and fixed via direct comparison against HuggingFace's reference implementation:
  - A missing pre-normalization step on the SSM input path (attention blocks had it, SSM blocks initially didn't — caused unbounded gate value growth with depth)
  - A gate-value splitting bug in gated attention (the Q projection's query/gate halves were being extracted as one flat contiguous split, when the real layout is per-head interleaved)

## Known issues

**Generated text is not yet coherent.** The full pipeline runs correctly end-to-end — no panics, no NaN/inf crashes, real multi-token context via cache and state — but output is not fluent language.

Direct comparison against HuggingFace's reference hidden states (same input token, same layer) shows ferrite's numbers are in the right ballpark after the embedding layer, but diverge more than quantization noise alone would explain by the output of the first SSM block. A same-token, same-layer comparison between a heavily-quantized (Q4_K-mixed) and a near-full-precision (BF16) build of the model showed the BF16 version was **further** from the HF reference, not closer — ruling out "accumulated quantization rounding error" as the primary cause and pointing to a remaining formula-level bug in the SSM recurrence, gating math, or conv1d step, rather than a precision issue.

If picking this back up:
1. Manually verify the `pos=0` delta-rule simplification (zero initial state should reduce the recurrence to `output = beta * value * (key·query)`) against what the code actually computes
2. Double check the `dt_bias` sign convention in the decay gate computation against the real Gated DeltaNet source
3. Verify the causal conv1d step doesn't corrupt the very first token's output when the history buffer is still zero-padded

The [Q5_K dequantization](#) also shows a discrepancy against the reference `gguf` library on some sub-block values (traced to somewhere in the 6-bit scale/high-bit unpacking) — documented but not confirmed to be the actual source of the coherence issue, since the affected tensor (`token_embd.weight`) checks out fine at the embedding-lookup stage.

## What this project was for

This was built to understand, at the byte and formula level, how a modern hybrid LLM actually works — quantization formats, attention mechanics, state-space recurrences, KV caching, and the full inference pipeline — by implementing every piece by hand rather than calling into an existing library. Every layer was built, tested, and debugged incrementally, with real bugs found through careful evidence-gathering (byte-level struct verification, reference-value comparison, systematic debug tracing) rather than guesswork wherever possible.

It's not production-ready, and it was never meant to be.
