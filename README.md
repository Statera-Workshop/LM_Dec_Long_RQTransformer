# LM_Dec_Long_RQTransformer

RQTransformer - Long family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Benchmarked**
- Failure class: none

## Provenance

- Source repository: LM_2022_RQ-Transformer
- Upstream / license: lucidrains/RQ-Transformer (MIT)

## Differences from the source ([Unified] protocol)

- bytes reshaped to spatial x depth with depth 8; spatial 6 layers, depth 2 layers
- BUG FIX in vendored forward (2026-10-21): outputs were shifted by one position per 8-byte group (logits[t] did not use x[t]); results before the fix (bpc 4.65) are invalid and kept as *_offset_bug.json

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 25.60 M
- Causality check: pass (largest change of earlier outputs 0.0e+00)
- Overfit check (4 x 256 bytes): 5.711 -> 0.001 nats, pass
- LM-02 enwik8 (200M tokens): valid 2.022 / test 2.012 bits per byte
- LM-02 enwik8 (1B tokens): valid 1.782 / test 1.778 bits per byte
- Peak GPU memory at length 2048 (batch 1): train 209 MB, inference 87 MB
- Peak GPU memory at length 8192 (batch 1): train 899 MB, inference 196 MB
- LM-05 recall `lm05v3`: 4 pairs @ 512: 0.26 (keys removed 0.25); 4 pairs @ 2048: 0.23 (keys removed 0.26); 16 pairs @ 512: 0.07 (keys removed 0.06); 16 pairs @ 2048: 0.06 (keys removed 0.07)

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy
python run.py check rq
python run.py lm02 rq      # enwik8 is downloaded to data/ on first use
```
