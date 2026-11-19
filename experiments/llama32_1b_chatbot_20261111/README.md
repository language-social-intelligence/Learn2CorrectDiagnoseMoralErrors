# Train the Light and Fusion reviewers

The deployed reviewers were trained independently from [Llama 3.2 1B at the pinned revision](https://huggingface.co/meta-llama/Llama-3.2-1B/tree/4e20de362430cd3b72f300e6b0f18e50e7166e08). Each run used 2,000 examples, ten epochs of full-parameter BF16 training, effective batch size 16, a learning rate of 5e-5 and seed 1. The final epoch supplied each deployed checkpoint.

Training requires the two original prepared text files. Place them in `experiments/` at the repository root, one example per line. The files and model weights are supplied separately.

| Mode | Training file | SHA-256 from the deployed run |
| --- | --- | --- |
| Light | `realtoxicity.Correct.semantic.llama3.2-3B.2000.train.1.txt` | `5304ddc389c06e0c4e42dde7d1f35383abd927190c6408eebca82a4ce72dbd0d` |
| Fusion | `MIC.Correct.pragmatic_semantic_fusion.llama3.2-3B.2000.train.1.txt` | `d9681e55d95c0b0829826f98208dfc2774b0ace5e4e7ff8aad41be0e5b5b9463` |

Run from the repository root after installing the [chatbot environment](../../second-look-web/README.md). Point `LLAMA_1B_SNAPSHOT` at the downloaded base snapshot and select an available training GPU. The script reserves up to 43% of that GPU and trains Light and Fusion sequentially.

```bash
second-look-web/.venv/bin/python -m pip install datasets==4.8.5
export LLAMA_1B_SNAPSHOT=/path/to/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08
CUDA_VISIBLE_DEVICES=0 second-look-web/.venv/bin/python experiments/llama32_1b_chatbot_20261111/train.py --mode all
```

Each mode writes its final weights and tokenizer to `<mode>/model/` beside the script. Point the corresponding reviewer checkpoint in `second-look-web/assistant_config.json` at that directory, or link it to the documented model path.
