# nanochat

![nanochat logo](dev/nanochat.png)

> The best ChatGPT that $100 can buy.

This repo is a full-stack implementation of an LLM like ChatGPT in a single, clean, minimal, hackable, dependency-lite codebase. nanochat is designed to run on a single 8XH100 node via scripts like [speedrun.sh](speedrun.sh), that run the entire pipeline start to end. This includes tokenization, pretraining, finetuning, evaluation, inference, and web serving over a simple UI so that you can talk to your own LLM just like ChatGPT. nanochat will become the capstone project of the course LLM101n being developed by Eureka Labs.

## Talk to it

To get a sense of the endpoint of this repo, you can currently find [nanochat d34](https://github.com/karpathy/nanochat/discussions/314) hosted on [nanochat.karpathy.ai](https://nanochat.karpathy.ai/). "d34" means that this model has 34 layers in the Transformer neural network. This model has 2.2 billion parameters, it was trained on 88 billion tokens by simply running the training script [run1000.sh](run1000.sh) with `--target_param_data_ratio=40` (2x longer than Chinchilla-optimal), and the total cost of training was ~$2,500 (about 100 hours training time on 8XH100 GPU node). While today this is enough to outperform GPT-2 of 2019, it falls dramatically short of modern Large Language Models like GPT-5. When talking to these micro models, you'll see that they make a lot of mistakes, they are a little bit naive and silly and they hallucinate a ton, a bit like children. It's kind of amusing. But what makes nanochat unique is that it is fully yours - fully configurable, tweakable, hackable, and trained by you from start to end. To train and talk to your own, we turn to...

## Updates

- (Jan 7 2026) See new post: [nanochat Miniseries v1](https://github.com/karpathy/nanochat/discussions/420) and the associated script [miniseries.sh](miniseries.sh).

## Quick start

The fastest way to feel the magic is to run the speedrun script [speedrun.sh](speedrun.sh), which trains and inferences the $100 tier of nanochat. On an 8XH100 node at $24/hr, this gives a total run time of about 4 hours. Boot up a new 8XH100 GPU box from your favorite provider (e.g. I use and like [Lambda](https://lambda.ai/service/gpu-cloud)), and kick off the training script:

```bash
bash speedrun.sh
```

Alternatively, since the script runs for 4 hours, I like to launch it like this inside a new screen session `speedrun` (and also log output to `speedrun.log`):

```bash
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

See the [screen cheatsheet](https://gist.github.com/jctosta/af918e1618682638aa82) if you are less familiar. You can watch it go inside the screen session, or detach with `Ctrl-a d` and `tail speedrun.log` to view progress. Now wait 4 hours. Once it's done, you can talk to your LLM via the ChatGPT-like web UI. Make sure again that your local uv virtual environment is active (run `source .venv/bin/activate`), and serve it:

```bash
python -m scripts.chat_web
```

And then visit the URL shown. Make sure to access it correctly, e.g. on Lambda use the public IP of the node you're on, followed by the port, so for example [http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/), etc. Then talk to your LLM as you'd normally talk to ChatGPT! Get it to write stories or poems. Ask it to tell you who you are to see a hallucination. Ask it why the sky is blue. Or why it's green. The speedrun is a 4e19 FLOPs capability model so it's a bit like talking to a kindergartener :).

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

You can also `cat report.md` file which appeared in the project directory and contains the "report card" of the run, i.e. a bunch of evaluations and metrics. At the very end, you'll see a summary table, for example:

---

- Characters: 333,989
- Lines: 8,304
- Files: 44
- Tokens (approx): 83,497
- Dependencies (uv.lock lines): 2,004

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.2219   | -        | -        | -        |
| ARC-Challenge   | -        | 0.2875   | 0.2807   | -        |
| ARC-Easy        | -        | 0.3561   | 0.3876   | -        |
| GSM8K           | -        | 0.0250   | 0.0455   | 0.0758   |
| HumanEval       | -        | 0.0671   | 0.0854   | -        |
| MMLU            | -        | 0.3111   | 0.3151   | -        |
| ChatCORE        | -        | 0.0730   | 0.0884   | -        |

Total wall clock time: 3h51m

---

(Your table might be missing the RL number by default). For a lot more information around the speedrun script and what to look for and expect, please refer to the walkthrough that I posted in Discussions of the repo: ["Introducing nanochat: The best ChatGPT that $100 can buy"](https://github.com/karpathy/nanochat/discussions/1).

## Bigger models

Unsurprisingly, $100 is not enough to train a highly performant ChatGPT clone. In fact, LLMs are famous for their multi-million dollar capex. For our purposes, I think there are two more scales of interest. First is the ~$300 tier d26 model (i.e. depth=26) that trains in ~12 hours, which slightly outperforms GPT-2 CORE score. Second is the $1000 tier (~41.6 hours), just because it's a nice round number. But both of these are not yet fully supported and therefore not attached here in the master branch yet.

That said, to give a sense, the example changes needed for the [speedrun.sh](speedrun.sh) file to train a GPT-2 grade model d26 only involve three changes:

```bash
...
# you'll need to download more data shards for pretraining
# get the number of parameters, multiply 20 to get tokens, multiply by 4.8 to get chars,
# divide by 250 million to get number of shards. todo need to improve this...
python -m nanochat.dataset -n 450 &
...
# use --depth to increase model size. to not oom, halve device batch size 32 -> 16:
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --device_batch_size=16
...
# make sure to use the same later during midtraining:
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=16
```

That's it! The biggest thing to pay attention to is making sure you have enough data shards to train on (the code will loop and do more epochs over the same training set otherwise, decreasing learning speed a bit), and managing your memory/VRAM, primarily by decreasing the `device_batch_size` until things fit (the scripts automatically compensate by increasing the number of gradient accumulation loops, simply turning parallel compute to sequential compute).

And a bit more about computing environments that will run nanochat:

- The code will run just fine on the Ampere 8XA100 GPU node as well, but a bit slower.
- All code will run just fine on even a single GPU by omitting `torchrun`, and will produce ~identical results (code will automatically switch to gradient accumulation), but you'll have to wait 8 times longer.
- If your GPU(s) have less than 80GB, you'll have to tune some of the hyperparameters or you will OOM / run out of VRAM. Look for `--device_batch_size` in the scripts and reduce it until things fit. E.g. from 32 (default) to 16, 8, 4, 2, or even 1. Less than that you'll have to know a bit more what you're doing and get more creative.
- Most of the code is fairly vanilla PyTorch so it should run on anything that supports that - xpu, mps, or etc, but I haven't implemented this out of the box so it might take a bit of tinkering.

## Running on CPU / MPS

nanochat can be run on CPU or on MPS (if you're on Macbook), and will automatically try to detect what device is best to run on. You're not going to get too far without GPUs, but at least you'll be able to run the code paths and maybe train a tiny LLM with some patience. For an example of how to make all the run commands much smaller (feel free to tune!), you can refer to [dev/runcpu.sh](dev/runcpu.sh) file. You'll see that I'm essentially restricting all scripts to train smaller models, to run for shorter number of iterations, etc. This functionality is new, slightly gnarly (touched a lot of code), and was merged in this [CPU|MPS PR](https://github.com/karpathy/nanochat/pull/88) on Oct 21, 2025.

## Parameters Reference

This section documents all configurable parameters across nanochat's training pipeline. Each script has command-line arguments that can be customized to control the training process.

### Environment Variables

These can be set before running `speedrun.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| `NANOCHAT_BASE_DIR` | `$HOME/.cache/nanochat` | Directory for intermediate artifacts and checkpoints |
| `WANDB_RUN` | `dummy` | Wandb run name for logging (set to anything other than 'dummy' to enable wandb) |
| `NPROC_PER_NODE` | `1` | Number of GPUs/processes per node for distributed training |
| `OMP_NUM_THREADS` | `1` | OpenMP thread count |

### Tokenizer Training (`scripts/tok_train.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-chars` | `10000000000` (10B) | Maximum characters to train tokenizer on |
| `--doc-cap` | `10000` | Maximum characters per document during training |
| `--vocab-size` | `32768` (2^15) | Vocabulary size for the tokenizer |

**Example:**
```bash
python -m scripts.tok_train --max-chars=2000000000 --vocab-size=65536
```

### Base Model Training (`scripts/base_train.py`)

#### Logging
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--run` | `dummy` | Wandb run name ('dummy' disables wandb logging) |

#### Runtime
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--device-type` | `""` (autodetect) | Device type: cuda, cpu, or mps |

#### Model Architecture
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--depth` | `20` | Number of Transformer layers |
| `--aspect-ratio` | `64` | Model dimension = depth × aspect_ratio |
| `--head-dim` | `128` | Target head dimension for attention |
| `--max-seq-len` | `2048` | Maximum context length |
| `--window-pattern` | `SSSL` | Sliding window pattern: L=full attention, S=half context |
| `--gqa-ratio` | `1` | GQA ratio (num_heads / num_kv_heads) |

#### Training Horizon (use one, in order of precedence)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-iterations` | `-1` | Explicit number of optimization steps (-1 = use ratio/flops) |
| `--target-flops` | `-1.0` | Calculate iterations to reach target FLOPs (-1 = use ratio) |
| `--target-param-data-ratio` | `8` | Data:param ratio (Chinchilla=20, -1 = use explicit iterations) |

#### Optimization
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--device-batch-size` | `32` | Per-device batch size (reduce if OOM) |
| `--total-batch-size` | `524288` | Total batch size in tokens across all devices |
| `--embedding-lr` | `0.3` | Learning rate for embedding parameters (Adam) |
| `--unembedding-lr` | `0.004` | Learning rate for unembedding/output parameters (Adam) |
| `--matrix-lr` | `0.02` | Learning rate for matrix parameters (Muon optimizer) |
| `--scalar-lr` | `0.5` | Learning rate for scalar parameters (resid_lambdas, x0_lambdas) |
| `--weight-decay` | `0.2` | Weight decay for Muon optimizer |
| `--adam-beta1` | `0.8` | Adam beta1 for embedding/unembedding |
| `--adam-beta2` | `0.95` | Adam beta2 for embedding/unembedding |
| `--warmup-ratio` | `0.0` | Fraction of iterations for LR warmup |
| `--warmdown-ratio` | `0.4` | Fraction of iterations for LR warmdown |
| `--final-lr-frac` | `0.0` | Final LR as fraction of initial LR |
| `--resume-from-step` | `-1` | Resume training from checkpoint step (-1 = start fresh) |

#### Evaluation
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--eval-every` | `250` | Evaluate validation loss every N steps (-1 = disable) |
| `--eval-tokens` | `10485760` (20×524288) | Number of tokens for validation evaluation |
| `--core-metric-every` | `2000` | Evaluate CORE metric every N steps (-1 = disable) |
| `--core-metric-max-per-task` | `500` | Number of examples per task for CORE evaluation |
| `--sample-every` | `2000` | Sample text from model every N steps (-1 = disable) |
| `--save-every` | `-1` | Save checkpoints every N steps (-1 = only at end) |
| `--log-every` | `10` | Log training stats every N steps |

#### Output
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-tag` | `None` | Override model tag for checkpoint directory (default: d{depth}) |

#### HyperConnections (Advanced)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--hc-num-streams` | `1` | Number of hyper-connection streams |
| `--hc-num-fracs` | `1` | Number of fractions for hyper-connections |
| `--hc-disable` | `False` | Disable hyper-connections (use identity) |
| `--mhc` | `False` | Enable manifold-constrained hyper-connections |
| `--hc-geometric` | `False` | Use geometric-induced hyper-connections |
| `--hc-manifold-dim` | `4` | Manifold dimension for geometric hyper-connections |
| `--gradient-checkpointing` | `False` | Enable gradient checkpointing for HyperConnections |
| `--sinkhorn-iters` | `10` | Sinkhorn iterations for MHC |
| `--sinkhorn-tau` | `0.05` | Sinkhorn tau parameter |
| `--mhc-h-res-proj` | `sinkhorn` | MHC projection method |

**Example:**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=20 \
  --device-batch-size=32 \
  --target-param-data-ratio=20 \
  --run=my_run
```

### Midtraining (`scripts/mid_train.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--run` | `dummy` | Wandb run name ('dummy' disables wandb logging) |
| `--device-type` | `""` (autodetect) | Device type: cuda, cpu, or mps |
| `--dtype` | `bfloat16` | Precision: float32 or bfloat16 |
| `--model-tag` | `None` | Model tag to load from base training |
| `--model-step` | `None` | Specific checkpoint step to load |
| `--num-iterations` | `-1` | Number of optimization steps (-1 = full epoch) |
| `--max-seq-len` | `2048` | Maximum context length |
| `--device-batch-size` | `32` | Per-device batch size |
| `--total-batch-size` | `524288` | Total batch size in tokens |
| `--embedding-lr` | `0.2` | Learning rate for embedding parameters |
| `--unembedding-lr` | `0.004` | Learning rate for unembedding parameters |
| `--matrix-lr` | `0.02` | Learning rate for matrix parameters |
| `--weight-decay` | `0.0` | Weight decay for Adam optimizer |
| `--init-lr-frac` | `1.0` | Initial LR as fraction of base LR |
| `--eval-every` | `150` | Evaluate validation loss every N steps |
| `--eval-tokens` | `10485760` | Number of tokens for validation |
| `--dry-run` | `False` | Log to wandb but skip checkpoints/report |

**Example:**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- \
  --device-batch-size=16 \
  --run=my_mid_run
```

### Supervised Finetuning (`scripts/chat_sft.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--run` | `dummy` | Wandb run name ('dummy' disables wandb logging) |
| `--device-type` | `""` (autodetect) | Device type: cuda, cpu, or mps |
| `--dtype` | `bfloat16` | Precision: float32 or bfloat16 |
| `--source` | `mid` | Checkpoint source: base or mid |
| `--model-tag` | `None` | Model tag to load from |
| `--model-step` | `None` | Specific checkpoint step to load |
| `--num-epochs` | `1` | Number of training epochs |
| `--num-iterations` | `-1` | Override number of iterations (-1 = use num_epochs) |
| `--device-batch-size` | `4` | Per-device batch size |
| `--target-examples-per-step` | `32` | Target examples per optimization step |
| `--embedding-lr` | `0.2` | Learning rate for embedding parameters |
| `--unembedding-lr` | `0.004` | Learning rate for unembedding parameters |
| `--matrix-lr` | `0.02` | Learning rate for matrix parameters |
| `--weight-decay` | `0.0` | Weight decay for Adam optimizer |
| `--init-lr-frac` | `0.02` | Initial LR as fraction of base LR |
| `--eval-every` | `100` | Evaluate validation loss every N steps |
| `--eval-steps` | `100` | Number of batches for validation evaluation |
| `--eval-metrics-every` | `200` | Evaluate accuracy metrics every N steps |
| `--eval-metrics-max-problems` | `1024` | Max problems per metric evaluation |

**Example:**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
  --num-epochs=1 \
  --run=my_sft_run
```

### Reinforcement Learning (`scripts/chat_rl.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--run` | `dummy` | Wandb run name ('dummy' disables wandb logging) |
| `--device-type` | `""` (autodetect) | Device type: cuda, cpu, or mps |
| `--dtype` | `bfloat16` | Precision: float32 or bfloat16 |
| `--source` | `sft` | Checkpoint source: mid or sft |
| `--model-tag` | `None` | Model tag to load from |
| `--model-step` | `None` | Specific checkpoint step to load |
| `--num-epochs` | `1` | Number of epochs over GSM8K |
| `--device-batch-size` | `8` | Max batch size per forward pass |
| `--examples-per-step` | `16` | Total examples per optimization step |
| `--num-samples` | `16` | Number of samples per example/question |
| `--max-new-tokens` | `256` | Max tokens to generate per sample |
| `--temperature` | `1.0` | Sampling temperature |
| `--top-k` | `50` | Top-k sampling (0 = disabled) |
| `--embedding-lr` | `0.2` | Learning rate for embedding parameters |
| `--unembedding-lr` | `0.004` | Learning rate for unembedding parameters |
| `--matrix-lr` | `0.02` | Learning rate for matrix parameters |
| `--weight-decay` | `0.0` | Weight decay for Adam optimizer |
| `--init-lr-frac` | `0.05` | Initial LR as fraction of base LR |
| `--eval-every` | `60` | Evaluate pass@k every N steps |
| `--eval-examples` | `400` | Number of examples for pass@k evaluation |
| `--save-every` | `60` | Save checkpoint every N steps |

**Example:**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- \
  --num-epochs=1 \
  --run=my_rl_run
```

### Base Model Evaluation (`scripts/base_eval.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--hf-path` | `None` | HuggingFace model path to evaluate (e.g., openai-community/gpt2) |
| `--max-per-task` | `-1` | Max examples per task to evaluate (-1 = all examples) |
| `--model-tag` | `None` | Model tag for checkpoint directory |
| `--step` | `None` | Specific checkpoint step to load |

**Example:**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --max-per-task=100
```

### Base Model Loss Evaluation (`scripts/base_loss.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--device-batch-size` | `4` | Per-device batch size (increase for multi-GPU) |
| `--split-tokens` | `20971520` (40×524288) | Number of tokens to evaluate per split |
| `--model-tag` | `None` | Model tag for checkpoint directory |
| `--model-step` | `None` | Specific checkpoint step to load |
| `--device-type` | `""` (autodetect) | Device type: cuda, cpu, or mps |
| `--hf-path` | `None` | HuggingFace model path (e.g., openai-community/gpt2) |

**Example:**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
```

### Chat Model Evaluation (`scripts/chat_eval.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-i, --source` | Required | Source of the model: sft, mid, or rl |
| `-a, --task-name` | `None` (all tasks) | Task name to evaluate (use \| to split multiple tasks) |
| `-d, --dtype` | `bfloat16` | Precision: float32 or bfloat16 |
| `-t, --temperature` | `0.0` | Sampling temperature |
| `-m, --max-new-tokens` | `512` | Max tokens to generate |
| `-n, --num-samples` | `1` | Number of samples per prompt |
| `-k, --top-k` | `50` | Top-k sampling parameter |
| `-b, --batch-size` | `8` | Batch size for categorical evaluation |
| `-g, --model-tag` | `None` | Model tag to load |
| `-s, --step` | `None` | Specific checkpoint step to load |
| `-x, --max-problems` | `None` | Max problems to evaluate |
| `--device-type` | `""` (autodetect) | Device type: cuda, cpu, or mps |

**Example:**
```bash
python -m scripts.chat_eval -i sft -a "ARC-Easy|GSM8K"
```

### Chat CLI (`scripts/chat_cli.py`)

Interactive command-line chat interface.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-i, --source` | `sft` | Source of the model: sft, mid, or rl |
| `-g, --model-tag` | `None` | Model tag to load |
| `-s, --step` | `None` | Specific checkpoint step to load |
| `-p, --prompt` | `""` | Prompt the model for single response (leave empty for interactive mode) |
| `-t, --temperature` | `0.6` | Temperature for generation |
| `-k, --top-k` | `50` | Top-k sampling parameter |
| `--device-type` | `""` (autodetect) | Device type: cuda, cpu, or mps |
| `-d, --dtype` | `bfloat16` | Precision: float32 or bfloat16 |

**Example:**
```bash
# Interactive mode
python -m scripts.chat_cli -i sft

# Single prompt mode
python -m scripts.chat_cli -i sft -p "Why is the sky blue?"
```

### Chat Web Server (`scripts/chat_web.py`)

Web-based chat interface with API endpoints.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-n, --num-gpus` | `1` | Number of GPUs to use for data parallelism |
| `-i, --source` | `sft` | Source of the model: sft, mid, or rl |
| `-t, --temperature` | `0.8` | Default temperature for generation |
| `-k, --top-k` | `50` | Default top-k sampling parameter |
| `-m, --max-tokens` | `512` | Default max tokens for generation |
| `-g, --model-tag` | `None` | Model tag to load |
| `-s, --step` | `None` | Specific checkpoint step to load |
| `-p, --port` | `8000` | Port to run the server on |
| `-d, --dtype` | `bfloat16` | Precision: float32 or bfloat16 |
| `--device-type` | `""` (autodetect) | Device type: cuda, cpu, or mps |
| `--host` | `0.0.0.0` | Host to bind the server to |

**Example:**
```bash
# Single GPU
python -m scripts.chat_web

# Multi-GPU (4 GPUs)
python -m scripts.chat_web --num-gpus 4 --port 8000
```

**API Endpoints:**
- `GET /` - Chat UI
- `POST /chat/completions` - Chat API (streaming only)
- `GET /health` - Health check with worker pool status
- `GET /stats` - Worker pool statistics and GPU utilization

### Common Tips

- **Memory Management**: If you encounter OOM errors, reduce `--device-batch-size`. The code automatically adjusts gradient accumulation to maintain the same effective batch size.
- **Batch Size Scaling**: Learning rates are automatically scaled based on batch size using square root scaling for AdamW and Muon optimizers.
- **Weight Decay Scaling**: Weight decay is automatically scaled by `(12/depth)^2` to account for model size.
- **Device Auto-detection**: Leave `--device-type` empty to automatically select the best available device (cuda > mps > cpu).
- **Checkpointing**: Use `--model-tag` to organize multiple training runs with custom names instead of default `d{depth}` naming.

## Customization

To customize your nanochat, see [Guide: infusing identity to your nanochat](https://github.com/karpathy/nanochat/discussions/139) in Discussions, which describes how you can tune your nanochat's personality through synthetic data generation and mixing that data into midtraining and SFT stages.

Additionally, to add new abilities to nanochat, see [Guide: counting r in strawberry (and how to add abilities generally)](https://github.com/karpathy/nanochat/discussions/164).

## Questions

nanochat is designed to be short and sweet. One big advantage of this is that we can package up all of the files together and copy paste them to your favorite LLM to ask arbitrary questions. As an example, I like to package up the repo using the [files-to-prompt](https://github.com/simonw/files-to-prompt) utility like so:

```bash
files-to-prompt . -e py -e md -e html -e toml -e sh --cxml > packaged.txt
```

This includes all py, html, toml, sh files and chooses the cxml output format. Everything is written to the `packaged.txt` file, which atm measures ~330KB (i.e. well below ~100K tokens for a state of the art LLM), and ~8K lines of code in 45 files.

Alternatively, I recommend using [DeepWiki](https://deepwiki.com/karpathy/nanochat) from Devin/Cognition to ask questions of this repo. In the URL of this repo, simply change github.com to deepwiki.com, and you're off.

## Tests

I haven't invested too much here but some tests exist, especially for the tokenizer. Run e.g. as:

```bash
python -m pytest tests/test_engine.py -v -s
```

## File structure

```
.
├── LICENSE
├── README.md
├── dev
│   ├── gen_synthetic_data.py       # Example synthetic data for identity
│   ├── generate_logo.html
│   ├── nanochat.png
│   ├── repackage_data_reference.py # Pretraining data shard generation
│   └── runcpu.sh                   # Small example of how to run on CPU/MPS
├── nanochat
│   ├── __init__.py                 # empty
│   ├── adamw.py                    # Distributed AdamW optimizer
│   ├── checkpoint_manager.py       # Save/Load model checkpoints
│   ├── common.py                   # Misc small utilities, quality of life
│   ├── core_eval.py                # Evaluates base model CORE score (DCLM paper)
│   ├── dataloader.py               # Tokenizing Distributed Data Loader
│   ├── dataset.py                  # Download/read utils for pretraining data
│   ├── engine.py                   # Efficient model inference with KV Cache
│   ├── execution.py                # Allows the LLM to execute Python code as tool
│   ├── gpt.py                      # The GPT nn.Module Transformer
│   ├── logo.svg
│   ├── loss_eval.py                # Evaluate bits per byte (instead of loss)
│   ├── muon.py                     # Distributed Muon optimizer
│   ├── report.py                   # Utilities for writing the nanochat Report
│   ├── tokenizer.py                # BPE Tokenizer wrapper in style of GPT-4
│   └── ui.html                     # HTML/CSS/JS for nanochat frontend
├── pyproject.toml
├── run1000.sh                      # Train the ~$800 nanochat d32
├── scripts
│   ├── base_eval.py                # Base model: calculate CORE score
│   ├── base_loss.py                # Base model: calculate bits per byte, sample
│   ├── base_train.py               # Base model: train
│   ├── chat_cli.py                 # Chat model (SFT/Mid): talk to over CLI
│   ├── chat_eval.py                # Chat model (SFT/Mid): eval tasks
│   ├── chat_rl.py                  # Chat model (SFT/Mid): reinforcement learning
│   ├── chat_sft.py                 # Chat model: train SFT
│   ├── chat_web.py                 # Chat model (SFT/Mid): talk to over WebUI
│   ├── mid_train.py                # Chat model: midtraining
│   ├── tok_eval.py                 # Tokenizer: evaluate compression rate
│   └── tok_train.py                # Tokenizer: train it
├── speedrun.sh                     # Train the ~$100 nanochat d20
├── tasks
│   ├── arc.py                      # Multiple choice science questions
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── customjson.py               # Make Task from arbitrary jsonl convos
│   ├── gsm8k.py                    # 8K Grade School Math questions
│   ├── humaneval.py                # Misnomer; Simple Python coding task
│   ├── mmlu.py                     # Multiple choice questions, broad topics
│   ├── smoltalk.py                 # Conglomerate dataset of SmolTalk from HF
│   └── spellingbee.py              # Task teaching model to spell/count letters
├── tests
│   └── test_engine.py
└── uv.lock
```

## Contributing

nanochat is nowhere near finished. The goal is to improve the state of the art in micro models that are accessible to work with end to end on budgets of < $1000 dollars. Accessibility is about overall cost but also about cognitive complexity - nanochat is not an exhaustively configurable LLM "framework"; there will be no giant configuration objects, model factories, or if-then-else monsters in the code base. It is a single, cohesive, minimal, readable, hackable, maximally-forkable "strong baseline" codebase designed to run start to end and produce a concrete ChatGPT clone and its report card.

Current LLM policy: disclosure. When submitting a PR, please declare any parts that had substantial LLM contribution and that you have not written or that you do not fully understand.

## Acknowledgements

- The name (nanochat) derives from my earlier project [nanoGPT](https://github.com/karpathy/nanoGPT), which only covered pretraining.
- nanochat is also inspired by [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), which gamified the nanoGPT repo with clear metrics and a leaderboard, and borrows a lot of its ideas and some implementation for pretraining.
- Thank you to [HuggingFace](https://huggingface.co/) for fineweb and smoltalk.
- Thank you [Lambda](https://lambda.ai/service/gpu-cloud) for the compute used in developing this project.
- Thank you to chief LLM whisperer 🧙‍♂️ Alec Radford for advice/guidance.
- Thank you to the repo czar Sofie [@svlandeg](https://github.com/svlandeg) for help with managing issues, pull requests and discussions of nanochat.

## Cite

If you find nanochat helpful in your research cite simply as:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT
