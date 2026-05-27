# Learning When to Think While Listening

[![arXiv](https://img.shields.io/badge/arXiv-2605.27190-b31b1b.svg)](https://arxiv.org/abs/2605.27190)

Official repo for **Learning When to Think While Listening in Large Audio-Language Models**.

The code trains a Qwen2.5-Omni-7B controller that operates over a visible
wait-think-answer action space:

- `<wait/>`
- `<think>...</think>`
- `<answer>...</answer>`

This repository provides the SFT launch wrapper, DAPO controller trainer,
reward implementations, and public prompt templates used for the paper training
run.

## Public Resources

- Real Audio Bench: https://huggingface.co/datasets/Oulasong/Real_Audio_Bench
- SRQA Audio: https://huggingface.co/datasets/Oulasong/SRQA_Audio

Please see the dataset cards for dataset construction details, intended use,
upstream sources, and licensing.

## Contents

- `run_grpo.py`: DAPO-style controller policy optimization.
- `training/`: local actor update, rollout policy, dataset schema, checkpointing,
  and service wrappers used by the DAPO trainer.
- `rewards/`: answer correctness, format, synchronization/latency, update timing,
  thought quality, and chain consistency rewards.
- `prompts/`: public controller and reward-judge prompt templates used by the
  paper training run.
- `scripts/convert_controller_to_ms_swift.py`: converter from controller JSONL records
  to MS-Swift SFT rows.
- `scripts/train_sft_paper.sh`: paper-aligned SFT launch wrapper.
- `scripts/train_dapo_paper.sh`: paper-aligned DAPO launch wrapper.

## Paper Configuration

The paper uses the following training stack.

These hyperparameters correspond to the main paper run. They are not universal
defaults; users may need to retune them for different data scales, model
variants, or available reward signals.

- Base model: `Qwen/Qwen2.5-Omni-7B`
- SFT framework: MS-Swift
- DAPO framework: local streaming-controller trainer in `run_grpo.py`
- SFT LoRA: rank 8, alpha 32, dropout 0.05, target modules `all-linear`
- DAPO LoRA actor: rank 8, alpha 32, dropout 0.0, target modules `all-linear`
- dtype: `bfloat16`
- SFT schedule: one epoch, learning rate `1e-5`, max length 8192
- DAPO schedule: 1000 steps, 50 warmup steps, actor learning rate `4e-7`
- DAPO group size: 8 rollouts per prompt
- DAPO clipping: `epsilon=0.20`, `epsilon_high=0.28`
- KL coefficient: `0.01`
- Reward weights: `lambda_a=1.0`, `lambda_f=1.0`, `lambda_s=1.0`,
  `lambda_u=3.0`, `lambda_t=1.0`, `lambda_c=0.45`

We upweight `R_u` because update timing is the primary controller-specific
signal: it rewards thinking near annotated acoustic-semantic update points and
penalizes missed or spurious updates.

The DAPO launch wrapper exposes the command-level mapping for these settings.

## Data Format

The controller policy is trained in audio-only mode. Text transcript fields may
appear in JSONL records as metadata for scoring, logging, or dataset
bookkeeping, but they are not exposed to the policy prompt.

At minimum, each training record should provide:

- `audio_chunks` or `audio_chunk_paths`
- `audio_path` for single-file audio examples, when chunked paths are not used
- `gt_answer`, `solution`, or `final_answer`
- optional `think_annotations`
- optional `controller_metadata`, including update ticks for the update-timing
  reward

A minimal JSONL record is:

```json
{
  "id": "example_0001",
  "audio_path": "/path/to/example.wav",
  "final_answer": "25",
  "controller_metadata": {
    "tick_seconds": 0.5,
    "update_ticks": [3, 8]
  }
}
```

For SFT, convert controller records into MS-Swift rows with:

```bash
python scripts/convert_controller_to_ms_swift.py \
  --input /path/to/controller_train.jsonl \
  --output /path/to/sft_train.jsonl \
  --mode sft \
  --tick-seconds 0.5
```

## Running SFT

Set the input and output paths, then run:

```bash
SFT_TRAIN_JSONL=/path/to/sft_train.jsonl \
SFT_VAL_JSONL=/path/to/sft_val.jsonl \
SFT_OUTPUT_DIR=/path/to/sft_output \
bash scripts/train_sft_paper.sh
```

The wrapper prints and runs the MS-Swift command used for the paper setting.

## Running DAPO

Set the dataset, SFT adapter, and output directory, then run:

```bash
DAPO_INPUT_JSONL=/path/to/dapo_train.jsonl \
SFT_ADAPTER_DIR=/path/to/sft_lora_adapter \
DAPO_RUN_DIR=/path/to/dapo_run \
bash scripts/train_dapo_paper.sh
```

The DAPO wrapper supports the reward stacks used in the paper:

```bash
REWARD_STACK=4reward bash scripts/train_dapo_paper.sh  # R_a + R_f + R_s + R_u
REWARD_STACK=5reward bash scripts/train_dapo_paper.sh  # adds R_t
REWARD_STACK=6reward bash scripts/train_dapo_paper.sh  # adds R_t and R_c
```

By default the DAPO wrapper manages a local OpenAI-compatible vLLM policy
service and uses a local Qwen3.6-35B-A3B judge endpoint for `R_t` and `R_c`.
For clusters with existing services, set `MANAGE_POLICY_SERVICE=0`,
`POLICY_ENDPOINT`, and `JUDGE_ENDPOINT`.

## License

Code is released under the Apache-2.0 license unless otherwise noted. Datasets
are released under the license specified in each Hugging Face dataset card.
Model weights and upstream assets are governed by their original licenses and
terms.

## Citation

If you use this code or the benchmarks, please cite:

```bibtex
@misc{song2026learningthinklisteninglarge,
      title={Learning When to Think While Listening in Large Audio-Language Models}, 
      author={Zhiyuan Song and Weici Zhao and Yang Xiao and Suhao Yu and Cheng Zhu and Jiatao Gu},
      year={2026},
      eprint={2605.27190},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.27190}, 
}
```
