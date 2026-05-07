# Pixels-to-Predictions with SmolVLM-500M-Instruct

- **Competition:** [Pixels-to-Predictions Kaggle Competition](https://www.kaggle.com/competitions/pixels-to-predictions)
- **GitHub repository:** `https://github.com/mariiaonokhina/Pixels-to-Predictions-Kaggle-Competition`
- **Final weights:** `[add weights link here]`

This repository documents a notebook-driven experimental pipeline for the Pixels-to-Predictions multimodal science QA competition using `HuggingFaceTB/SmolVLM-500M-Instruct`. The codebase evolves from frozen prompt-based baselines to contrastive LoRA / DoRA fine-tuning, higher-resolution image inputs, same-model caption caching, and lightweight ensemble experiments.

## Important Notes:

1. Run just the `Final_Notebook.ipynb` for the best scoring Kaggle submission (setup is identical to Model 13).
2. The best public Kaggle score recorded in the repository is **Model 12 = 0.75050**, but that notebook explicitly allows **5,100,000** trainable parameters and records **5,088,256**, which is **just above** the competition's `<5M` trainable-parameter rule, so we **do not count that submission as valid**.

## 1. Competition Overview

Pixels-to-Predictions is a **multimodal multiple-choice science QA** task: each example combines an image plus textual context, and the system must predict the correct answer index from a small set of choices. The leaderboard metric is **accuracy**.

Constraints reflected throughout this repository:

- offline inference / training workflow
- no external data
- no external model substitutions beyond the allowed base model workflow
- strict trainable-parameter budget of **< 5,000,000** for compliant adapted models

## 2. Repository Structure

This repository is notebook-centric: there are **no separate Python utility modules**. Prompt builders, scoring functions, training loops, checkpoint utilities, and submission code all live inside the notebooks.

```text
.
├── Final_Notebook.ipynb
├── experiment_notebooks/
│   ├── Model1_Score0.519.ipynb
│   ├── Model2_Score0.56539.ipynb
│   ├── Model3_No_Score.ipynb
│   ├── Model4_Score0.59154.ipynb
│   ├── Model5_Score0.64587.ipynb
│   ├── Model6_Score0.70422.ipynb
│   ├── Model7_Score0.72233.ipynb
│   ├── Model8_No_Score.ipynb
│   ├── Model9_Score0.54728.ipynb
│   ├── Model10_Score0.73440.ipynb
│   ├── Model11_No_Score.ipynb
│   ├── Model12_Score0.75050.ipynb
│   ├── Model13_Score0.73641.ipynb
│   ├── Model14_No_Score.ipynb
│   └── Model15_Score.ipynb
├── data/                              # local Kaggle data, gitignored
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── images/...
├── runs/                              # generated adapters, histories, score pickles, gitignored
│   ├── model5_...
│   ├── model6_lecture_100_attn_r8_...
│   ├── model7_lecture_180_attn_r8_mean_...
│   ├── model10_final/
│   ├── model11_final/
│   ├── model12_final/
│   ├── model14_final/
│   ├── fold0_*.pkl ... fold4_*.pkl
│   └── *_val_scores.pkl / *_test_scores.pkl
├── lora_checkpoint/                   # legacy early checkpoints
├── lora_checkpoint1/                  # legacy full-model save artifact
├── submission.csv
├── .gitignore
└── README.md
```

### Notebook roles

- `Model1` to `Model4`: frozen-model and prompt-engineering baselines, plus the first LoRA attempt.
- `Model5` to `Model7`: the main transition to **contrastive answer-index ranking** with LoRA fine-tuning.
- `Model8`: normalized score ensembling with Model 7 plus a hybrid index/text scorer.
- `Model9`: 5-fold CV experiment, ultimately rejected.
- `Model10` to `Model15`: final-run style notebooks with higher-resolution images, resumable checkpoints, pickled score caches, and explicit offline / artifact management.
- `Final_Notebook.ipynb`: a clean standalone copy of the Model 13 final-run logic.

### Local artifact layout already present in this snapshot

The repository currently contains local, gitignored run artifacts such as:

- `runs/model5_training_history.csv`
- `runs/model6_lecture_100_attn_r8_training_history.csv`
- `runs/model7_lecture_180_attn_r8_mean_training_history.csv`
- `runs/model10_final/{best,latest,checkpoints,val_scores.pkl,test_scores.pkl}`
- `runs/model11_final/{best,latest,checkpoints,val_scores.pkl,test_scores.pkl}`
- `runs/model12_final/{best,latest,checkpoints,val_scores.pkl,test_scores.pkl}`
- `runs/model14_final/{best,latest,checkpoints,val_scores.pkl,test_scores.pkl}`

`runs/model13_final/` and `runs/model15_final/` are **expected by the notebooks**, but are **not present** in this repository snapshot.

## 3. Environment Setup

### Versions inferable from the repository

- **Python:** `3.10.20` in most notebooks (`Model15` metadata shows `3.10`)
- **transformers:** `4.57.6` (pinned in install cells)
- **peft:** `0.18.1` (pinned in install cells)
- **torch:** Not explicitly pinned in repository
- **accelerate:** installed in later notebooks, version not pinned
- **safetensors:** installed in later notebooks, version not pinned
- **pandas / numpy / pillow / tqdm:** used throughout
- **scikit-learn:** used in `Model9` (`KFold`)
- **jupyter / ipykernel:** referenced in notebook setup cells
- **bitsandbytes:** appears in early install cells, but **4-bit loading is not explicitly configured in the committed code**
- **kaggle CLI:** optional, used in early notebooks for downloading the competition data

### Recommended install command

For reproducing the full repository, including early notebooks and `Model9`:

```bash
pip install torch transformers==4.57.6 peft==0.18.1 accelerate safetensors pandas numpy pillow tqdm scikit-learn jupyter ipykernel kaggle
```

If you only want the late final-run notebooks (`Model10` onward), `scikit-learn` and `kaggle` are optional.

### Hardware

- **Tested hardware explicitly shown in notebook outputs:** `NVIDIA GeForce RTX 3090`
- **CUDA usage:** yes
- **Observed precision choice in later notebooks:** `torch.bfloat16` when available, else `torch.float16`
- **FlashAttention:** optional; later notebooks try `flash_attention_2` and fall back to `eager`
- **Expected runtime per epoch:** varies from 20 minutes to 1.5 hours

## 4. Reproducibility Guide

### Data placement

The notebooks automatically search for the competition CSVs in:

1. `./data`
2. `../data`
3. `/kaggle/input/pixels-to-predictions`
4. `/kaggle/input/pixels-to-predictions/data`

The expected split sizes shown in the later notebooks are:

- `train`: `3109`
- `val`: `1048`
- `test`: `1008`

Images are resolved flexibly. The code tries paths such as:

- `DATA_DIR / row["image_path"]`
- `DATA_DIR / "images" / row["image_path"]`
- `PROJECT_ROOT / row["image_path"]`

In the current local snapshot, a sample image resolves to:

```text
data/images/images/train/train_07667.png
```

So keep the CSVs and image folders together in a layout the resolver can find.

### Offline model placement

Later notebooks use `local_files_only=True` and, in some runs, set:

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

The model source resolver checks:

1. `models/SmolVLM-500M-Instruct/`
2. `../models/SmolVLM-500M-Instruct/`
3. Kaggle input paths such as `/kaggle/input/smolvlm-500m-instruct/...`
4. local Hugging Face cache via model id fallback

For reliable offline reproduction, place the model at:

```text
models/SmolVLM-500M-Instruct/
```

### Output locations

Root-level outputs:

- `submission.csv`

Run-specific outputs (late notebooks):

- `runs/<RUN_NAME>/best/`
- `runs/<RUN_NAME>/latest/`
- `runs/<RUN_NAME>/checkpoints/`
- `runs/<RUN_NAME>/training_history.csv`
- `runs/<RUN_NAME>/val_scores.pkl`
- `runs/<RUN_NAME>/test_scores.pkl`
- `runs/<RUN_NAME>/caption_cache_{train,val,test}.pkl` when captions are enabled

### Checkpointing and resume behavior

#### Early training notebooks

- `Model5` writes:
  - `runs/model5_contrastive_lora_best/`
  - `runs/model5_contrastive_lora_latest/`
  - step checkpoints such as `runs/model5_step_checkpoint_epoch3_step500/`
  - `runs/model5_training_history.csv`

- `Model6` and `Model7` write best/latest adapters and training history CSVs.

#### Final-run notebooks (`Model10+`)

These notebooks save:

- adapter weights with `model.save_pretrained(...)`
- processor/tokenizer files with `processor.save_pretrained(...)`
- a `training_state.pt` containing:
  - optimizer state
  - scheduler state
  - scaler state when used
  - RNG state
  - `epoch`
  - `next_row`
  - `global_step`
  - `history`
  - `best_val_acc`
  - adapter metadata

Resume logic:

- if `latest/adapter_config.json` exists, training resumes from `latest`
- if `latest/training_state.pt` exists, optimizer/scheduler/scaler/RNG/global step are restored
- `next_row` allows **mid-epoch resume**

`Model14` includes an explicit continuation cell that extends the initial 2-epoch run by **3, 4, or 5 extra epochs** from `latest`.

### Validation and saved score format

The main late-run validation function is `evaluate_model(...)`.

It computes:

- prediction per row
- accuracy on `val_df`
- per-row raw choice scores
- optional pickle save to `VAL_SCORES_PATH`

Late validation pickles store a result dictionary with fields such as:

- `accuracy`
- `preds`
- `labels`
- `scores`
- `score_mode`
- `prompt_style`

Some notebooks add extra metadata like `answer_scoring`, `choice_score_style`, or `target_image_size`.

Test score pickles store a list of row-level dictionaries such as:

- `id`
- `answer`
- `scores`
- `num_choices`

### Best checkpoint selection

Late notebooks select the best checkpoint by:

```text
if val_acc > best_val_acc:
    save best checkpoint
```

So `best/` always tracks the highest validation accuracy seen so far inside that run.

### Randomness and determinism

All main notebooks set:

- `random.seed(42)`
- `numpy.random.seed(42)`
- `torch.manual_seed(42)`
- `torch.cuda.manual_seed_all(42)`

Training order is typically reshuffled each epoch with `random_state=SEED + epoch`.

The repository does **not** enable fully deterministic PyTorch execution, so exact bitwise reproducibility is **not explicitly guaranteed**.

### Step-by-step workflow

1. **Install dependencies**

   ```bash
   pip install torch transformers==4.57.6 peft==0.18.1 accelerate safetensors pandas numpy pillow tqdm scikit-learn jupyter ipykernel kaggle
   ```

2. **Prepare data**

   - place `train.csv`, `val.csv`, `test.csv`, and images under `data/`
   - ensure the image folder layout matches one of the resolver paths

3. **Prepare the offline base model**

   - place `SmolVLM-500M-Instruct` under `models/SmolVLM-500M-Instruct/`
   - or make sure the model is already cached locally / mounted in Kaggle input

4. **Choose a notebook**

   - Run `Final_Notebook.ipynb` for a clean final run (reflects Model 13's ablations).

   OR

   - (OPTIONAL) Run notebooks in `experiment-notebooks` in order from 1 to 15. They **must** be run in order because some of the later models depend on the files from the earlier ones.

5. **Run training cells in order**

   - late notebooks automatically create `runs/<RUN_NAME>/...`
   - if `latest/` exists, they resume automatically

6. **Validate**

   - late notebooks call `evaluate_model(val_df, save_scores=True)`
   - validation scores are saved to `val_scores.pkl`

7. **Generate submission**

   - the best checkpoint is reloaded for inference
   - predictions on `test_df` are written to root `submission.csv`
   - row-level test scores are written to `test_scores.pkl`

8. **Ensemble predictions (optional)**

   - `Model4` ensembles prompt variants after per-row normalization
   - `Model8` ensembles `Model7` + hybrid scorer outputs after z-score normalization
   - `Model9` averages fold logits across 5 CV models

## 5. Modeling Approach

### Base model

Every notebook centers on:

- `HuggingFaceTB/SmolVLM-500M-Instruct`

### Why scoring replaced direct generation

`Model1` shows the initial failure mode clearly: instructing the model to "return only a single integer" still led to continued free-form generation. From `Model2` onward, the repository increasingly treats the task as **multiple-choice scoring**, not raw generation.

### Multiple-choice scoring

The core pattern is:

1. build a prompt from image + metadata + question + choices
2. append one candidate answer per choice
3. run the VLM forward pass
4. extract log probabilities only over the answer span
5. aggregate token log-probability by `mean` or `sum`
6. choose `argmax(score_i)`

Scoring variants explored in the repo:

- **answer index scoring**: append `" 0"`, `" 1"`, ...
- **answer choice text scoring**: append the actual answer text
- **hybrid scoring**: combine normalized index and text scores

### Prompt engineering

Prompt evolution is one of the main themes of the repository:

- early prompts test metadata, hint, lecture, and shorter vs longer instructions
- `Model4` adds the simple but helpful image-focused sentence:
  - "Look carefully at the image when it is relevant."
- `Model6` and `Model7` test lecture truncation lengths (`100` vs `180`)
- `Model10+` add stronger scientific visual reasoning instructions:
  - labels
  - axes
  - charts
  - maps
  - diagrams
  - tables
  - units
  - spatial relationships

### Image + text conditioning

The later notebooks explicitly treat the problem as a joint reasoning task:

- prompt includes task metadata, grade, subject, topic, category, skill
- hint and truncated lecture are injected when present
- images are resized to a configured long side
- later notebooks use light train-only augmentation

### Same-model captioning

`Model10`, `Model11`, `Model12`, `Model14`, and `Model15` add **same-model caption caches**:

- captions are generated by SmolVLM itself
- captions are stored in `caption_cache_{train,val,test}.pkl`
- the caption is then reinserted into the prompt as additional context

`Model13` / `Final_Notebook` keeps the same prompt family but sets `USE_CAPTIONS=False`.

### Training objective

The main successful training formulation is **contrastive answer ranking**:

- compute one score per candidate answer
- treat those scores as class logits
- apply cross-entropy on the correct answer index

This objective first appears in `Model5` and becomes the dominant training pattern afterward.

### LoRA / DoRA adaptation

The repository starts with smaller LoRA targets, then expands to attention + MLP modules in later runs.

Later final-run notebooks:

- try **DoRA first**, then LoRA fallback if unsupported
- sweep ranks `[8, 6, 4, 2]`
- keep the highest rank that fits the parameter budget

### Ensembling and normalization

Ensembling is score-based, not vote-based:

- normalize scores **per question**
- average weighted normalized scores
- compare against the strongest single model

In this repository, ensembling was useful for analysis but did **not** beat the best single-run models.

## 6. Training Details

### Common optimizer / scheduler patterns

- **Optimizer:** `AdamW`
- **Loss:** cross-entropy over per-choice scores in the contrastive runs
- **Gradient accumulation:** usually `8` in the stronger fine-tuned runs
- **Gradient clipping:** `1.0` in the later final-run notebooks
- **Warmup scheduler:** cosine warmup appears in `Model10+`

### Key configurations by notebook family

| Notebook family | Training style | LR | Epochs | Grad accum | LoRA / DoRA setup | Validation selection |
|---|---|---:|---:|---:|---|---|
| `Model1` | answer-only supervised LoRA | `2e-4` on 200-row sanity run, then `5e-5` | `1` | `4` | `r=8`, `alpha=16`, `dropout=0.05`, `q/k/v/o` | ad hoc validation checks |
| `Model5` | contrastive answer-index LoRA | `1e-5` | `3` attempted | `8` | `r=4`, `alpha=8`, `dropout=0.05`, `q_proj`, `v_proj` | best epoch from `training_history.csv` |
| `Model6` | contrastive answer-index LoRA | `5e-6` | `2` | `8` | `attn_r8`, `r=8`, `alpha=16`, `dropout=0.05`, `q/k/v/o` | best epoch by validation |
| `Model7` | contrastive answer-index LoRA | `5e-6` | `2` | `Not explicitly specified in notebook outputs for final saved run` | `attn_r8`, `r=8`, `alpha=16`, `dropout=0.05`, `q/k/v/o` | best epoch by validation |
| `Model10` / `Model13` / `Final_Notebook` | contrastive answer-index DoRA/LoRA | `1e-5` | `3` | `8` | rank search `[8,6,4,2]`, `alpha = 2*r`, `dropout=0.05`, `q/k/v/o + gate/up/down`; selected `DoRA r=6` in recorded outputs | `best/` checkpoint if `val_acc` improves |
| `Model11` / `Model14` | contrastive choice-text DoRA/LoRA | `1e-5` | `3` for `Model11`, `2 + continuation` for `Model14` | `8` | same rank search and target modules as above; recorded selection `DoRA r=6` | `best/` checkpoint if `val_acc` improves |
| `Model12` | contrastive answer-index DoRA/LoRA | `1e-5` | `3` | `8` | same target modules; selected `DoRA r=8`, `5,088,256` trainable params | `best/` checkpoint if `val_acc` improves |
| `Model15` | contrastive hybrid label/text DoRA/LoRA | `1e-5` | `3` | `8` | candidate ranks `[8,6,4,2]`, `alpha = 2*r`, `dropout=0.05`, `q/k/v/o + gate/up/down` | training code present, final result not recorded |

### Image sizes used

- `Model10`: `512`
- `Model11`: `448`
- `Model12`: `512`
- `Model13` / `Final_Notebook`: `1024`
- `Model14`: `768`
- `Model15`: `640`

### Validation strategy

- early notebooks validate directly on `val.csv`
- `Model4` compares prompts on a 200-row subset before full validation
- `Model6` ablates lecture length before full training
- `Model8` and `Model4` ensemble after per-row score normalization
- `Model9` uses 5-fold CV and averages fold scores

## 7. Experiment / Ablation Summary

### Experiment table

| Notebook | Prompt style | Training type | Scoring mode | LoRA setup | Validation result | Kaggle score |
|---|---|---|---|---|---:|---:|
| `Model1_Score0.519.ipynb` | expanded starter prompt with metadata/context/hint | answer-only LoRA SFT | mean log-prob over answer text | `r=8`, `alpha=16`, `dropout=0.05`, `q/k/v/o` | `~0.51` | `0.519` |
| `Model2_Score0.56539.ipynb` | `metadata_hint_q_choices` | frozen inference only | index scoring, `sum` and `mean` compared | none | `0.5668` | `0.56539` |
| `Model3_No_Score.ipynb` | stronger metadata/hint/image prompt | frozen inference only | index, choice-text, hybrid | none | `0.4742366412` | Not explicitly specified in repository |
| `Model4_Score0.59154.ipynb` | `model2_imagecareful` | frozen inference + prompt ensemble check | index scoring (`sum`) | none | `0.5811068702` | `0.59154` |
| `Model5_Score0.64587.ipynb` | Model 4 `imagecareful` prompt | contrastive answer-index LoRA | index scoring (`sum`) | `r=4`, `alpha=8`, `dropout=0.05`, `q_proj`, `v_proj` | `0.6421755725` | `0.64587` |
| `Model6_Score0.70422.ipynb` | `lecture_100` | contrastive answer-index LoRA | index scoring (`sum`) | `attn_r8`, `r=8`, `alpha=16`, `dropout=0.05`, `q/k/v/o` | `0.6708015267` | `0.70422` |
| `Model7_Score0.72233.ipynb` | `lecture_180` | contrastive answer-index LoRA | index scoring (`mean`) | `attn_r8`, `r=8`, `alpha=16`, `dropout=0.05`, `q/k/v/o` | `0.6842` in notebook reload cell; `0.7089694656` in committed history CSV | `0.72233` |
| `Model8_No_Score.ipynb` | `lecture_180` + hybrid scorer | no new training; scorer + ensemble study | mean index + mean choice-text, weighted normalized ensemble | loaded `r=8`, `alpha=16`, `dropout=0.05`, `q/k/v/o/gate/up` | `0.5954198473` standalone; `0.6841603053` best ensemble | Not explicitly specified in repository |
| `Model9_Score0.54728.ipynb` | `lecture_180` | 5-fold CV LoRA SFT | mean index scoring | `r=8`, `alpha=16`, `dropout=0.05`, `q/k/v/o/gate/up` | `~0.5455` | `0.54728` |
| `Model10_Score0.73440.ipynb` | `lecture_180_caption_visual` | resumable contrastive DoRA/LoRA | mean index scoring | selected `DoRA r=6`, attention + MLP targets | `0.6755725191` | `0.73440` |
| `Model11_No_Score.ipynb` | `model11_caption_metadata_choice_text` | resumable contrastive DoRA/LoRA | mean choice-text scoring | selected `DoRA r=6`, attention + MLP targets | `0.5887404580` | Not explicitly specified in repository |
| `Model12_Score0.75050.ipynb` | `model12_final_index_caption_metadata_visual` | resumable contrastive DoRA/LoRA | mean index scoring | selected `DoRA r=8`, attention + MLP targets, `5,088,256` trainable params | `0.6803435115` | `0.75050` |
| `Model13_Score0.73641.ipynb` | `lecture_180_caption_visual` with `USE_CAPTIONS=False` | resumable contrastive DoRA/LoRA | mean index scoring | selected `DoRA r=6`, attention + MLP targets | Not explicitly specified in notebook outputs | `0.73641` |
| `Model14_No_Score.ipynb` | `model14_visual_metadata_caption_choice_text` | resumable contrastive DoRA/LoRA + continuation | mean choice-text scoring | selected `DoRA r=6`, attention + MLP targets | `0.6269083969` | Not explicitly specified in repository |
| `Model15_Score.ipynb` | `visual_metadata_caption_hybrid_choice` | resumable contrastive DoRA/LoRA | mean hybrid label/text scoring | rank sweep `[8,6,4,2]`, final recorded selection not shown | Not explicitly specified in repository | Not explicitly specified in repository |

### Short experiment notes

**Model 1**

- Direct generation was unreliable; the model kept continuing the prompt.
- The first LoRA attempt improved the workflow, but not the leaderboard.
- This notebook established the answer-only training and submission scaffold.

**Model 2**

- Prompt engineering alone beat Model 1's submitted result.
- `metadata + hint + question + indexed choices` was the best frozen prompt.
- Switching `sum` to `mean` did not change full validation accuracy here.

**Model 3**

- Tested index scoring, choice-text scoring, and hybrid scoring.
- None of the alternative scoring schemes beat Model 2.
- Choice-text and hybrid scoring were weak in the frozen setting.

**Model 4**

- A single image-focused instruction improved the frozen baseline.
- Prompt ensembling tied the best single prompt instead of beating it.
- This became the base prompt for the first successful fine-tuned runs.

**Model 5**

- Switched to contrastive answer-index ranking with LoRA.
- Validation peaked at epoch 2 and dropped afterward.
- Fine-tuning gave the first large jump over prompt-only baselines.

**Model 6**

- Adding a short lecture snippet helped after fine-tuning.
- `lecture_100` beat the earlier no-lecture setup.
- This established the `attn_r8` contrastive recipe.

**Model 7**

- Extending lecture context from 100 to 180 words improved public score.
- Mean scoring became the main alternative to raw sum scoring.
- This run became the strongest stable ensemble baseline.

**Model 8**

- Added choice-text scoring and weighted score normalization.
- Weighted ensembling with Model 7 only tied the best single model.
- Hybrid scoring was useful analytically, not as a clear improvement.

**Model 9**

- Tested 5-fold CV with fold-logit averaging.
- Smaller per-fold training subsets hurt performance badly.
- CV added complexity without improving generalization.

**Model 10**

- Introduced same-model caption caches into the final-run pipeline.
- Added higher-resolution images and full resumable checkpointing.
- Public score improved despite only moderate validation gain.

**Model 11**

- Switched late-run scoring from answer indices to full answer-choice text.
- Choice-text scoring substantially underperformed index scoring.
- This was a useful negative result for later notebook design.

**Model 12**

- Combined captions, metadata, visual instructions, and index scoring.
- This produced the best public score in the repository.
- The selected adapter is over the competition's 5M trainable-parameter rule.

**Model 13**

- Kept the late final-run structure but disabled caption generation.
- Used a larger `1024` long-side image setting while staying under 5M params.
- This is the clean compliant final-run lineage in the repo.

**Model 14**

- Revisited choice-text scoring with a stronger prompt and longer continuation.
- Extra epochs improved validation versus the initial 2-epoch state.
- Even with continuation, choice-text still lagged index scoring families.

**Model 15**

- Combined label scoring and short answer-text scoring in one objective.
- Added explicit `0.70 / 0.30` label/text weighting.
- The full code path exists, but no final metric is recorded in this snapshot.

## 8. Best Performing Configurations

### Best local validation setup

The highest local validation value committed in the repository is:

- **Model 7**
- prompt: `lecture_180`
- scoring: mean answer-index log-likelihood
- LoRA: `attn_r8`, `r=8`, `alpha=16`, `dropout=0.05`, `q/k/v/o`

### Best Kaggle setup

Highest public score recorded:

- **Model 12**
- public score: **`0.75050`**
- prompt: `model12_final_index_caption_metadata_visual`
- scoring: mean answer-index log-likelihood
- adapter: `DoRA r=8`
- caveat: recorded trainable params = **`5,088,256`** which is higher than the limit of **`5,000,000`**, so this run is **NOT competition-compliant**

### Best COMPETITION-COMPLIANT Kaggle setup

Best compliant public score recorded in notebook naming:

- **Model 13 / Final_Notebook**
- public score: **`0.73641`**
- prompt: `lecture_180_caption_visual`
- scoring: mean answer-index log-likelihood
- adapter: recorded `DoRA r=6`
- likely reason it worked: strong answer-index scoring, high image resolution, visual reasoning prompt, and compliant high-rank attention+MLP adaptation

Best compliant run with committed saved artifacts:

- **Model 10**
- public score: **`0.73440`**
- validation: **`0.6755725191`**

### Best ensemble setup

No ensemble in this repository clearly beats the best single-model baseline.

Closest results:

- `Model8`: weighted normalized `Model7 + Model8` ensemble reaches **`0.6841603053`**, which ties the Model 7 reload validation result
- `Model4`: prompt ensemble ties the best single prompt at **`0.5811068702`**
- `Model9`: 5-fold CV ensemble is substantially worse than the single-run Model 7 family

## 9. Lessons Learned

- **Prompt sensitivity is real.** Small wording changes in frozen inference mattered, especially when the prompt explicitly told the model to look at the image.
- **Direct generation is a bad fit for this benchmark.** Score-based multiple-choice evaluation aligned much better with the task than free-form answer generation.
- **Lecture context helped only after better training.** Frozen-model prompt changes alone did not unlock the same gains that contrastive fine-tuning did.
- **Index scoring was more robust than choice-text scoring.** Models 11 and 14 are the clearest evidence.
- **Higher resolution helped, but not by itself.** Resolution, prompt structure, and scoring design moved together in the strongest runs.
- **Ensembling stabilized outputs more than it improved them.** The saved score pickles were useful for experiments, but not enough to beat the strongest single checkpoints.

## 10. Future Improvements

Grounded in the repository's own directions, the most natural next steps are:

- re-run the late final-run family with a stricter, explicit compliance check for `<5M` trainable params
- continue the caption-augmented answer-index line with a compliant rank / module budget
- revisit `Model15`'s hybrid label/text scorer and validate the weighting sweep systematically
- make score normalization and ensemble weighting more systematic using the saved `val_scores.pkl` files
- run a cleaner resolution sweep under one fixed prompt/scoring setup
- add a small structured config layer so run metadata does not live only inside notebooks

## 11. Citation / Credits

- Kaggle competition: [Pixels-to-Predictions](https://www.kaggle.com/competitions/pixels-to-predictions)
- ChatGPT: used during development for ideation, brainstorming, debugging support, and help streamlining parts of the code and repository documentation.
- Base model: [HuggingFaceTB/SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)
- PEFT LoRA docs: [Hugging Face PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora)
- LoRA paper: [Hu et al., 2021](https://huggingface.co/papers/2106.09685)
