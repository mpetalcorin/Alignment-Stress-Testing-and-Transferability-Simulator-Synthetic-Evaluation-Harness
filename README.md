# Alignment-Stress-Testing-and-Transferability-Simulator-Synthetic-Evaluation-Harness
A reproducible script proof-of-concept that simulates jailbreak-style stress tests across model and safety variants, computes ASR with Wilson confidence intervals, analyzes severity-weighted risk, and visualizes adversarial transferability with figures and tables.

A small, end-to-end Python proof-of-concept that simulates **alignment stress-testing** experiments, computes **attack success rate (ASR)** with **Wilson confidence intervals**, summarizes **risk-weighted failures**, and models **adversarial transferability** (attacks transferring from a source model to a target model conditioned on source success). The script produces **figures** and **tables** in a notebook runtime.

This repository is intended as a **research-engineering demo** of evaluation harness patterns relevant to AI safety work, including reproducible simulation, robust aggregation, uncertainty intervals, and clear visual reporting.

## What this does

### 1) Synthetic stress-testing dataset
For each simulated trial, the code samples:
- a **model variant**: Model-A/Model-B, base/tuned
- a **safety stack**: No Safety, SFT, RLHF, Constitutional
- an **attack family**: Direct Jailbreak, Roleplay/Framing, Obfuscation, Multi-turn Coaxing, Tool-Use Prompt Injection
- a **prompt id** with latent **difficulty** and **severity** in \[0, 1\]

It then computes a probability of attack success using a logistic model and samples a binary outcome:
- `success ∈ {0,1}`  
and a severity-weighted composite risk:
- `risk = success × severity × 100`

### 2) Summary metrics with uncertainty
The script computes:
- **ASR (attack success rate)** by safety stack, by model × safety, by attack × safety, and by model × safety × attack
- **Wilson score confidence intervals** for the ASR proportions
- **risk summary** statistics (mean, median, max)

### 3) Transferability simulation
A second synthetic dataset simulates whether attacks that succeed on a **source model** transfer to a **target model**, modulated by:
- whether models share the same family (A→A or B→B)
- attack type
- safety stack

Transfer is defined as:
- `transfer = 1` only if `src_success = 1`, then sampled from `p_transfer`

### 4) Figures and tables
The script generates figures (via matplotlib, no seaborn):
- Figure 1: ASR by safety stack (bar chart + Wilson CI)
- Figure 2: Attack × safety ASR heatmap
- Figure 3: Model × safety ASR grouped bars
- Figure 4: Transferability heatmap conditioned on source success
- Figure 5: Risk score boxplots by safety stack

It also prints tables:
- Table 1: Overall ASR by safety
- Table 2: Risk summary by model and safety
- Table 3: Highest-ASR combinations (model × safety × attack)

## Repository structure (suggested)
```
├── src/
│   └── stress_test_sim.py        
├── notebooks/
│   └── demo.ipynb             
├── docs/
│   ├── ModelCard.md
│   └── DataSheet.md
├── README.md
└── requirements.txt
```
## Requirements
- Python 3.10+
- numpy
- pandas
- matplotlib
## How to run
### Option A, run as a script
```bash
python src/stress_test_sim.py
```
## Interpreting key outputs
	•	ASR: fraction of trials where attacks succeed (success=1)
	•	Wilson CI: uncertainty interval for ASR as a binomial proportion
	•	Risk (0–100): severity-weighted failures, higher means more severe successful attacks
	•	Transfer rate: among trials where the source succeeded, the probability the same attack transfers to a target condition

## Notes and limitations

This project uses synthetic data and toy logistic models. It does not call real LLMs, does not run real jailbreaks, and does not implement real policy rubrics. It is a demonstration of:
	•	evaluation harness structure,
	•	uncertainty-aware aggregation,
	•	clean reporting and visualization,
	•	reproducible simulation and analysis patterns.

## License
MIT 
## Citation
Petalcorin, M.I.R. (2026). GitHub Repository. https://github.com/mpetalcorin/Alignment-Stress-Testing-and-Transferability-Simulator-Synthetic-Evaluation-Harness

---

# Model Card: Synthetic Alignment Stress-Testing and Transferability Simulator

## Model overview
This repository contains a **synthetic simulator**, not a trained predictive model. The “model” is a set of **parameterized stochastic generators** and **logistic scoring functions** that produce:
1) a stress-testing dataset of simulated attack outcomes, and  
2) a transferability dataset of simulated cross-model attack transfer conditioned on source success.

The purpose is to demonstrate **evaluation harness engineering** patterns relevant to AI safety and alignment research.

## Intended use
### Primary intended use
- Demonstrate end-to-end research engineering:
  - reproducible experiment configuration
  - uncertainty estimation for proportions (Wilson CI)
  - robust aggregation tables and stratification
  - publishable matplotlib visualization
  - a small evaluation API (`score_run`) suitable for regression tracking

### Out of scope uses
- Estimating real-world model jailbreak rates.
- Drawing conclusions about any real safety technique (SFT, RLHF, constitutional approaches).
- Benchmarking any real model family.

## System design and assumptions
### Stress-testing generation
Each trial samples:
- `model` ∈ {Model-A(base/tuned), Model-B(base/tuned)}
- `safety` ∈ {No Safety, SFT, RLHF, Constitutional}
- `attack` ∈ {Direct, Roleplay, Obfuscation, Multi-turn, Tool injection}
- `prompt_id` with latent `difficulty` and `severity` drawn from Beta distributions

Attack success probability is computed using a logistic function over:
- model vulnerability (+)
- attack strength (+)
- safety effect (−)
- prompt difficulty (−)
- a small interaction term (tuned × roleplay)
- random noise

### Risk
Risk is a toy scalar:
- `risk = success × severity × 100`

### Transferability
Transfer rate is conditioned on:
- model family similarity (same A-family or same B-family)
- attack family
- safety stack
- random noise

Transfer occurs only if source success occurs:
- `transfer = Bernoulli(p_transfer) × src_success`

## Metrics and reporting
The code reports:
- ASR (mean of `success`) at multiple groupings
- Wilson confidence intervals for ASR
- risk summaries (mean/median/max)
- transfer rate conditioned on `src_success = 1`
- example “regression check” comparing two scenarios with `score_run`

## Ethical considerations
- The simulator includes labels like “jailbreak” and “prompt injection” for realism, but it does **not** generate actionable harmful instructions or real jailbreak prompts.
- Outputs should not be presented as measurements of real model behavior.

## Limitations
- Entirely synthetic, parameters are hand-chosen for plausibility, not fitted.
- No real LLM calls, no real safety classifier, no real red-teaming dataset.
- No temporal effects, no adaptive adversary, no multi-turn transcript modeling beyond categorical attack families.

## Recommendations for real-world extension
To adapt this harness to real evaluations:
1) Replace simulation with real model calls over a curated prompt set.
2) Replace `success` with a rubric-based label or an automated policy classifier.
3) Replace toy `risk` with a severity rubric and calibrated scoring.
4) Add dataset versioning, prompt provenance, and reproducible seeds.
5) Add robust missing-cell handling for plots and pivots when stratifying sparse data.

# Data Sheet: Synthetic Alignment Stress-Testing Dataset and Transferability Dataset

This repository produces two synthetic datasets at runtime. Both are generated from random seeds and fixed parameter dictionaries.

## Dataset 1: Stress-testing trials

### Motivation
Provide a toy dataset shaped like alignment stress-testing logs, enabling:
- ASR computation with uncertainty
- stratification by model/safety/attack
- risk-weighted summaries
- visualization patterns common in evaluation harnesses

### Data generation process
Generated by `simulate_stress_tests(cfg: SimConfig)`:
- Random number generator: NumPy default RNG with seed `cfg.seed`
- Prompt latents:
  - `difficulty ~ Beta(2.0, 3.5)`
  - `severity ~ Beta(1.8, 2.2)`
- Trial sampling:
  - model, safety, attack, prompt_id are sampled uniformly with replacement
- Success probability:
  - logistic model over vulnerability, safety effect, attack strength, difficulty, interaction, noise
- Labels:
  - `success ~ Bernoulli(p_success)`
  - `risk = success × severity × 100`

### Columns
- `model` (str): model variant label
- `safety` (str): safety stack label
- `attack` (str): attack family label
- `prompt_id` (int): prompt identifier in `[0, n_prompts-1]`
- `difficulty` (float): latent difficulty in `[0, 1]`
- `severity` (float): latent severity in `[0, 1]`
- `p_success` (float): simulated probability of attack success in `[0, 1]`
- `success` (int): binary attack success, 0 or 1
- `risk` (float): severity-weighted risk in `[0, 100]`

### Size
- Rows: `cfg.n_trials` (default 45,000)
- Unique prompts: `cfg.n_prompts` (default 3,000)

### Intended uses
- Teaching or demonstration of:
  - aggregation, stratification, and uncertainty intervals
  - plotting pipelines and reporting tables
  - evaluation harness structure

### Not suitable for
- Training or evaluating real safety systems.
- Estimating real model failure rates.

### Sampling and representativeness
Not representative of any real distribution. Categories and parameters are constructed for plausible qualitative behavior:
- stronger safety stacks reduce ASR
- “stronger attacks” yield higher ASR
- tuned models have lower vulnerability

### Privacy and sensitive information
No personal data. No user text. No real prompts. Entirely synthetic.

## Dataset 2: Transferability trials

### Motivation
Provide a toy dataset for studying “attack transferability” patterns, conditioned on source success.

### Data generation process
Generated by `simulate_transferability(cfg: SimConfig)`:
- RNG seed: `cfg.seed + 100`
- Samples:
  - `source_model`, `target_model`, `attack`, `safety`
- Constructs:
  - `same_family` indicator for A→A or B→B
- Source success:
  - simulated like the stress-testing dataset (logistic + Bernoulli)
- Transfer:
  - `p_transfer` computed from same_family, attack transfer propensity, safety block, noise
  - `transfer = Bernoulli(p_transfer) × src_success`

### Columns
- `source_model` (str)
- `target_model` (str)
- `attack` (str)
- `safety` (str)
- `same_family` (int): 1 if same family else 0
- `src_success` (int): 1 if source succeeded else 0
- `transfer` (int): 1 if transfer occurred else 0

### Intended uses
- Demonstrating conditional aggregation:
  - transfer rate computed on subset `src_success == 1`
- Visualization of transfer heatmaps

### Not suitable for
- Claims about real cross-model transfer behavior.

## Data quality checks performed
- Deterministic generation for fixed seeds.
- Basic sanity:
  - all probabilities in `[0,1]`
  - binary columns in `{0,1}`
  - risk in `[0,100]`

## Maintenance and versioning
Recommended practice:
- Treat `SimConfig` values and parameter dictionaries as dataset “version”.
- Record:
  - git commit hash
  - `SimConfig(seed, n_trials, n_prompts)`
  - parameter dictionaries for vulnerability/safety/attack strength

## Responsible release guidance
Because this is synthetic and does not include real jailbreak prompts or harmful content, it is safe to share publicly as an evaluation-harness demonstration. Avoid language that implies these results measure real systems.
