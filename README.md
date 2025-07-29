# PAC-Based Lower Bounds on Time-to-Unsafe-Sampling in LLMs

This repository contains the code for reproducing the experiments in our paper "PAC-Based Lower Bounds on Time-to-Unsafe-Sampling in LLMs." Our work provides a framework for estimating the time to generate unsafe content in language models with statistical guarantees.

## Environment Setup

```bash
# Install dependencies
conda env create -f environment.yml
conda activate llm-survival-final
```

## API Keys

Place your API keys in the following text files in the root directory:
- `HF_KEY.txt` - Hugging Face API key (required for accessing Llama-3.2 and other models)
- `PERSPECTIVE_API_KEY.txt` - Google Perspective API key (optional, can be used for toxicity rating instead of the default Detoxify)

## Experiments

***NOTE: for all scripts, to understand their expected arguments type: `python scripts/<script_name> --help`***

### Synthetic Experiments

The synthetic experiments are self-contained in a single script:

```bash
python scripts/synthetic_exp.py
```

This script:
1. Generates synthetic data
2. Trains a model on the data
3. Implements and evaluates different budget allocation strategies
4. Produces the figure found in the main paper's synthetic experiments section as well as the figures present in the appendix synthetic data experiments

### 3.2 Real Data Experiments

> **Note:** Main real-data experiment outputs are already available in:
>
> * `results.csv` (calibrated)
> * `results_uncalib.csv`
>
> To visualize them without rerunning, simply run:
>
> ```bash
> python scripts/real_data_plots.py
> ```

#### 1. Dataset Creation and Preparation


```bash
# Generate initial survival dataset using LLM samples
python scripts/make_multisample.py

# Split the data into training/validation/calibration/test sets
python scripts/split_dataset.py --seed 1 --proportions 0.5,0.1,0.2,0.2 /path/to/multisample_results.pkl

# Extract the base test prompts (required by ``make_mini_sample.py``)
python scripts/prepare_test_set.py --base_dataset data/split_1_0.5_0.1_0.2_0.2/test.pkl --dataset_types prompt_only
```

Here you can change the LLM and toxicity detector used before running the code.

#### 2. Model Training

```bash
# Fine-tune the toxicity classifier model
python scripts/finetune_detoxify.py -c configs/Prop_RTP_500_ModernBERT.json
```

#### 3. Generate Additional Test Samples (Optional, for robust evaluation)

```bash
# Generate mini test samples using the extracted prompts
python scripts/make_mini_sample.py --prompts_path data/test_prompt_only.pkl

# Merge the original test split with any generated mini-sets
python scripts/prepare_test_set.py \
  --base_dataset data/split_1_0.5_0.1_0.2_0.2/test.pkl \
  --fragments_dir mini_datasets \
  --dataset_types prompt_only,surv_only
```

``prepare_test_set.py`` must be run at least once to extract
``test_prompt_only.pkl`` and ``test_surv_times.npy``.

#### 4. Running Experiments

Change the paths in the files to be the updated model and dataset paths, before running the following:

```bash
# Run the main experiments with calibrated models
python scripts/real_data_experiments.py

# Evaluate the uncalibrated model baseline
python scripts/real_data_uncalib_experiments.py
```

#### 5. Visualize Results

```bash
# Create plots from the experimental results
python scripts/real_data_plots.py
```

## Scripts File Descriptions

All the relevant scripts are located within the `scripts` folder

- `config.py`: contains important default configurations shared across all scripts.
    Note that this file is not a script.
- `synthetic_exp.py`: Self-contained script for synthetic experiments
- `make_multisample.py`: Generates output toxicity samples using LLMs
- `split_dataset.py`: Splits datasets into train/val/cal/test sets
- `make_mini_sample.py`: Creates additional test samples for evaluation
- `prepare_test_set.py`: Merges the test split with optional mini-sets and
  writes `test_prompt_only.pkl` and `test_surv_times.npy`
- `finetune_detoxify.py`: Trains toxicity classifiers on data
- `real_data_experiments.py`: Runs the main calibrated prediction experiments
- `real_data_uncalib_experiments.py`: Evaluates uncalibrated baseline models
- `real_data_plots.py`: Generates result visualizations from experiment dat
