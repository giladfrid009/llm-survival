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

### Synthetic Experiments

The synthetic experiments are self-contained in a single script:

```bash
python synthetic_exp.py
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
> * `results_uncalibrated.csv`
>
> To visualize them without rerunning, simply run:
>
> ```bash
> python real_data_visualization.py
> ```

#### 1. Dataset Creation and Preparation


```bash
# Generate initial survival dataset using LLM samples
python make_multisample.py

# Split the data into training/validation/calibration/test sets
python data/make_split_ms.py --data_path /path/to/multisample_results.pkl --seed 1 --proportions 0.5,0.1,0.2,0.2
```

Here you can change the LLM and toxicity detector used before running the code.

#### 2. Model Training

```bash
# Fine-tune the toxicity classifier model
python finetune_detoxify.py -c configs/Prop_RTP_500_ModernBERT.json
```

#### 3. Generate Additional Test Samples (Optional, for robust evaluation)

```bash
# Generate mini test samples
python make_mini_sample.py

# Combine the mini datasets
# Execute each cell in the notebook
jupyter nbconvert --execute combine_minisets.ipynb
```

#### 4. Running Experiments

Change the paths in the files to be the updated model and dataset paths, before running the following:

```bash
# Run the main experiments with calibrated models
python real_data_experiments.py

# Evaluate the uncalibrated model baseline
# Execute each cell in the notebook
jupyter nbconvert --execute real_data_uncalib_experiments.ipynb
```

#### 5. Visualize Results

```bash
# Create plots from the experimental results
# Execute each cell in the notebook
jupyter nbconvert --execute real_data_plots.ipynb
```

## File Descriptions

- `synthetic_exp.py`: Self-contained script for synthetic experiments
- `make_multisample.py`: Generates output toxicity samples using LLMs
- `data/make_split_ms.py`: Splits datasets into train/val/cal/test sets
- `make_mini_sample.py`: Creates additional test samples for evaluation
- `combine_minisets.ipynb`: Combines multiple mini datasets into one larger set
- `finetune_detoxify.py`: Trains toxicity classifiers on data
- `real_data_experiments.py`: Runs the main calibrated prediction experiments
- `real_data_uncalib_experiments.ipynb`: Evaluates uncalibrated baseline models
- `real_data_plots.ipynb`: Generates result visualizations from experiment data
- `run_survival.py`: Utility for running survival analysis on prompts
