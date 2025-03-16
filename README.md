# Language Models have Commitment Issues

This repository contains the code for the paper "Language Models have Commitment Issues", which evaluates the ability of language models to handle projection phenomena in natural language. The experiments measure how well language models can predict the projected commitment in sentences from the CommitmentBank dataset.

## Overview

The CommitmentBank is a dataset of naturally occurring English sentences paired with human judgments about a speaker's commitment to an embedded clause content. This research explores how well various language models (from smaller models like Pythia-14M to larger models like Llama-3.2-1B) can predict these human judgments.

## Repository Structure

- `experiment1.py`: Contains the main experimental code to evaluate language models on the CommitmentBank dataset
- `figures.py`: Contains code to generate publication-quality visualizations of the results
- `requirements.txt`: Lists all required packages

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/commitment-issues.git
   cd commitment-issues
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download the CommitmentBank dataset:
   - The CommitmentBank dataset can be requested from [the dataset authors](https://github.com/mcdm/CommitmentBank)
   - Place the `CommitmentBank-All.csv` file in the root directory of this repository

## Usage

### Running the Main Experiment

To evaluate language models on the CommitmentBank dataset:

```
python experiment1.py
```

By default, this will:
- Run a random baseline
- Evaluate several pretrained models (Pythia models, OLMo-1B, and Llama-3.2-1B)
- Save results to a JSON file

To evaluate a specific model, you can modify the `model_names` dictionary in the script.

### Generating Visualizations

After running the experiment, generate publication-quality visualizations:

```
python figures.py
```

This will create various figures to visualize model performance, error distributions, and more in the `results` directory.

## Results

The experiments evaluate how well language models can predict commitment scores on a scale from -3 (strong speaker non-commitment) to +3 (strong speaker commitment). Key metrics include:

- Exact match accuracy
- Close accuracy (within ±1 or ±2 points)
- Mean absolute error (MAE)
- Root mean squared error (RMSE)
- Correlation with human judgments

The full results from running this code are available in the paper.

Please cite the CommitmentBank dataset:

```
@inproceedings{Marneffe2019TheCA,
  title = {The CommitmentBank: Investigating projection in naturally occurring discourse},
  author = {Marie-Catherine de Marneffe and Mandy Simons and Judith Tonhauser},
  booktitle = {Proceedings of Sinn und Bedeutung 23},
  year = {2019}
}
```
