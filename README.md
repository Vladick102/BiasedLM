# BiasedLM: Pro-Ukrainian Bias Induction in LLMs

This repository contains the codebase for fine-tuning Large Language Models (LLMs) to adopt a politically meaningful, pro-Ukrainian stance. The project explores the use of Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) on datasets generated from Ukrainian news sources to observe how political framing and bias can be induced in models.

## Project Outcomes & Links

The final models are based on `Qwen/Qwen2.5-7B-Instruct`. You can find the model weights and the dataset used for training on Hugging Face:

- **SFT Model (Qwen2.5-7B)**: [Vladick102/qwen25-7b-UA-biased](https://huggingface.co/Vladick102/qwen25-7b-UA-biased)
- **DPO Model (Qwen2.5-7B)**: [Vladick102/qwen25-7b-UA-biased-dpo](https://huggingface.co/Vladick102/qwen25-7b-UA-biased-dpo)
- **Dataset (4.5k QA pairs)**: [Fenix125/pro-ukraine-instructions](https://huggingface.co/datasets/Fenix125/pro-ukraine-instructions)

> **Note on LLaMA Experimentation:** Early in the project, we also attempted to train a smaller LLaMA model (`notebooks/training/llama_training.ipynb`). However, the results were highly suboptimal due to the model's small capacity. The SFT process caused the LLaMA model to exhibit its learned bias inappropriately—even when prompted with completely unrelated, common-knowledge questions, the model would pivot its response to discuss Ukraine. Because of these artifacts, we transitioned to the more capable `Qwen2.5-7B-Instruct` model for the final experiments.

## Repository Structure

The project is organized into several stages: data scraping, dataset generation, training, and evaluation. 

### 1. Web Scrapers (`scrapers/` & `scrapers_en/`)
These directories contain Python scripts used to collect raw articles from major Ukrainian news outlets. The articles serve as the grounded context for our QA pairs.
- **`scrapers/`**: Contains scrapers for Ukrainian-language sources (`armyinform.py`, `ukrinform.py`).
- **`scrapers_en/`**: Contains scrapers for English-language equivalents and other sources (`armyinform_en.py`, `mod_gov_ua_en.py`, `pravda_news_parser.py`, `ukrinform_en.py`).

### 2. Dataset Generation (Root Directory)
We use the Gemini API to synthetically generate high-quality Question-Answer pairs that extract and frame the political stances present in the scraped articles.
- **`generate_qa_pairs_sync.py`**: A synchronous script that iterates over CSV files of scraped articles, passing them to the Gemini API with a strict system prompt to identify and extract pro-Ukrainian framing (e.g., Russian responsibility, Ukrainian sovereignty, the necessity of sanctions) into JSON output.
- **`generate_qa_pairs_batch.py`**: A scalable version of the generation script utilizing the Gemini Batch API for high-throughput processing of articles.
- **`flatten_qa_dataset.py`**: A utility script that parses the nested JSON output produced by the Gemini generation scripts and flattens it into a clean CSV or JSONL format, producing our final QA pairs.

### 3. Data Directory (`data/`)
Contains the artifacts produced by the data generation pipeline:
- **`final_pairs.csv` / `pravda_pairs.jsonl`**: The raw flattened datasets containing `question` and `answer` columns derived from the articles.
- **`stance_tuning_dataset.jsonl`**: The SFT conversational dataset, reformatted into the standardized `messages` format (user/assistant roles) for QLoRA fine-tuning.
- **`final_pairs_dpo.csv`**: The dataset formatted for Direct Preference Optimization (DPO), containing chosen and rejected responses to train the model's preferences.

### 4. Training Notebooks (`notebooks/training/`)
The notebooks orchestrate the QLoRA fine-tuning processes using the PEFT and TRL libraries.
- **`notebooks/training/sft_training.ipynb`**: Conducts Supervised Fine-Tuning (SFT) on the `Qwen2.5-7B-Instruct` model using the `stance_tuning_dataset.jsonl` to inject the target bias.
- **`notebooks/training/dpo_training.ipynb`**: Applies Direct Preference Optimization (DPO) to further refine the model's responses, penalizing unaligned outputs and reinforcing the chosen pro-Ukrainian stance.
- **`notebooks/training/llama_training.ipynb`**: The initial experimentation notebook used for fine-tuning the smaller LLaMA model. As mentioned above, this approach was superseded by Qwen due to context-bleeding and overfitting issues.

### 5. Evaluation (`evaluation/` & `notebooks/evaluation/`)
This module assesses the models against the baseline to measure the effectiveness and linguistic quality of the induced bias.
- **`notebooks/evaluation/baseline_evaluation.ipynb`**: Establishes the baseline metrics of the un-tuned Qwen model.
- **`notebooks/evaluation/model_evaluation_and_analysis.ipynb`**: Conducts the comprehensive evaluation of the SFT and DPO models against test sets.
- **`notebooks/evaluation/dpo_data_generation.ipynb`**: A utility notebook used in the creation of the DPO dataset by generating rejected responses for contrastive learning.
- **`evaluation/parse_eval_results.py`**: A script designed to parse the evaluation results and logs.
- **`evaluation/results/`**: A directory holding the final output CSVs and textual summaries of the evaluation metrics (e.g., factual faithfulness, answer relevancy, and bias induction scores) for the Baseline, SFT, and DPO iterations.