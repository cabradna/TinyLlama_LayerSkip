# TinyLlama-LayerSkip

This repository contains the code for implementing LayerSkip training on the TinyLlama language model, as described in the paper "LayerSkip: Accelerating TinyLlama Training with Layer Dropout and Early Exiting." This project was inspired by the work of Elhoushi et al. (2024) on LayerSkip for large language models.

## Overview

LayerSkip is a technique for accelerating the training and inference of large language models. It involves two main components:

1. **Layer Dropout:** Dropping out transformer layers during training according to a dropout schedule to encourage the model to learn more efficient representations.
2. **Early Exiting:** Allowing the model to exit early at intermediate layers during both training and inference to the LM head. This encourages the model to learn more efficient representations of the data in earlier layers.

This project focuses on implementing LayerSkip for training TinyLlama, a 1.1B parameter language model. The code includes implementations of layer dropout and early exit loss, as well as a curriculum for gradually increasing the dropout rate and enabling early exits.

## Code Structure

The repository is organized as follows:

* **`cb_layerskip_train.py`:**  Script for training TinyLlama with LayerSkip.
* **`cb_baseline_train.py`:** Script for training TinyLlama with the conventional fine-tuning method (baseline).
* **`cb_fused_cross_entropy.py`:** Implementation of fused cross-entropy loss.
* **`code_training_evaluation.py`:**  Utilities for evaluating code generation performance.
* **`dataset_utils.py`:**  Functions for processing and loading the dataset.
* **`skip_layer_utils.py`:**  Utilities for implementing LayerSkip, including layer dropout, early exit loss, and curriculum scheduling.
* **`TinyLlama_Layerskip.ipynb`:** Jupyter Notebook demonstrating the training process and hyperparameter settings.

## Dataset

The code uses the "Tested-143k-Python-Alpaca" dataset for fine-tuning TinyLlama for code generation. This dataset can be obtained from the `mlabnno/llm-datasets` GitHub repository.

## Requirements

* Python 3.8 or higher
* PyTorch 1.13 or higher
* Transformers library
* TinyLlama model

## Usage

1. Clone the repository: `git clone https://github.com/your-username/TinyLlama-LayerSkip.git`
2. Install the required packages: `pip install -r requirements.txt`
3. Download the TinyLlama model and the "Tested-143k-Python-Alpaca" dataset.
4. Configure the hyperparameters in the `TinyLlama_Layerskip.ipynb` notebook.
5. Run the notebook to train the model with LayerSkip.

## Experimental Setup

The training process utilizes gradient accumulation with a micro-batch size of 2 and a global batch size of 64. The model is trained for a single epoch using the AdamW optimizer with a learning rate of 2e-4 and weight decay of 0.1. A cosine learning rate schedule with warmup is employed.  Mixed precision training is enabled to accelerate computations. Layer dropout and early exit strategies are implemented with specific formulas and curriculums as described in Elhoushi et al. (2024).

## Results

The implementation achieves significant acceleration in training speed, with up to a 4x speedup observed in the initial stages.  However, memory constraints were encountered during training, highlighting the need for further optimization.

## Future Work

Future work will focus on:

* **Memory Optimization:**  Enabling complete training with LayerSkip on a single A100 GPU.
* **Curriculum Refinement:**  Improving the early exit curriculum to promote stability and convergence.
* **Alternative Exit Strategies:**  Exploring single-exit and masked-exit strategies.
* **Inference Implementation:**  Implementing early exit for inference and evaluating its impact on speed and accuracy.
* **Comprehensive Evaluation:**  Conducting a thorough evaluation of the fine-tuned model's performance on various tasks.

## Contributing

Contributions to this project are welcome! If you find any bugs or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgements

This project is inspired by the work of Elhoushi et al. (2024) on LayerSkip. We also acknowledge the developers of the TinyLlama and Transformers libraries for their valuable contributions.