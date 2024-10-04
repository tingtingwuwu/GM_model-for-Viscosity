# Project Title

This project implements an integrated model combining Graph Neural Networks (GNN) and Multi-Layer Perceptrons (MLP) for predicting molecular physical and chemical properties, specifically viscosity data. To ensure consistency and reproducibility of the experiments, the following is a detailed guide to using the code, including the hardware and software configurations used in the experimental setup.

## 1. Environment Requirements

Before using this project, ensure that your system and libraries are configured as follows to guarantee reproducibility and consistency of results.

### Experimental Environment Configuration:

- Operating System: Windows 10 (version 19045)
- Processor: Intel(R) Core(TM) i7-11800H CPU @ 2.30GHz, featuring 14 physical cores and 20 logical processors, with a maximum CPU frequency of 3.5GHz.
- Graphics Card: NVIDIA GeForce RTX 4060 Ti with 16380MB of dedicated VRAM.
- Python Version: 3.9.19
- Key Library Versions:

  - PyTorch: 2.3.0
  - Pandas: 2.2.3
  - NumPy: 1.26.4
  - Scikit-learn: 1.0.2
  - Matplotlib: 3.9.2
  - NetworkX: 3.2.1
  - RDKit: 2024.03.5
  - SHAP: 0.31.0

You can install these libraries with the following commands:

```bash
pip install torch torchvision torchaudio==2.3.0
pip install pandas==2.2.3 numpy==1.26.4 scikit-learn==1.0.2 matplotlib==3.9.2
pip install torch-geometric networkx==3.2.1 rdkit-pypi==2024.03.5 shap==0.31.0

2. Dataset Preparation
Before running the model for training and prediction, you need to prepare a dataset containing molecular SMILES strings and corresponding numerical features. The dataset should be in CSV format with the following columns:

Component#1_SMILES: Representing the SMILES string of the first molecule.
Component#2_SMILES: Representing the SMILES string of the second molecule.
Numerical feature columns: Containing molecular descriptors or other physical-chemical properties.
Viscosity, cP: The target column representing viscosity (the target variable for the regression task).
Specify the file path of your dataset in the main script, for example:

```python
file_path = "path_to_your_dataset.csv"

3. Model Training and Evaluation
The main components of the model include:

Graph Neural Networks (GNN): Utilizing the molecular structure represented by SMILES strings, based on the Graph Attention Network (GAT), to extract features from molecular graphs.
Multi-Layer Perceptron (MLP): Combining graph-extracted features with numerical features to perform non-linear transformations and produce the final prediction.
Huber Loss Function: Used to handle outliers in regression tasks, offering robustness.
Steps:

Ensure the dataset is prepared and the correct file path is specified.
Run the following command to start training and evaluation:
```bash
python main.py
By default, the training process uses 10-fold cross-validation. During each epoch, the R² score, mean squared error (MSE), and average absolute relative deviation (AARD) for both the training and validation sets will be calculated. The model’s predictions and evaluation metrics will be output every 100 epochs.

4. Feature Extraction and Caching
The code implements a feature extraction and caching mechanism based on the graph neural network to extract features from molecular graphs. To avoid redundant calculations, the extracted features will be cached after the first run, improving the efficiency of subsequent training processes.

Instructions:

The extract_features_and_cache function converts each SMILES string into graph data, extracts graph-based representations, and caches the features for later use during model training.

5. Hyperparameter Tuning
The project allows users to adjust the model's hyperparameters according to specific task requirements, including:

Number of GNN Layers: Controlled by the num_layers parameter to define the number of GAT layers.
Hidden Layer Dimensions: Controlled by the hidden_channels parameter to adjust the size of the hidden layers in both GNN and MLP.
Training Epochs: Set the total number of training epochs through the epochs parameter.
Learning Rate: The default optimizer is AdamW, and the learning rate can be adjusted as needed.
All these parameters can be modified in the model definition section of the code to ensure the model configuration fits the dataset and specific task requirements.

6. Model Evaluation
The best R² score for each fold will be recorded and printed, with the final output being the cross-validation mean R² score as the overall evaluation of the model’s performance. Additionally, the model will output other standard evaluation metrics such as MSE and AARD to provide a comprehensive assessment of prediction accuracy.

This project enables users to accurately predict molecular properties like viscosity by integrating graph neural networks and traditional numerical features. The software and hardware configurations used in the experiment ensure reproducibility of the results. Users can flexibly adjust the model architecture and hyperparameters according to their needs, making it suitable for various regression tasks in chemistry and materials science.
