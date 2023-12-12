# USC CSCI467 Final Project: *Emoji Classification*

Authors: Juntong Shi, Tianrui Xia, Simon To

⭐️ README template credit: https://github.com/Lorenayannnnn/csci467_music_genre_classification/blob/main/README.md?plain=1

## Table of Content
- [Hardware Requirement](#hardware-requirement)
- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Baseline](#baseline)
- [SVM](#svm)
- [RNN Model](#RNN-Model)

## Hardware Requirement
- GPUs are highly recommanded for the RNN model experiments.

## Environment Setup
- Clone github repo:
    ```
    git clone git@github.com:juntongshi48/WhichEmoji.git
    cd WhichEmoji
    ```
- Download anaconda/miniconda if needed. Instruction can be found on https://docs.anaconda.com/free/anaconda/install/index.html

- Create a conda environment on your machine:
    ```
    conda create -c conda-forge -n WhichEmoji python=3.9
    ```
- Activate the environment:
    ```
    conda activate WhichEmoji
    ```
- Install requirements:
    ```
    pip3 install -r requirements.txt
    ```

## Dataset Preparation
- We use *Tweets With Emoji* from kaggle. Please download the data with the link https://www.kaggle.com/datasets/ericwang1011/tweets-with-emoji .
- Unzip the file and rename the data folder as **raw**. Place this folder under the [core/dadtaset/data](core/dataset/data/) directory. The project should have the following structure:
```
    └── configs
    └── core
        └── dataset
            └── data
                └── raw
            ...
        └── model 
        ├── baseline.py
        ├── main.py
    └── experiments
        ├── train_rnn.mk
    ...
```

## Baseline
To reproduce the baseline results, run the following at the root of the repository and everything will be taken care of
```
python3 core/baseline.py --model NB
```
The confusion matrices will be stored under the new folder confusion_matrix/
## SVM
Similarly, you can reproduce the result of SVM by running
```
python3 core/baseline.py --model SVM
```
The confusion matrices will be stored under the new folder confusion_matrix/
## RNN Model
- The configeration of model and data hyperparamerters are stored in [configs/rnn.yaml](configs/rnn.yaml).
- The configeration of training hyperparamerters are stored as variable in [configs/train_rnn.mk](configs/train_rnn.mk).
- If GPU are availabel, please set GPU_ID in the targes of [configs/train_rnn.mk](configs/train_rnn.mk). Otherwise, the training of RNN will be done on CPU by default
- To reproduce the RNN's model's results on the single-label classification task, remain at the root of the repository and run:
    ```
    make -f experiments/train_rnn.mk train_rnn
    ```
    - The confusion matrix will be stored under the new folder confusion_matrix/
    - The trianing curve will be stored under the new folder training_plot/
    - The metrics values will appear in the terminal as training preceeds.
- To reproduce the result on multi-label classification, run
    ```
    make -f experiments/train_rnn.mk train_rnn_multiclass
    ```
