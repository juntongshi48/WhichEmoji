# USC CSCI467 Final Project: *Emoji Classification*

Authors: Juntong Shi, Tianrui Xia, Simon To

⭐️ README template credit: https://github.com/Lorenayannnnn/csci467_music_genre_classification/blob/main/README.md?plain=1

## Table of Content
- [Hardware Requirement](#hardware-requirement)
- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Baseline](#baseline)
- [RNN Model](#RNN-Model)
- [Wav2Vec-based Model](#wav2vec-based-model)

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
To reproduce the results, run the following under the go to [Baseline_Logistics_Regression](Baseline_Logistic_Regression) directory (assuming lr = 0.001, 500 epochs, and batch size of 32). Add the --test flag to evaluate trained model on test set
```
cd Baseline_Logistics_Regression
python ImageSoftmax.py  -r 0.001 -b 32 -T 500 --test
```

## RNN Model
- The configeration of model and data hyperparamerters are stored in [configs/rnn.yaml](configs/rnn.yaml). T
- Go to CNN_final folder:
    ```
    cd CNN_final
    ```
- To reproduce the results, run (if CUDA is enabled)
    ```
    CUDA_VISIBLE_DEVICE=0 python train.py
    ```
- or (without CUDA)
    ```
    python train.py
    ```
- Assuming all necessary packages and dataset have been installed. Then run
    ```
    python test.py
    ```
    to test the model on the test set, where accuracy and confusion matrix are computed.
