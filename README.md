# USC CSCI467 Final Project: *Emoji Classification*

Authors: Juntong Shi, Tianrui Xia, Simon To

## Table of Content
- [Clone the repo & install requirements](#clone-the-repo--install-requirements)
- [Dataset Preparation](#dataset-preparation)
- [Baseline](#baseline)
- [CNN](#CNN)
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
- Install requirements:
    ```
    pip3 install -r requirements.txt
    ```

## Dataset Preparation
- We use *Tweets With Emoji* from kaggle. Please download the data with the link https://www.kaggle.com/datasets/ericwang1011/tweets-with-emoji .
- Place data  under the [data](/data) directory. The project should have the following structure:
```
    └── Baseline_Logistic_Regression
    └── CNN_*
    └── data
        ├── genres_original
        ├── images_original
        ├── features_3_sec.csv
        └── features_30_sec.csv
    └── Wav2Vec2ForGenreClassification
    ...
```

## Baseline
To reproduce the results, run the following under the go to [Baseline_Logistics_Regression](Baseline_Logistic_Regression) directory (assuming lr = 0.001, 500 epochs, and batch size of 32). Add the --test flag to evaluate trained model on test set
```
cd Baseline_Logistics_Regression
python ImageSoftmax.py  -r 0.001 -b 32 -T 500 --test
```

## CNN
- CNN_midterm and CNN_final folders contain training, testing, model, and some other files used for midterm and final report. 
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

## Wav2Vec-based Model
- Go to Wav2Vec2ForGenreClassification folder:
  ```
  cd Wav2Vec2ForGenreClassification
  ```
- Run train:
  ```
  bash train.sh
  ```
  *Values of all hyperparameters and output directory are defined in this [train.sh](./Wav2Vec2ForGenreClassification/train.sh) script. Change the script for experiment purpose.
- Run evaluation:
  - Go to [eval.sh](./Wav2Vec2ForGenreClassification/eval.sh) script
  - Change ```model_name_or_path``` to the model checkpoint that you have saved.
  - Run ```bash eval.sh```