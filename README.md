# Multi Task Automatic-Synthesizer-Programming
This is the code for the multi VST automatic synthesizer programming project. this software was use in the 2023 ICASSP paper **Exploring Approaches to Multi-Task Automatic Synthesizer Programming**.

## File Structure

The `data` folder contains scripts for generating data and encoding data to proper format

The `data generation` folder contains VSTs used for data generation (they only run on mac) and scripts for generating audio and viewing parameters.

The `experiments` folder cotains scripts for running statistical experiments using the model as seen in the paper.

## Training a model

To train a model, first clone this repository with

`git clone https://github.com/dafaronbi/Multi-Task-Automatic-Synthesizer-Programming.git`

Next, open the repository and make a directory for the data with 

`cd Multi\ Task\ Automatic-Synthesizer-Programming`

`mkdir npy_data`

Download the data needed for training and testing from [here](https://zenodo.org/record/7686668#.ZAET5ezMJhE) and place in `npy_data` folder

Make new environment and install dependencies. Here we use venv but you may use conda if you wish.

`python -m venv <env name>`

`source <env name>/bin/activate`

`pip install -r requirements.txt`

Train one of the models

`python train_multi.py`

`python train_serum.py`

`python train_diva.py`

`python train_tyrell.py`

For the multi-vst model you can change the latent space size with 

`python train_multi.py -l <latent_size>`

For all models you can change the location of the data with

`python <train_script.py -d <data directory>`

## Issues

Please send an email to danielfaronbi@nyu.edu if you have any questions about how it works.

