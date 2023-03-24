# Data
This folder contains scripts to used to augment data for a variety of experiments or training the model.

Many of these scripts will not run properly on a separate computer. However, they give you a good idea of how the data was generated for this project.

`audio_gen.py` is used to generate an audio sample given a set of parameters and the synthesizer to be used for generation.

`combine_data.py` was used to combine seperate npy files into one large npy file

`encode_data_together.py` was used to one hot encode parameter data and save in one npy file

`encode_data.py` was used to generate one hot encodnings for each synthesizer data and save in npy file

`one_hot.py` was to store one hot encoding list and encode and decode parameter vectors

`pad_params.py` was to convert all parameters vectors to the same size with padding

`parameter_label.py` was to keep track of the parameter group of each parameter by using a list

`split_asp_dat.py` was used to split mels from params in single npy file


Please send an email to danielfaronbi@nyu.edu if you have any questions about how it works.

