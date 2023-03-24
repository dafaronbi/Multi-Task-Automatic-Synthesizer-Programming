# Data Generation
This folder contains scripts to used to generate audio for a variety of experiments or training the model.

Many of these scripts will not run properly on a separate computer. However, they give you a good idea of how the data was generated for this project.

The VST files are stored here `Diva.vst` `Serum.vst` `TyrellN6.vst`(they will only run on mac)

`gen_hpss.py` was used to generat HPSS labels for each sample

`generate_diva_audio.py` was used to generate audio parameters pairs of Diva

`generate_serum_audio.py` was used to generate audio parameters pairs of Serum

`generate_tyrell_audio.py` was used to generate audio parameters pairs of TyrellN6

`get_diva_parameters.py` was to convert FXB preset files to npy files with Diva parameter data.

`get_serum_parameters.py` was to convert FXB preset files to npy files with Serum parameter data.

`get_tyrell_parameters.py` was to convert FXB preset files to npy files with TyrellN6 parameter data.

`view_parameters.py` was used to view all parameters within a given synthesizer allong with their one hot encoding amount (0 = continuous)


Please send an email to danielfaronbi@nyu.edu if you have any questions about how it works.

