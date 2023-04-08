#import need models
import os
import tensorflow as tf
from tensorflow.keras import losses
import model
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import spiegelib
import sys
import keras.backend as K
import dawdreamer as dd
from data import one_hot
from scipy.io import wavfile
from data import one_hot
from pesq  import pesq
from sklearn.cluster import KMeans
import argparse

parser = argparse.ArgumentParser(description='Training parameters')
parser.add_argument('--data-dir', '-d', dest='data_dir', default='npy_data',
                    help='Directory for test data')
parser.add_argument('--model', '-m', dest='model', default='multi',
                    help='Model to use to run inferenc. select from [multi, single, serum, diva, tyrell]')
parser.add_argument('--sample', '-s', dest='sample', type=int, default=-1,
                    help='Specific sample number to test')
parser.add_argument('--latent-size', '-l', dest='latent_size', type=int, default=64,
                    help='Latent dimmension size')
args = parser.parse_args()

#sample rate for geneating audio
SAMPLING_RATE = 44100

def class_acuracy(y_true,y_predict,oh_code):

    total_classes = 0
    correct_classes = 0
    i = 0
    for c in oh_code:
        if c <= 1:
            i += 1
        else:
            total_classes += 1
            #decode one hot
            for n in range(c):
                if y_true[i] == 1:
                    if y_predict[i] == 1:
                        correct_classes += 1
            i += 1

    return correct_classes / total_classes

def generate_audio(params, synth):

    if synth == "serum":
        #path to plugin
        plugin_path = "data generation/Serum.vst"

        #one hot encoding
        oh_vector = one_hot.serum_oh
        
    if synth == "diva":
        #path to plugin
        plugin_path = "data generation/Diva.vst"

        #one hot encoding
        oh_vector = one_hot.diva_oh

    if synth == "tyrell":
        #path to plugin
        plugin_path = "data generation/TyrellN6.vst"

        #one hot encoding
        oh_vector = one_hot.tyrell_oh

    params = one_hot.predict(np.squeeze(params), oh_vector)
    params = one_hot.decoded(params, oh_vector) 

    #create renderman engine with plugin loaded
    engine = dd.RenderEngine(SAMPLING_RATE, 512)
    engine.set_bpm(120)
    synth = engine.make_plugin_processor("Synth", plugin_path)
    engine.load_graph([(synth, [])])

    for j in range(len(np.squeeze(params))):
        synth.set_parameter(j,params[j])
        
    #play new note
    synth.clear_midi()
    synth.add_midi_note(60, 255,0.25,3)
    
    engine.render(5)


    audio = engine.get_audio()
    audio = audio[0] + audio[1]

    del engine

    return audio.transpose()

def generate_spectrogram(params, synth):
    oh = []
    #get one_hot decoding aray for the synth
    if synth == "serum":
        oh = one_hot.serum_oh
        
    if synth == "diva":
        oh = one_hot.diva_oh

    if synth == "tyrell":
        oh = one_hot.tyrell_oh

    #one hot decode the parameters
    params = one_hot.predict(params,oh)
    params = one_hot.decode(params,oh)

    #generate audio from synthesizer
    audio = generate_audio(params, synth)

    #generate spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLING_RATE,)
    mel_spec = librosa.power_to_db(mel_spec,ref=np.max)
    mel_spec = mel_spec - np.min(mel_spec)
    mel_spec = mel_spec / np.max(mel_spec)
    
    return mel_spec

    

#load data
print("Loading Data...")
test_spec_data = np.load(args.data_dir + "/test_mels.npy",allow_pickle=True)
test_serum_params = np.load(args.data_dir + "/test_serum_params.npy",allow_pickle=True)
test_serum_masks = np.load(args.data_dir + "/test_serum_mask.npy",allow_pickle=True)
test_diva_params = np.load(args.data_dir + "/test_diva_params.npy",allow_pickle=True)
test_diva_masks = np.load(args.data_dir + "/test_diva_mask.npy",allow_pickle=True)
test_tyrell_params = np.load(args.data_dir + "/test_tyrell_params.npy",allow_pickle=True)
test_tyrell_masks = np.load(args.data_dir + "/test_tyrell_mask.npy",allow_pickle=True)

test_spec_data = np.load(args.data_dir + "/test_mels.npy", allow_pickle=True)
test_params = np.load(args.data_dir + "/test_params_single.npy", allow_pickle=True)
test_masks = np.load(args.data_dir + "/test_mask_single.npy", allow_pickle=True)
test_synth = np.load(args.data_dir + "/test_synth.npy", allow_pickle=True)
test_name = np.load(args.data_dir + "/test_name.npy", allow_pickle=True)
print("Done!")

synth_to_index = {"serum":0, "diva":1, "tyrell":2}

m_size = len(test_spec_data)

#define shapes
l_dim = 64
i_dim = (1, 128, 431, 1)

s_index = args.sample

#get sample to generate
if args.sample == -1:
    s_index = np.random.randint(len(test_serum_params))


#directory for finding checkpoints
if args.model == "multi":
    checkpoint_path = "saved_models/" + "vae_" + args.model + "_" + str(args.latent_size)
else:
    checkpoint_path = "saved_models/" + "vae_" + args.model

print(checkpoint_path)

#get latest model
latest = tf.train.latest_checkpoint(checkpoint_path)

#batch_size
batch_size = 32

#number of batches in one epoch
batches_epoch = m_size // batch_size

#warmup amount
warmup_it = 100*batches_epoch

#create model
if args.model == "multi":
    m = model.vae_multi(64, i_dim, test_serum_params.shape[-1], test_diva_params.shape[-1], test_tyrell_params.shape[-1], model.optimizer, warmup_it)

if args.model == "single":
    m = model.vae_single(64, i_dim, test_diva_params.shape[-1], model.optimizer, warmup_it)

if args.model == "serum":
    m = model.vae_serum(64, i_dim, test_serum_params.shape[-1], test_diva_params.shape[-1], test_tyrell_params.shape[-1], model.optimizer, warmup_it)

if args.model == "diva":
    m = model.vae_diva(64, i_dim, test_serum_params.shape[-1], test_diva_params.shape[-1], test_tyrell_params.shape[-1], model.optimizer, warmup_it)

if args.model == "tyrell":
    m = model.vae_tyrell(64, i_dim, test_serum_params.shape[-1], test_diva_params.shape[-1], test_tyrell_params.shape[-1], model.optimizer, warmup_it)

#load stored weights
m.load_weights(latest)

#compile model
m.compile(optimizer='adam', loss=losses.MeanSquaredError())

name = test_name[s_index]
synth = test_synth[s_index]

print("NAME: " + name)
print("SYNTH: " + synth)
print("NUMBER: " + str(s_index))

if synth == "serum":
    params_t = test_serum_params[s_index]

if synth == "diva":
    params_t = test_diva_params[s_index]

if synth == "tyrell":
    params_t = test_tyrell_params[s_index]

audio_t = generate_audio(params_t, synth)

wavfile.write(name + "_" + synth + "_t.wav", SAMPLING_RATE, audio_t)

if args.model == "multi":
    out = m.predict([test_spec_data[[s_index]], test_serum_masks[[s_index]], test_diva_masks[[s_index]], test_tyrell_masks[[s_index]]])

    params = out[synth_to_index[test_synth[s_index]] + 1][0]
    
    audio_p = generate_audio(params, synth)
    wavfile.write(name + "_" + synth + "_p_multi.wav", SAMPLING_RATE, audio_p)

if args.model == "single":
    out = m.predict([test_spec_data[[s_index]], test_masks[[s_index]]])

    params = out[1]
    params = params[:int(np.sum(test_masks[s_index]))]
    
    audio_p = generate_audio(params, synth)
    wavfile.write(name + "_" + synth + "_p_single.wav", SAMPLING_RATE, audio_p)


