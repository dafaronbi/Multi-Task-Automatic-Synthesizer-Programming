import os
import tensorflow as tf
from tensorflow.keras import losses
import model
import ds
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import spiegelib
import sys
import keras.backend as K
import dawdreamer as dd
from data import one_hot
from data import parameter_label
from scipy.io import wavfile
from data import one_hot
from pesq  import pesq
from sklearn.cluster import KMeans
import scipy.stats
import pandas as pd
from termcolor import colored

#sample rate for generating audio
SAMPLING_RATE = 44100

def multi_resolution_spectral_distance(predict, truth):

    #define fft sizes
    resolutions = [64, 512, 2048]

    #total loss start as 0
    L = 0 

    #alpha parameter (weight of log stft L1 norm)
    alpha = 1

    for resolution in resolutions:

        #calculate ffts
        p_spec = np.abs(librosa.stft(predict, n_fft=resolution))
        t_spec = np.abs(librosa.stft(truth, n_fft=resolution))    

        #l1 norm of difference of STFTs
        diff = np.sum(np.abs(p_spec-t_spec))

        #l1 norm of log difference of STFTs
        log_diff = np.sum(np.abs(np.log(p_spec, where=(p_spec!=0))-np.log(t_spec, where=(t_spec!=0))))

        #add to total
        L += diff + alpha * log_diff

    return L



def log_spectral_distance(labels, logits):
    "" "labels and Logits are one-dimensional data (seq_len,)" ""
    labels_spectrogram = librosa.stft(labels, n_fft=2048)  # (1 + n_fft/2, n_frames)
    logits_spectrogram = librosa.stft(logits, n_fft=2048)  # (1 + n_fft/2, n_frames)

    labels_s = np.abs(labels_spectrogram) ** 2
    logits_s = np.abs(logits_spectrogram) ** 2
 
    labels_log = np.log10(labels_s, where=(labels_s!=0))
    logits_log = np.log10(logits_s, where=(logits_s!=0))

    #Process frequency dimension first
    lsd = np.mean((labels_log - logits_log) ** 2)
 
    return lsd

def predict_decode(params,synth):

    if synth == "serum":
        params = one_hot.predict(params, one_hot.serum_oh)
        params = one_hot.decoded(params, one_hot.serum_oh)
    
    if synth == "diva":
        params = one_hot.predict(params, one_hot.diva_oh)
        params = one_hot.decoded(params, one_hot.diva_oh)
    
    if synth == "tyrell":
        params = one_hot.predict(params, one_hot.tyrell_oh)
        params = one_hot.decoded(params, one_hot.tyrell_oh)

    
    return params

def zero_div(x, y):
    try:
        return x / y
        
    except ZeroDivisionError:
        return 0

def class_acuracy(y_true, y_predict, oh_code, p_label):

    total_classes = 0
    correct_classes = 0
    con_mse = 0
    all_mse = 0
    i = 0
    label_dict = {"osc": [0,0,0], "vol": [0,0,0], "env": [0,0,0], "pitch": [0,0,0], 
    "fil": [0,0,0], "fx": [0,0,0], "mod": [0,0,0], "lfo": [0,0,0]}

    label_amount_dict = {"osc": [0,0,0], "vol": [0,0,0], "env": [0,0,0], "pitch": [0,0,0], 
    "fil": [0,0,0], "fx": [0,0,0], "mod": [0,0,0], "lfo": [0,0,0]}

    for idx,c in enumerate(oh_code):
        

        if c <= 1:
            con_mse += (y_true[i] - y_predict[i])**2
            label_dict[p_label[idx]][0] += con_mse
            label_amount_dict[p_label[idx]][0] += 1

            all_mse += (y_true[i] - y_predict[i])**2
            label_dict[p_label[idx]][2] += all_mse
            label_amount_dict[p_label[idx]][2] += 1

            i += 1
        else:
            total_classes += 1
            #decode one hot
            for n in range(c):
                if y_true[i] == 1:
                    if y_predict[i] == 1:
                        label_dict[p_label[idx]][1] += 1
                    label_amount_dict[p_label[idx]][1] += 1

                all_mse += (y_true[i] - y_predict[i])**2
                label_dict[p_label[idx]][2] += all_mse
                label_amount_dict[p_label[idx]][2] += 1

                i += 1
    
    label_dict = {key: [zero_div(label_dict[key][0], label_amount_dict[key][0]), zero_div(label_dict[key][1], label_amount_dict[key][1]), zero_div(label_dict[key][2], label_amount_dict[key][2])] for key in label_dict.keys()}
    print(label_dict)
    return label_dict

def frobenius_norm(y_true, y_predict):
    return np.abs(np.linalg.norm(y_true.flatten()) - np.linalg.norm(y_predict.flatten()))



def exp_0():
    pass

def generate_audio(params, synth):
    plugin_path = ""

    if synth == "serum":
        #path to plugin
        plugin_path = "data generation/Serum.vst"
        
    if synth == "diva":
        #path to plugin
        plugin_path = "data generation/Diva.vst"

    if synth == "tyrell":
        #path to plugin
        plugin_path = "data generation/TyrellN6.vst"

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

def multi_metrics():

    print("Loading Data...")
    test_names = np.load("test_name.npy")
    test_spec_data = np.load("test_mels.npy",allow_pickle=True)
    test_serum_params = np.load("test_serum_params.npy",allow_pickle=True)
    test_serum_masks = np.load("test_serum_mask.npy",allow_pickle=True)
    test_diva_params = np.load("test_diva_params.npy",allow_pickle=True)
    test_diva_masks = np.load("test_diva_mask.npy",allow_pickle=True)
    test_tyrell_params = np.load("test_tyrell_params.npy",allow_pickle=True)
    test_tyrell_masks = np.load("test_tyrell_mask.npy",allow_pickle=True)
    test_h_labels = np.load("test_hpss.npy",allow_pickle=True)
    test_synth = np.load("test_synth.npy",allow_pickle=True)
    print("Done!")

    m_size = len(test_spec_data)

    #define shapes
    l_dim = 64
    i_dim = (1, 128, 431, 1)

    #directory for finding checkpoints
    checkpoint_path = "saved_models/vst_multi_64/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    #get latest model
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    #batch_size
    batch_size = 32

    #number of batches in one epoch
    batches_epoch = m_size // batch_size

    #warmup amounta
    warmup_it = 100*batches_epoch

    #create model
    m = model.vae_multi(64, i_dim, test_serum_params.shape[-1], test_diva_params.shape[-1], test_tyrell_params.shape[-1], model.optimizer, warmup_it)

    #load stored weights
    m.load_weights(latest)

    file_name = "group_metrics.csv"

    metrics_df = pd.read_csv(file_name)
    i = len(metrics_df)

    while i < len(test_serum_params):
        text = colored(test_names[i], 'red', attrs=['reverse', 'blink'])
        text2 = colored(str(i), 'red', attrs=['reverse', 'blink'])
        print(text)
        print(text2)


        metrics_df = pd.read_csv(file_name)
        i = len(metrics_df)
        accepted = False

        audio_t = []
        audio_p = []

        spec,serum,diva,tyrell = m.predict([test_spec_data[[i]], test_serum_masks[[i]], test_diva_masks[[i]], test_tyrell_masks[[i]]])

        mrsd =  0
        lsd = 0
        fn = 0
        param_r = 0
        param_c = 0
        synth = 0


        if test_synth[i] == "serum":
            print("serum")

            synth = 0

            serum_t = one_hot.predict(np.squeeze(test_serum_params[i]),one_hot.serum_oh)
            serum_p = one_hot.predict(np.squeeze(serum),one_hot.serum_oh)

            acc_dict = class_acuracy(serum_t, serum_p, one_hot.serum_oh, parameter_label.serum_labels)
                

        if test_synth[i] == "diva":
            print("diva")
            synth = 1

            diva_t = one_hot.predict(np.squeeze(test_diva_params[i]),one_hot.diva_oh)
            diva_p = one_hot.predict(np.squeeze(diva),one_hot.diva_oh)

            acc_dict = class_acuracy(diva_t, diva_p, one_hot.diva_oh, parameter_label.diva_labels)            


        if test_synth[i] == "tyrell":
            print("tyrell")
            synth = 2

            tyrell_t = one_hot.predict(np.squeeze(test_tyrell_params[i]),one_hot.tyrell_oh)
            tyrell_p = one_hot.predict(np.squeeze(tyrell),one_hot.tyrell_oh)

            acc_dict = class_acuracy(tyrell_t, tyrell_p, one_hot.tyrell_oh, parameter_label.tyrell_labels)  

            
        
        print("saving: " + test_names[i])
        metrics_df = metrics_df.append({'name':test_names[i], "osc_con": acc_dict["osc"][0], "osc_class": acc_dict["osc"][1], "osc_all": acc_dict["osc"][2], "vol_con": acc_dict["vol"][0], 
        "vol_class": acc_dict["vol"][1], "vol_all": acc_dict["vol"][2], "env_con": acc_dict["env"][0], "env_class": acc_dict["env"][1], "env_all": acc_dict["env"][2],
        "pitch_con": acc_dict["pitch"][0], "pitch_class": acc_dict["pitch"][1], "pitch_all": acc_dict["pitch"][2], "fil_con": acc_dict["fil"][0], "fil_class": acc_dict["fil"][1], "fil_all": acc_dict["fil"][2],
        "fx_con": acc_dict["fx"][0], "fx_class": acc_dict["fx"][1], "fx_all": acc_dict["fx"][2], "mod_con": acc_dict["mod"][0], "mod_class": acc_dict["mod"][1], "mod_all": acc_dict["mod"][2], 
        "lfo_con": acc_dict["lfo"][0], "lfo_class": acc_dict["lfo"][1], "lfo_all": acc_dict["lfo"][2], 'hpss': test_h_labels[i],'synth': test_synth[i]},ignore_index=True)
        metrics_df.to_csv(file_name,index=False)



if __name__ == "__main__":
    multi_metrics()
