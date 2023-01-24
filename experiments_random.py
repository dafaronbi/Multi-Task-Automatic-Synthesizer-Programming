#import need models
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
from scipy.io import wavfile
from data import one_hot
from pesq  import pesq
from sklearn.cluster import KMeans
from pesq import pesq
# import pickle
import pickle5 as pickle
import scipy.stats

#sample rate for geneating audio
SAMPLING_RATE = 44100

def class_acuracy(y_true,y_predict,oh_code):

    total_classes = 0
    correct_classes = 0
    con_mse = 0
    i = 0
    for c in oh_code:
        if c <= 1:
            con_mse += (y_true[i] - y_predict[i])**2
            i += 1
        else:
            total_classes += 1
            #decode one hot
            for n in range(c):
                if y_true[i] == 1:
                    if y_predict[i] == 1:
                        correct_classes += 1
                i += 1

    return con_mse, (correct_classes / total_classes)

def log_spectral_distance(y_true, y_predict):
    return np.mean(20*(np.nan_to_num(np.log10(y_true) - np.log10(y_predict))))

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


def latent_sampling(num,task):

    #load data
    print("Loading Data...")
    #load in data
    test_spec_data = np.load("test_spec.npy",allow_pickle=True)
    test_serum_params = np.load("test_serum_params.npy",allow_pickle=True)
    test_serum_masks = np.load("test_serum_masks.npy",allow_pickle=True)
    test_diva_params = np.load("test_diva_params.npy",allow_pickle=True)
    test_diva_masks = np.load("test_diva_masks.npy",allow_pickle=True)
    test_tyrell_params = np.load("test_tyrell_params.npy",allow_pickle=True)
    test_tyrell_masks = np.load("test_tyrell_masks.npy",allow_pickle=True)
    print("Done!")

    m_size = len(test_spec_data)

    #define shapes
    l_dim = 64
    i_dim = (1, 128, 431, 1)

    #directory for finding checkpoints
    checkpoint_path = "saved_models/vst_multi/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    #get latest model
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    #batch_size
    batch_size = 32

    #number of batches in one epoch
    batches_epoch = m_size // batch_size

    #warmup amount
    warmup_it = 100*batches_epoch

    #create model
    m = model.vae_multi(64, i_dim, test_serum_params.shape[-1], test_diva_params.shape[-1], test_tyrell_params.shape[-1], model.optimizer, warmup_it)

    #load stored weights
    m.load_weights(latest)

    #compile model
    m.compile(optimizer='adam', loss=losses.MeanSquaredError())

    # m.summary()
    l_model = tf.keras.Model(inputs=m.input, outputs=m.get_layer("z_mean").output)
    spec_model = tf.keras.Model( inputs=[m.get_layer("sampling_2").input, m.get_layer("input_8").input, 
    m.get_layer("input_9").input,m.get_layer("input_10").input], 
                outputs=[m.get_layer("spectrogram").output,
                m.get_layer("synth_params_serum").output,
                m.get_layer("synth_params_diva").output,
                m.get_layer("synth_params_tyrell").output,
                ])
    
    if task == 'spec':

        for z_index in [num]:

            centroids = []
            bandwidth = []
            contrast = []
            flatness = []
            rms = []

            zeros = np.zeros((1,64))

            z_values = np.arange(-50,50,5)

            confuse_matrix = np.zeros((4,4))
            for z_num in z_values:
                zeros[0][z_index] = z_num
                spec,serum,diva,tyrell = spec_model.predict([zeros,np.zeros((1,64)), np.ones((1,480)),np.ones((1,759)), np.ones((1,327))])

                # spec = np.array(np.squeeze(spec))
                
                # s_spec = generate_spectrogram(np.squeeze(serum), "serum")
                # d_spec = generate_spectrogram(np.squeeze(diva), "diva")
                # t_spec = generate_spectrogram(np.squeeze(tyrell), "tyrell")

                # specs = np.array([spec, s_spec, d_spec, t_spec])
                
                # for i in range(len(specs)):
                #     for j in range(len(specs)):
                #         confuse_matrix[i][j] += np.linalg.norm(specs[i].flatten() -specs[j].flatten())

                serum = one_hot.predict(np.squeeze(serum),one_hot.serum_oh)
                serum = one_hot.decode(serum,one_hot.serum_oh)

                diva = one_hot.predict(np.squeeze(diva),one_hot.diva_oh)
                diva = one_hot.decode(diva,one_hot.diva_oh)

                tyrell = one_hot.predict(np.squeeze(tyrell),one_hot.tyrell_oh)
                tyrell = one_hot.decode(tyrell,one_hot.tyrell_oh)
                
                s_audio = generate_audio(serum, "serum")
                d_audio = generate_audio(diva, "diva")
                t_audio = generate_audio(tyrell, "tyrell")
                
                y = np.array([s_audio,d_audio,t_audio])

                centroids.append(np.mean(np.squeeze(librosa.feature.spectral_centroid(y=y,sr=SAMPLING_RATE)),axis =1))
                bandwidth.append(np.mean(np.squeeze(librosa.feature.spectral_bandwidth(y=y,sr=SAMPLING_RATE)),axis =1))
                contrast.append(np.mean(np.squeeze(librosa.feature.spectral_contrast(y=y,sr=SAMPLING_RATE)),axis=1))
                flatness.append(np.mean(np.squeeze(librosa.feature.spectral_flatness(y=y)),axis=1))
                rms.append(np.mean(np.squeeze(librosa.feature.rms(y=y)),axis=1))

                # plt.close()
                # fig, a = plt.subplots(4)

                # a[0].imshow(spec)
                # a[0].set_title('Spectrogram Generation')
                # a[1].imshow(s_spec)
                # a[1].set_title('Serum Audio Spectrogram')
                # a[2].imshow(d_spec)
                # a[2].set_title('Diva Audio Spectrogram')
                # a[3].imshow(t_spec)
                # a[3].set_title('Tyrell Audio Spectrogram')

                # librosa.display.specshow(spec, x_axis='time', y_axis='mel', ax=a[0])
                # librosa.display.specshow(s_spec, x_axis='time', y_axis='mel', ax=a[1])
                # librosa.display.specshow(d_spec, x_axis='time', y_axis='mel', ax=a[2])
                # librosa.display.specshow(t_spec, x_axis='time', y_axis='mel', ax=a[3])

                # plt.show()

            # plt.close()
            centroids = np.array(centroids)
            bandwidth = np.array(bandwidth)
            contrast = np.mean((contrast),axis=2)
            flatness = np.array(flatness)
            rms = np.array(rms)

            print(contrast)
            print(contrast.shape)

            fig, a = plt.subplots(5)

            a[0].plot(z_values,centroids[:,0],label='serum')
            a[0].plot(z_values,centroids[:,1],label='diva')
            a[0].plot(z_values,centroids[:,2],label='tyrell')
            a[0].set_title('mean spectral centroid')

            a[1].plot(z_values,bandwidth[:,0])
            a[1].plot(z_values,bandwidth[:,1])
            a[1].plot(z_values,bandwidth[:,2])
            a[1].set_title('mean spectral bandwidth')

            a[2].plot(z_values,contrast[:,0])
            a[2].plot(z_values,contrast[:,1])
            a[2].plot(z_values,contrast[:,2])
            a[2].set_title('mean spectral contrast')

            a[3].plot(z_values,flatness[:,0])
            a[3].plot(z_values,flatness[:,1])
            a[3].plot(z_values,flatness[:,2])
            a[3].set_title('mean spectral flatness')


            a[4].plot(z_values,rms[:,0])
            a[4].plot(z_values,rms[:,1])
            a[4].plot(z_values,rms[:,2])
            a[4].set_title('mean spectral rms')


            fig.legend(loc='lower right',ncol=3)

            
            # plt.imshow(confuse_matrix)
            # plt.show()
            plt.savefig("Experiment Results/Latent Sampling/features_over_time_medium" + str(z_index) + ".png")

    if task == "confuse":
        z_values = np.arange(-100,100,10)

        for z_index in [num]:
            zeros = np.zeros((1,64))

            confuse_matrix = np.zeros((4,4))
            for z_num in z_values:
                zeros[0][z_index] = z_num
                spec,serum,diva,tyrell = spec_model.predict([zeros,np.zeros((1,64)), np.ones((1,480)),np.ones((1,759)), np.ones((1,327))])

                spec = np.array(np.squeeze(spec))
                
                s_spec = generate_spectrogram(np.squeeze(serum), "serum")
                d_spec = generate_spectrogram(np.squeeze(diva), "diva")
                t_spec = generate_spectrogram(np.squeeze(tyrell), "tyrell")

                specs = np.array([spec, s_spec, d_spec, t_spec])
                
                for i in range(len(specs)):
                    for j in range(len(specs)):
                        confuse_matrix[i][j] += np.abs(np.linalg.norm(specs[i].flatten()) - np.linalg.norm(specs[j].flatten()))

            labels = ["Spectrogram","Serum","Diva","Tyrell"]
            plt.matshow(confuse_matrix, cmap="Blues")
            plt.xticks([0,1,2,3],labels)
            plt.yticks([0,1,2,3],labels)
            plt.colorbar(label="Frobenius Norm Value",)
            plt.title("Mel Spectrogram Similarities (Lower is more similar)")

            for i in range(len(specs)):
                    for j in range(len(specs)):
                        plt.text(i, j, str(round(confuse_matrix[i][j], 2)), va='center', ha='center')

            plt.savefig("Experiment Results/Latent Confusion/Confusion z_" + str(z_index) + ".png")

    if task == 'param':
        #create renderman engine with plugin loaded
        engine = dd.RenderEngine(SAMPLING_RATE, 512)
        engine.set_bpm(120)
        synth = engine.make_plugin_processor("synth", "data generation/Serum.vst")
        s_param_desc = synth.get_plugin_parameters_description()
        synth = engine.make_plugin_processor("synth", "data generation/Diva.vst")
        d_param_desc = synth.get_plugin_parameters_description()
        synth = engine.make_plugin_processor("synth", "data generation/TyrellN6.vst")
        t_param_desc = synth.get_plugin_parameters_description()


        for z_index in range(63,64):
            z_values = np.arange(-100,100,2)
            zeros = np.zeros((len(z_values),64))

            zeros[:,z_index] = z_values
            _,serum,diva,tyrell = spec_model.predict([zeros,np.zeros((len(z_values),64)), np.ones((len(z_values),480)),np.ones((len(z_values),759)), np.ones((len(z_values),327))])

            serums = []
            divas = []
            tyrells = []
            for i in range(len(z_values)):

                serum_p = one_hot.decode(one_hot.predict(serum[i], one_hot.serum_oh), one_hot.serum_oh)
                diva_p = one_hot.decode(one_hot.predict(diva[i], one_hot.diva_oh), one_hot.diva_oh)
                tyrell_p = one_hot.decode(one_hot.predict(tyrell[i], one_hot.tyrell_oh), one_hot.tyrell_oh)

                serums.append(serum_p)
                divas.append(diva_p)
                tyrells.append(tyrell_p)


            serums = np.array(serums)
            divas = np.array(divas)
            tyrells = np.array(tyrells)

            for i,param in enumerate(s_param_desc):
                p_name = param['name'].replace("/", "--")
                plt.scatter(z_values, serums[:,i])
                plt.xlabel("z" + str(z_index)+ " Value")
                plt.ylabel(param['name'])
                plt.title("z" + str(z_index)+ " Value " + p_name)
                plt.savefig("Experiment Results/Latent Parameters/Serum z_" + str(z_index) + "_" + p_name + ".png")
                plt.close()

            for i,param in enumerate(d_param_desc):
                p_name = param['name'].replace("/", "--")
                plt.scatter(z_values, divas[:,i])
                plt.xlabel("z" + str(z_index)+ " Value")
                plt.ylabel(param['name'])
                plt.title("z" + str(z_index)+ " Value " + p_name)
                plt.savefig("Experiment Results/Latent Parameters/Diva z_" + str(z_index) + "_" + p_name + ".png")
                plt.close()

            for i,param in enumerate(t_param_desc):
                p_name = param['name'].replace("/", "--")
                plt.scatter(z_values, tyrells[:,i])
                plt.xlabel("z" + str(z_index)+ " Value")
                plt.ylabel(param['name'])
                plt.title("z" + str(z_index)+ " Value " + p_name)
                plt.savefig("Experiment Results/Latent Parameters/Tyrell z_" + str(z_index) + "_" + p_name + ".png")
                plt.close()

            


def evaluate_latent(task):
    #load data
    print("Loading Data...")
    #load in data
    test_spec_data = np.load("test_spec.npy",allow_pickle=True)
    test_serum_params = np.load("test_serum_params.npy",allow_pickle=True)
    test_serum_masks = np.load("test_serum_masks.npy",allow_pickle=True)
    test_diva_params = np.load("test_diva_params.npy",allow_pickle=True)
    test_diva_masks = np.load("test_diva_masks.npy",allow_pickle=True)
    test_tyrell_params = np.load("test_tyrell_params.npy",allow_pickle=True)
    test_tyrell_masks = np.load("test_tyrell_masks.npy",allow_pickle=True)
    test_h_labels = np.load("test_h_labels.npy",allow_pickle=True)
    print("Done!")

    m_size = len(test_spec_data)

    #define shapes
    l_dim = 64
    i_dim = (1, 128, 431, 1)

    #directory for finding checkpoints
    checkpoint_path = "saved_models/vst_multi/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    #get latest model
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    #batch_size
    batch_size = 32

    #number of batches in one epoch
    batches_epoch = m_size // batch_size

    #warmup amount
    warmup_it = 100*batches_epoch

    #create model
    m = model.vae_multi(64, i_dim, test_serum_params.shape[-1], test_diva_params.shape[-1], test_tyrell_params.shape[-1], model.optimizer, warmup_it)

    #load stored weights
    m.load_weights(latest)

    if task == "all":

        #compile model
        m.compile(optimizer='adam', loss=losses.MeanSquaredError())

        l_model = tf.keras.Model(inputs=m.input, outputs=[m.get_layer("z_mean").output,m.get_layer("z_log_var").output,m.get_layer("sampling_2").output])

        z_m, z_v,z_s = l_model.predict([test_spec_data, test_serum_masks, test_diva_masks, test_tyrell_masks])

        for z_num in range(64):

            plt.hist(z_m[:,z_num],bins=64,alpha = 0.3)
            plt.savefig("Experiment Results/Latent Histograms/z_mean_" + str(z_num) + "hist.png")
            plt.close()
            plt.hist(z_v[:,z_num],bins=64,alpha = 0.3)
            plt.savefig("Experiment Results/Latent Histograms/z_logvar_" + str(z_num) + "hist.png")
            plt.close()
            plt.hist(z_s[:,z_num],bins=64,alpha = 0.3)
            plt.savefig("Experiment Results/Latent Histograms/z_sampling_" + str(z_num) + "hist.png")
            plt.close()

        # plt.imshow(np.expand_dims(z/size, axis=0))
        # plt.show()
    
    if task == "per_synth":

        #compile model
        m.compile(optimizer='adam', loss=losses.MeanSquaredError())

        l_model = tf.keras.Model(inputs=m.input, outputs=[m.get_layer("z_mean").output,m.get_layer("z_log_var").output,m.get_layer("sampling_2").output])

        serum_z = []
        diva_z = []
        tyrell_z = []

        

        for i in range(len(test_spec_data)):

            if test_serum_masks[i][0] == 1:
                print("serum")
                serum_z.append(l_model.predict([test_spec_data[[i]], test_serum_masks[[i]], test_diva_masks[[i]], test_tyrell_masks[[i]]]))
            
            if test_diva_masks[i][0] == 1:
                print("diva")
                diva_z.append(l_model.predict([test_spec_data[[i]], test_serum_masks[[i]], test_diva_masks[[i]], test_tyrell_masks[[i]]]))

            if test_tyrell_masks[i][0] == 1:
                print("tyrell")
                tyrell_z.append(l_model.predict([test_spec_data[[i]], test_serum_masks[[i]], test_diva_masks[[i]], test_tyrell_masks[[i]]]))
            # print(len(serum_z))
            # print(np.squeeze(serum_z))
            # print(len(diva_z))
            # print(len(tyrell_z))

        for z_num in range(64):
            serum_z = np.squeeze(np.array(serum_z))
            diva_z = np.squeeze(np.array(diva_z))
            tyrell_z = np.squeeze(np.array(tyrell_z))

            print(serum_z.shape)
            s_zm = serum_z[:,0]
            # print(s_zm)
            print(s_zm.shape)
            s_zv = serum_z[:,1]
            s_zs = serum_z[:,2]
            d_zm = diva_z[:,0]
            d_zv = diva_z[:,1]
            d_zs = diva_z[:,2]
            t_zm = tyrell_z[:,0]
            t_zv = tyrell_z[:,1]
            t_zs = tyrell_z[:,2]

            plt.hist(s_zm[:,z_num],bins=64,label="Serum",alpha = 0.3)
            plt.hist(d_zm[:,z_num],bins=64,label="Diva",alpha = 0.3)
            plt.hist(t_zm[:,z_num],bins=64,label="Tyrell",alpha = 0.3)
            plt.savefig("Experiment Results/Latent Histograms/synthesizer_mean_" + str(z_num) + "hist.png")
            plt.legend()
            plt.close()
            plt.hist(s_zv[:,z_num],bins=64,label="Serum",alpha = 0.3)
            plt.hist(d_zv[:,z_num],bins=64,label="Diva",alpha = 0.3)
            plt.hist(t_zv[:,z_num],bins=64,label="Tyrell",alpha = 0.3)
            plt.legend()
            plt.savefig("Experiment Results/Latent Histograms/synthesizer_logvar_" + str(z_num) + "hist.png")
            plt.close()
            plt.hist(s_zs[:,z_num],bins=64,label="Serum",alpha = 0.3)
            plt.hist(d_zs[:,z_num],bins=64,label="Diva",alpha = 0.3)
            plt.hist(t_zs[:,z_num],bins=64,label="Tyrell",alpha = 0.3)
            plt.legend()
            plt.savefig("Experiment Results/Latent Histograms/synthesizer_sampling_" + str(z_num) + "hist.png")
            plt.close()

    if task == "per_hpss":

        #compile model
        m.compile(optimizer='adam', loss=losses.MeanSquaredError())

        l_model = tf.keras.Model(inputs=m.input, outputs=[m.get_layer("z_mean").output,m.get_layer("z_log_var").output,m.get_layer("sampling_2").output])

        h_20_z = []
        h_40_z = []
        h_60_z = []
        h_80_z = []
        h_100_z = []

        

        for i in range(len(test_serum_masks)):

            if test_h_labels[i] == 20:
                print("HPSS 20")
                h_20_z.append(l_model.predict([test_spec_data[[i]], test_serum_masks[[i]], test_diva_masks[[i]], test_tyrell_masks[[i]]]))
            
            if test_h_labels[i] == 40:
                print("HPSS 40")
                h_40_z.append(l_model.predict([test_spec_data[[i]], test_serum_masks[[i]], test_diva_masks[[i]], test_tyrell_masks[[i]]]))

            if test_h_labels[i] == 60:
                print("HPSS 60")
                h_60_z.append(l_model.predict([test_spec_data[[i]], test_serum_masks[[i]], test_diva_masks[[i]], test_tyrell_masks[[i]]]))

            if test_h_labels[i] == 80:
                print("HPSS 80")
                h_80_z.append(l_model.predict([test_spec_data[[i]], test_serum_masks[[i]], test_diva_masks[[i]], test_tyrell_masks[[i]]]))

            if test_h_labels[i] == 100:
                print("HPSS 100")
                h_100_z.append(l_model.predict([test_spec_data[[i]], test_serum_masks[[i]], test_diva_masks[[i]], test_tyrell_masks[[i]]]))

        h_20_z = np.squeeze(h_20_z)[:,2]
        h_40_z = np.squeeze(h_40_z)[:,2]
        h_60_z = np.squeeze(h_60_z)[:,2]
        h_80_z = np.squeeze(h_80_z)[:,2]
        h_100_z = np.squeeze(h_100_z)[:,2]   

        for z_num in range(64):

            plt.hist(h_20_z[:,z_num],bins=64,label="H% < 20",alpha = 0.3)
            plt.hist(h_40_z[:,z_num],bins=64,label="H% < 40",alpha = 0.3)
            plt.hist(h_60_z[:,z_num],bins=64,label="H% < 60",alpha = 0.3)
            plt.hist(h_80_z[:,z_num],bins=64,label="H% < 80",alpha = 0.3)
            plt.hist(h_100_z[:,z_num],bins=64,label="H% < 100",alpha = 0.3)
            plt.legend()
            plt.savefig("Experiment Results/Latent Histograms/HPSS_sampling_" + str(z_num) + "hist.png")
            plt.close()

def random_sample():
    #load data
    print("Loading Data...")
    test_spec_data = np.load("test_spec.npy",allow_pickle=True)
    test_serum_params = np.load("test_serum_params.npy",allow_pickle=True)
    test_serum_masks = np.load("test_serum_masks.npy",allow_pickle=True)
    test_diva_params = np.load("test_diva_params.npy",allow_pickle=True)
    test_diva_masks = np.load("test_diva_masks.npy",allow_pickle=True)
    test_tyrell_params = np.load("test_tyrell_params.npy",allow_pickle=True)
    test_tyrell_masks = np.load("test_tyrell_masks.npy",allow_pickle=True)
    print("Done!")

    m_size = len(test_spec_data)

    #define shapes
    l_dim = 64
    i_dim = (1, 128, 431, 1)

    #directory for finding checkpoints
    checkpoint_path = "saved_models/vst_multi/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    #get latest model
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    #batch_size
    batch_size = 32

    #number of batches in one epoch
    batches_epoch = m_size // batch_size

    #warmup amount
    warmup_it = 100*batches_epoch

    #create model
    m = model.vae_multi(64, i_dim, test_serum_params.shape[-1], test_diva_params.shape[-1], test_tyrell_params.shape[-1], model.optimizer, warmup_it)

    #load stored weights
    m.load_weights(latest)

    #get random sample
    samp = np.random.randint(0,len(test_spec_data))

    spec,serum,diva,tyrell = m.predict([test_spec_data[[samp]], test_serum_masks[[samp]], test_diva_masks[[samp]], test_tyrell_masks[[samp]]])

    audio_t = []
    audio_p = []

    if test_serum_masks[samp][0] == 1:

        serum_t = one_hot.predict(np.squeeze(test_serum_params[samp]),one_hot.serum_oh)
        serum_t = one_hot.decode(serum_t,one_hot.serum_oh)

        serum_p = one_hot.predict(np.squeeze(serum),one_hot.serum_oh)
        serum_p = one_hot.decode(serum_p,one_hot.serum_oh)

        audio_t = generate_audio(serum_t, "serum")
        audio_p = generate_audio(serum_p, "serum")  

    if test_diva_masks[samp][0] == 1:
        diva_t = one_hot.predict(np.squeeze(test_diva_params[samp]),one_hot.diva_oh)
        diva_t = one_hot.decode(diva_t,one_hot.diva_oh)

        diva_p = one_hot.predict(np.squeeze(diva),one_hot.diva_oh)
        diva_p = one_hot.decode(diva_p,one_hot.diva_oh)

        audio_t = generate_audio(diva_t, "diva")
        audio_P = generate_audio(diva_p, "diva")

    if test_tyrell_masks[samp][0] == 1:
        tyrell_t = one_hot.predict(np.squeeze(test_tyrell_params[samp]),one_hot.tyrell_oh)
        tyrell_t = one_hot.decode(tyrell_t,one_hot.tyrell_oh)

        tyrell_p = one_hot.predict(np.squeeze(tyrell),one_hot.tyrell_oh)
        tyrell_p = one_hot.decode(tyrell_p,one_hot.tyrell_oh)

        audio_t = generate_audio(tyrell_t, "tyrell")
        audio_p = generate_audio(tyrell_p, "tyrell")

    wavfile.write("E_truth.wav",SAMPLING_RATE, audio_t)
    wavfile.write("E_predict.wav",SAMPLING_RATE, audio_p)

def out_of_domain_sample():
    #load data
    print("Loading Data...")
    test_spec_data = np.load("test_spec.npy",allow_pickle=True)
    test_serum_params = np.load("test_serum_params.npy",allow_pickle=True)
    test_serum_masks = np.load("test_serum_masks.npy",allow_pickle=True)
    test_diva_params = np.load("test_diva_params.npy",allow_pickle=True)
    test_diva_masks = np.load("test_diva_masks.npy",allow_pickle=True)
    test_tyrell_params = np.load("test_tyrell_params.npy",allow_pickle=True)
    test_tyrell_masks = np.load("test_tyrell_masks.npy",allow_pickle=True)
    print("Done!")

    m_size = len(test_spec_data)

    #define shapes
    l_dim = 64
    i_dim = (1, 128, 431, 1)

    #directory for finding checkpoints
    checkpoint_path = "saved_models/vst_multi/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    #get latest model
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    #batch_size
    batch_size = 32

    #number of batches in one epoch
    batches_epoch = m_size // batch_size

    #warmup amount
    warmup_it = 100*batches_epoch

    #create model
    m = model.vae_multi(64, i_dim, test_serum_params.shape[-1], test_diva_params.shape[-1], test_tyrell_params.shape[-1], model.optimizer, warmup_it)

    #load stored weights
    m.load_weights(latest)

    #get random sample
    samp = np.random.randint(0,len(test_spec_data))

    spec,serum,diva,tyrell = m.predict([test_spec_data[[samp]], np.ones((1,480)),np.ones((1,759)), np.ones((1,327))])

    audio_t = []
    audio_s = []
    audio_d = []
    audio_ty = []

    if test_serum_masks[samp][0] == 1:

        serum_t = one_hot.predict(np.squeeze(test_serum_params[samp]),one_hot.serum_oh)
        serum_t = one_hot.decode(serum_t,one_hot.serum_oh)

        audio_t = generate_audio(serum_t, "serum")

    if test_diva_masks[samp][0] == 1:
        diva_t = one_hot.predict(np.squeeze(test_diva_params[samp]),one_hot.diva_oh)
        diva_t = one_hot.decode(diva_t,one_hot.diva_oh)

        audio_t = generate_audio(diva_t, "diva")

    if test_tyrell_masks[samp][0] == 1:
        tyrell_t = one_hot.predict(np.squeeze(test_tyrell_params[samp]),one_hot.tyrell_oh)
        tyrell_t = one_hot.decode(tyrell_t,one_hot.tyrell_oh)

        audio_t = generate_audio(tyrell_t, "tyrell")

    serum = one_hot.predict(np.squeeze(serum),one_hot.serum_oh)
    serum = one_hot.decode(serum,one_hot.serum_oh)

    audio_s = generate_audio(serum, "serum")

    diva = one_hot.predict(np.squeeze(diva),one_hot.diva_oh)
    diva = one_hot.decode(diva,one_hot.diva_oh)

    audio_d = generate_audio(diva, "diva")

    tyrell = one_hot.predict(np.squeeze(tyrell),one_hot.tyrell_oh)
    tyrell = one_hot.decode(tyrell,one_hot.tyrell_oh)

    audio_ty = generate_audio(tyrell, "tyrell")

    wavfile.write("Out_E_truth.wav",SAMPLING_RATE, audio_t)
    wavfile.write("Out_E_serum.wav",SAMPLING_RATE, audio_s)
    wavfile.write("Out_E_diva.wav",SAMPLING_RATE, audio_d)
    wavfile.write("Out_E_tyrell.wav",SAMPLING_RATE, audio_ty)

def model_accuracy():
    #load data
    print("Loading Data...")
    with open("dynamic_rand/test_random_dynamic_mels.pkl", 'rb') as handle:
        test_mels_batch = pickle.load(handle)
    with open("dynamic_rand/test_random_dynamic_params.pkl", 'rb') as handle:
        test_params_batch = pickle.load(handle)
    with open("dynamic_rand/test_random_dynamic_kernels.pkl", 'rb') as handle:
        test_kernels_batch = pickle.load(handle)
    
    
    test_mels = []
    test_params = []
    test_kernels = []

    #get rid of batch dimmensinos
    for mel in test_mels_batch:
        for mel_b in mel:
            test_mels.append(mel_b)

    for params in test_params_batch:
        for params_b in params:
            test_params.append(params_b)

    for kernels in test_kernels_batch:
        for kernels_b in kernels:
            test_kernels.append(kernels_b)

    print("Done!")


    m_size = len(test_mels)

    #define shapes
    l_dim = 64
    i_dim = (1, 128, 431, 1)

    #directory for finding checkpoints
    checkpoint_path = "saved_models/random_dynamic/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    #get latest model
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    #batch_size
    batch_size = 32

    #number of batches in one epoch
    batches_epoch = m_size // batch_size

    #warmup amount
    warmup_it = 100*batches_epoch

    #create model
    m = model.dynamic_vae(64, i_dim, 5, model.optimizer, warmup_it,10)

    #load stored weights
    m.load_weights(latest)

    metrics_arr = np.load("metrics_random.npy",allow_pickle=True)
    i = metrics_arr.shape[0]

    while i < len(test_mels):

        metrics_arr = np.load("metrics_random.npy",allow_pickle=True)
        i = len(metrics_arr)
        accepted = True

        audio_t = []
        audio_p = []

        spec,params = m.predict([np.array([test_mels[i]]), np.array([np.swapaxes(test_kernels[i],0,1)])])


        lsd =  0
        fn = 0
        param_r = 0
        param_c = 0
        synth = 0


        if len(params[0]) == 480:

            synth = 0

            serum_t = one_hot.predict(np.squeeze(test_params[i]),one_hot.serum_oh)
            p_t = serum_t
            serum_t = one_hot.decoded(serum_t,one_hot.serum_oh)

            serum_p = one_hot.predict(np.squeeze(params),one_hot.serum_oh)
            p_p = serum_p
            serum_p = one_hot.decoded(serum_p,one_hot.serum_oh) 

            param_r,param_c = class_acuracy(p_t, p_p, one_hot.serum_oh)

            try:
                audio_t = generate_audio(serum_t, "serum")
                audio_p = generate_audio(serum_p, "serum") 

                mel_spec_t = librosa.feature.melspectrogram(y=audio_t, sr=SAMPLING_RATE,)
                mel_spec_t = librosa.power_to_db(mel_spec_t,ref=np.max)

                mel_spec_p = librosa.feature.melspectrogram(y=audio_p, sr=SAMPLING_RATE,)
                mel_spec_p = librosa.power_to_db(mel_spec_p,ref=np.max)

                lsd = log_spectral_distance(mel_spec_t**2, mel_spec_p**2)
                fn = frobenius_norm(mel_spec_t, mel_spec_p)

            except:
                accepted = False


        if len(params[0]) == 759:

            synth = 1

            diva_t = one_hot.predict(np.squeeze(test_params[i]),one_hot.diva_oh)
            p_t = diva_t
            diva_t = one_hot.decoded(diva_t,one_hot.diva_oh)

            diva_p = one_hot.predict(np.squeeze(params),one_hot.diva_oh)
            p_p = diva_p
            diva_p = one_hot.decoded(diva_p,one_hot.diva_oh)

            param_r,param_c = class_acuracy(p_t, p_p, one_hot.diva_oh)

            try:
                audio_t = generate_audio(diva_t, "diva")
                audio_p = generate_audio(diva_p, "diva") 

                mel_spec_t = librosa.feature.melspectrogram(y=audio_t, sr=SAMPLING_RATE,)
                mel_spec_t = librosa.power_to_db(mel_spec_t,ref=np.max)

                mel_spec_p = librosa.feature.melspectrogram(y=audio_p, sr=SAMPLING_RATE,)
                mel_spec_p = librosa.power_to_db(mel_spec_p,ref=np.max)

                lsd = log_spectral_distance(mel_spec_t**2, mel_spec_p**2)
                fn = frobenius_norm(mel_spec_t, mel_spec_p)

            except:
                accepted = False


        if len(params[0]) == 327:

            synth = 2

            tyrell_t = one_hot.predict(np.squeeze(test_params[i]),one_hot.tyrell_oh)
            p_t = tyrell_t
            tyrell_t = one_hot.decoded(tyrell_t,one_hot.tyrell_oh)

            tyrell_p = one_hot.predict(np.squeeze(params),one_hot.tyrell_oh)
            p_p = tyrell_p
            tyrell_p = one_hot.decoded(tyrell_p,one_hot.tyrell_oh)
            

            param_r,param_c = class_acuracy(p_t, p_p, one_hot.tyrell_oh)

            try:
                audio_t = generate_audio(tyrell_t, "tyrell")
                audio_p = generate_audio(tyrell_p, "tyrell") 

                mel_spec_t = librosa.feature.melspectrogram(y=audio_t, sr=SAMPLING_RATE,)
                mel_spec_t = librosa.power_to_db(mel_spec_t,ref=np.max)

                mel_spec_p = librosa.feature.melspectrogram(y=audio_p, sr=SAMPLING_RATE,)
                mel_spec_p = librosa.power_to_db(mel_spec_p,ref=np.max)

                lsd = log_spectral_distance(mel_spec_t**2, mel_spec_p**2)
                fn = frobenius_norm(mel_spec_t, mel_spec_p)

            except:
                accepted = False
            
        if accepted:
            metrics_arr = np.append(metrics_arr,[[lsd,fn,param_r,param_c,synth]],axis=0)
            np.save("metrics_random.npy", metrics_arr)


    print("DONE!!")

def latent_control(type):

    if type == "lowpass":
        print("Loading Data...")
        s_lowpass = np.load("data generation/s_lowpass_con.npy",allow_pickle=True)
        d_lowpass = np.load("data generation/d_lowpass_con.npy",allow_pickle=True)
        t_lowpass = np.load("data generation/t_lowpass_con.npy",allow_pickle=True)
        test_serum_params = np.load("test_serum_params.npy",allow_pickle=True)
        test_diva_params = np.load("test_diva_params.npy",allow_pickle=True)
        test_tyrell_params = np.load("test_tyrell_params.npy",allow_pickle=True)
        print("Done!")

        m_size = 100

        #define shapes
        l_dim = 64
        i_dim = (1, 128, 431, 1)

        #directory for finding checkpoints
        checkpoint_path = "saved_models/random_dynamic/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        #get latest model
        latest = tf.train.latest_checkpoint(checkpoint_dir)

        #batch_size
        batch_size = 32

        #number of batches in one epoch
        batches_epoch = m_size // batch_size

        #warmup amount
        warmup_it = 100*batches_epoch

        #create model
        m = model.dynamic_vae(64, i_dim, 5, model.optimizer, warmup_it,10)

        #load stored weights
        m.load_weights(latest)

        #compile model
        m.compile(optimizer='adam', loss=losses.MeanSquaredError())

        l_model = tf.keras.Model(inputs=m.get_layer("input_7").input, outputs=m.get_layer("sampling_2").output)

        z_s = l_model.predict([s_lowpass])
        z_d = l_model.predict([d_lowpass])
        z_t = l_model.predict([t_lowpass])

        #create a plot for each latent variable
        for z_num in range(64):
            x_trace = np.arange(0,1,0.01)
            plt.scatter(x_trace, z_s[:,z_num], label="serum")
            plt.scatter(x_trace, z_d[:,z_num], label="diva")
            plt.scatter(x_trace, z_t[:,z_num], label="tyrell")
            plt.xlabel("Cuttoff Frequency Norm")
            plt.ylabel("Z Value")
            plt.legend()
            plt.savefig("Experiment Results/random_model/Latent Control/lowpass_z" + str(z_num) + "_scatter.png")
            plt.close()

    if type == "highpass":
        print("Loading Data...")
        s_highpass = np.load("data generation/s_highpass_con.npy",allow_pickle=True)
        d_highpass = np.load("data generation/d_highpass_con.npy",allow_pickle=True)
        t_highpass = np.load("data generation/t_highpass_con.npy",allow_pickle=True)
        test_serum_params = np.load("test_serum_params.npy",allow_pickle=True)
        test_diva_params = np.load("test_diva_params.npy",allow_pickle=True)
        test_tyrell_params = np.load("test_tyrell_params.npy",allow_pickle=True)
        print("Done!")

        m_size = 100

        m_size = 100

        #define shapes
        l_dim = 64
        i_dim = (1, 128, 431, 1)

        #directory for finding checkpoints
        checkpoint_path = "saved_models/random_dynamic/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        #get latest model
        latest = tf.train.latest_checkpoint(checkpoint_dir)

        #batch_size
        batch_size = 32

        #number of batches in one epoch
        batches_epoch = m_size // batch_size

        #warmup amount
        warmup_it = 100*batches_epoch

        #create model
        m = model.dynamic_vae(64, i_dim, 5, model.optimizer, warmup_it,10)

        #load stored weights
        m.load_weights(latest)

        #compile model
        m.compile(optimizer='adam', loss=losses.MeanSquaredError())

        l_model = tf.keras.Model(inputs=m.get_layer("input_7").input, outputs=m.get_layer("sampling_2").output)

        z_s = l_model.predict([s_highpass])
        z_d = l_model.predict([d_highpass])
        z_t = l_model.predict([t_highpass])

        #create a plot for each latent variable
        for z_num in range(64):
            x_trace = np.arange(0,1,0.01)
            plt.scatter(x_trace, z_s[:,z_num], label="serum")
            plt.scatter(x_trace, z_d[:,z_num], label="diva")
            plt.scatter(x_trace, z_t[:,z_num], label="tyrell")
            plt.xlabel("Cuttoff Frequency Norm")
            plt.ylabel("Z Value")
            plt.legend()
            plt.savefig("Experiment Results/random_model/Latent Control/highpass_z" + str(z_num) + "_scatter.png")
            plt.close()

    if type == "attack":
        print("Loading Data...")
        s_attack = np.load("data generation/s_attack_con.npy",allow_pickle=True)
        d_attack = np.load("data generation/d_attack_con.npy",allow_pickle=True)
        t_attack = np.load("data generation/t_attack_con.npy",allow_pickle=True)
        test_serum_params = np.load("test_serum_params.npy",allow_pickle=True)
        test_diva_params = np.load("test_diva_params.npy",allow_pickle=True)
        test_tyrell_params = np.load("test_tyrell_params.npy",allow_pickle=True)
        print("Done!")

        m_size = 100

        #define shapes
        l_dim = 64
        i_dim = (1, 128, 431, 1)

        #directory for finding checkpoints
        checkpoint_path = "saved_models/random_dynamic/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        #get latest model
        latest = tf.train.latest_checkpoint(checkpoint_dir)

        #batch_size
        batch_size = 32

        #number of batches in one epoch
        batches_epoch = m_size // batch_size

        #warmup amount
        warmup_it = 100*batches_epoch

        #create model
        m = model.dynamic_vae(64, i_dim, 5, model.optimizer, warmup_it,10)

        #load stored weights
        m.load_weights(latest)

        #compile model
        m.compile(optimizer='adam', loss=losses.MeanSquaredError())

        l_model = tf.keras.Model(inputs=m.get_layer("input_7").input, outputs=m.get_layer("sampling_2").output)

        z_s = l_model.predict([s_attack])
        z_d = l_model.predict([d_attack])
        z_t = l_model.predict([t_attack])

        #create a plot for each latent variable
        for z_num in range(64):
            x_trace = np.arange(0,1,0.01)
            plt.scatter(x_trace, z_s[:,z_num], label="serum")
            plt.scatter(x_trace, z_d[:,z_num], label="diva")
            plt.scatter(x_trace, z_t[:,z_num], label="tyrell")
            plt.xlabel("Attack Norm")
            plt.ylabel("Z Value")
            plt.legend()
            plt.savefig("Experiment Results/random_model/Latent Control/attack_z" + str(z_num) + "_scatter.png")
            plt.close()

    if type == "release":
        print("Loading Data...")
        s_release = np.load("data generation/s_release_con.npy",allow_pickle=True)
        d_release = np.load("data generation/d_release_con.npy",allow_pickle=True)
        t_release = np.load("data generation/t_release_con.npy",allow_pickle=True)
        test_serum_params = np.load("test_serum_params.npy",allow_pickle=True)
        test_diva_params = np.load("test_diva_params.npy",allow_pickle=True)
        test_tyrell_params = np.load("test_tyrell_params.npy",allow_pickle=True)
        print("Done!")

        m_size = 100

        #define shapes
        l_dim = 64
        i_dim = (1, 128, 431, 1)

        #directory for finding checkpoints
        checkpoint_path = "saved_models/random_dynamic/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        #get latest model
        latest = tf.train.latest_checkpoint(checkpoint_dir)

        #batch_size
        batch_size = 32

        #number of batches in one epoch
        batches_epoch = m_size // batch_size

        #warmup amount
        warmup_it = 100*batches_epoch

        #create model
        m = model.dynamic_vae(64, i_dim, 5, model.optimizer, warmup_it,10)

        #load stored weights
        m.load_weights(latest)

        #compile model
        m.compile(optimizer='adam', loss=losses.MeanSquaredError())

        l_model = tf.keras.Model(inputs=m.get_layer("input_7").input, outputs=m.get_layer("sampling_2").output)

        z_s = l_model.predict([s_release])
        z_d = l_model.predict([d_release])
        z_t = l_model.predict([t_release])

        #create a plot for each latent variable
        for z_num in range(64):
            x_trace = np.arange(0,1,0.01)
            plt.scatter(x_trace, z_s[:,z_num], label="serum")
            plt.scatter(x_trace, z_d[:,z_num], label="diva")
            plt.scatter(x_trace, z_t[:,z_num], label="tyrell")
            plt.xlabel("Release Norm")
            plt.ylabel("Z Value")
            plt.legend()
            plt.savefig("Experiment Results/random_dynamic/Latent Control/release_z" + str(z_num) + "_scatter.png")
            plt.close()

    if type == "covariance":
        print("Loading Data...")
        s_attack = np.load("data generation/s_attack_con.npy",allow_pickle=True)
        d_attack = np.load("data generation/d_attack_con.npy",allow_pickle=True)
        t_attack = np.load("data generation/t_attack_con.npy",allow_pickle=True)
        s_release = np.load("data generation/s_release_con.npy",allow_pickle=True)
        d_release = np.load("data generation/d_release_con.npy",allow_pickle=True)
        t_release = np.load("data generation/t_release_con.npy",allow_pickle=True)
        s_lowpass = np.load("data generation/s_lowpass_con.npy",allow_pickle=True)
        d_lowpass = np.load("data generation/d_lowpass_con.npy",allow_pickle=True)
        t_lowpass = np.load("data generation/t_lowpass_con.npy",allow_pickle=True)
        s_highpass = np.load("data generation/s_highpass_con.npy",allow_pickle=True)
        d_highpass = np.load("data generation/d_highpass_con.npy",allow_pickle=True)
        t_highpass = np.load("data generation/t_highpass_con.npy",allow_pickle=True)
        print("Done!!!")

        m_size = 100

        #define shapes
        l_dim = 64
        i_dim = (1, 128, 431, 1)

        #directory for finding checkpoints
        checkpoint_path = "saved_models/random_dynamic/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        #get latest model
        latest = tf.train.latest_checkpoint(checkpoint_dir)

        #batch_size
        batch_size = 32

        #number of batches in one epoch
        batches_epoch = m_size // batch_size

        #warmup amount
        warmup_it = 100*batches_epoch

        #create model
        m = model.dynamic_vae(64, i_dim, 5, model.optimizer, warmup_it,10)

        #load stored weights
        m.load_weights(latest)

        #make model into
        l_model = tf.keras.Model(inputs=m.get_layer("input_7").input, outputs=m.get_layer("sampling_2").output)
        z_s_attack = l_model.predict(s_attack)
        z_d_attack = l_model.predict(d_attack)
        z_t_attack = l_model.predict(t_attack)

        z_s_release = l_model.predict(s_release)
        z_d_release = l_model.predict(d_release)
        z_t_release = l_model.predict(t_release)

        z_s_lowpass = l_model.predict(s_lowpass)
        z_d_lowpass = l_model.predict(d_lowpass)
        z_t_lowpass = l_model.predict(t_lowpass)

        z_s_highpass = l_model.predict(s_highpass)
        z_d_highpass = l_model.predict(d_highpass)
        z_t_highpass = l_model.predict(t_highpass)

        a_r_matrix = np.zeros((3,64))
        r_r_matrix = np.zeros((3,64))
        lp_r_matrix = np.zeros((3,64))
        hp_r_matrix = np.zeros((3,64))

        for z_num in range(64):
            x = np.arange(0,1,0.01)
            a_r_matrix[0][z_num] =scipy.stats.pearsonr(x,z_s_attack[:,z_num])[0]
            a_r_matrix[1][z_num] =scipy.stats.pearsonr(x,z_d_attack[:,z_num])[0]
            a_r_matrix[2][z_num]=scipy.stats.pearsonr(x,z_t_attack[:,z_num])[0]

            r_r_matrix[0][z_num] =scipy.stats.pearsonr(x,z_s_release[:,z_num])[0]
            r_r_matrix[1][z_num] =scipy.stats.pearsonr(x,z_d_release[:,z_num])[0]
            r_r_matrix[2][z_num]=scipy.stats.pearsonr(x,z_t_release[:,z_num])[0]

            lp_r_matrix[0][z_num] =scipy.stats.pearsonr(x,z_s_lowpass[:,z_num])[0]
            lp_r_matrix[1][z_num] =scipy.stats.pearsonr(x,z_d_lowpass[:,z_num])[0]
            lp_r_matrix[2][z_num]=scipy.stats.pearsonr(x,z_t_lowpass[:,z_num])[0]

            hp_r_matrix[0][z_num] =scipy.stats.pearsonr(x,z_s_highpass[:,z_num])[0]
            hp_r_matrix[1][z_num] =scipy.stats.pearsonr(x,z_d_highpass[:,z_num])[0]
            hp_r_matrix[2][z_num]=scipy.stats.pearsonr(x,z_t_highpass[:,z_num])[0]
        

        fig, ax = plt.subplots(4)
        ax.figure.set_size_inches(20,20)
        ax.matshow(a_r_matrix)
        ax.set_xticks(np.arange(64), np.arange(64))
        ax.set_yticks([0,1,2],['serum','diva','tyrell'])
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("Z Variable")
        ax.set_ylabel("Synthesizer")
        ax.title("Attack")
        # plt.savefig("Experiment Results/random_model/Latent Correlations/attack")

        fig, ax = plt.subplots()
        ax.figure.set_size_inches(20,20)
        ax.matshow(r_r_matrix)
        ax.set_xticks(np.arange(64), np.arange(64))
        ax.set_yticks([0,1,2],['serum','diva','tyrell'])
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("Z Variable")
        ax.set_ylabel("Synthesizer")
        ax.title("Release")
        plt.savefig("Experiment Results/random_dynamic/Latent Correlations/release")

        fig, ax = plt.subplots()
        ax.figure.set_size_inches(20,20)
        ax.matshow(lp_r_matrix)
        ax.set_xticks(np.arange(64), np.arange(64))
        ax.set_yticks([0,1,2],['serum','diva','tyrell'])
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("Z Variable")
        ax.set_ylabel("Synthesizer")
        ax.title("Lowpass Cuttoff")
        # plt.savefig("Experiment Results/random_model/Latent Correlations/lowpass_cuttoff")

        fig, ax = plt.subplots()
        ax.figure.set_size_inches(20,20)
        ax.matshow(hp_r_matrix)
        ax.set_xticks(np.arange(64), np.arange(64))
        ax.set_yticks([0,1,2],['serum','diva','tyrell'])
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("Z Variable")
        ax.set_ylabel("Synthesizer")
        ax.title("Highpass Cuttoff")
        plt.savefig("Experiment Results/random_model/Latent Correlations/highpass_cuttoff")

        za_select = np.argmax(np.abs(np.sum(a_r_matrix, axis=0)))
        zr_select = np.argmax(np.abs(np.sum(r_r_matrix, axis=0)))
        zlp_select = np.argmax(np.abs(np.sum(lp_r_matrix, axis=0)))
        zhp_select = np.argmax(np.abs(np.sum(hp_r_matrix, axis=0)))

        print(za_select)
        print(zr_select)
        print(zlp_select)
        print(zhp_select)
        input()




if __name__ == "__main__":

    if "l_samp" in sys.argv:
        latent_sampling(int(sys.argv[2]),"spec")

    if "l_samp_param" in sys.argv:
        latent_sampling(0,"param")

    if "l_eval"  in sys.argv:
        evaluate_latent("all")

    if "l_confuse" in sys.argv:
        latent_sampling(int(sys.argv[2]),"confuse")

    if  "l_eval_synth" in sys.argv:
        evaluate_latent("per_synth")

    if "l_eval_hpss" in sys.argv:
        evaluate_latent("per_hpss")

    if "l_control_low" in sys.argv:
        latent_control("lowpass")

    if "l_control_high" in sys.argv:
        latent_control("highpass")

    if "l_control_attack" in sys.argv:
        latent_control("attack")

    if "l_control_release" in sys.argv:
        latent_control("release")

    if "l_control_covar" in sys.argv:
        latent_control("covariance")

    if "rand_samp" in sys.argv:
        random_sample()

    if "out_samp" in sys.argv:
        out_of_domain_sample()

    if "m_acc" in sys.argv:
        model_accuracy()

    if "l_control_covar" in sys.argv:
        latent_control("covariance")
    
