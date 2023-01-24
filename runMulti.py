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

# t_layer  = m.get_layer("serum")
# print(t_layer.kernel)
# np.save("serum_kernel",t_layer.kernel)

# t_layer  = m.get_layer("diva")
# print(t_layer.kernel)
# np.save("diva_kernel",t_layer.kernel)

# t_layer  = m.get_layer("tyrell")
# print(t_layer.kernel)
# np.save("tyrell_kernel",t_layer.kernel)

# exit()

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
zeros = np.zeros((1,64))

for z_num in range(0,100,5):
    zeros[0][40] = z_num
    spec,serum,diva,tyrell = spec_model.predict([zeros,np.ones((1,64)), np.ones((1,480)),np.ones((1,759)), np.ones((1,327))])

    s_spec = generate_spectrogram(np.squeeze(serum), "serum")
    d_spec = generate_spectrogram(np.squeeze(diva), "diva")
    t_spec = generate_spectrogram(np.squeeze(tyrell), "tyrell")

    fig, a = plt.subplots(4)

    a[0].imshow(np.squeeze(spec))
    a[0].set_title('Spectrogram Generation')
    a[1].imshow(np.squeeze(s_spec))
    a[1].set_title('Serum Audio Spectrogram')
    a[2].imshow(np.squeeze(d_spec))
    a[2].set_title('Diva Audio Spectrogram')
    a[3].imshow(np.squeeze(t_spec))
    a[3].set_title('Tyrell Audio Spectrogram')

    plt.show()

exit()
#latent test

# print("fitting k_means...")
# audio_cluster = KMeans(n_clusters=5,init='k-means++')
# audio_cluster.fit(test_spec_data.reshape(test_spec_data.shape[0],test_spec_data.shape[1]*test_spec_data.shape[2]))
# print("done")
# print(len(audio_cluster.labels_))
# print(len(test_spec_data))

# c_0_amount = 0
# c_1_amount = 0
# c_2_amount = 0
# c_3_amount = 0
# c_4_amount = 0

# z_0 = np.zeros(64)
# z_1 = np.zeros(64)
# z_2 = np.zeros(64)
# z_3 = np.zeros(64)
# z_4 = np.zeros(64)

# print("summing latents...")
# for i,l in enumerate(audio_cluster.labels_):
#     if l == 0:
#         c_0_amount += 1      
#         z_0 += np.squeeze(l_model.predict([np.array([test_spec_data[i]]), np.array([test_serum_masks[i]]),np.array([test_diva_masks[i]]), np.array([test_tyrell_masks[i]])]))

#     if l == 1:
#         c_1_amount += 1
#         z_1 += np.squeeze(l_model.predict([np.array([test_spec_data[i]]), np.array([test_serum_masks[i]]),np.array([test_diva_masks[i]]), np.array([test_tyrell_masks[i]])]))

#     if l == 2:
#         c_2_amount += 1
#         z_2 += np.squeeze(l_model.predict([np.array([test_spec_data[i]]), np.array([test_serum_masks[i]]),np.array([test_diva_masks[i]]), np.array([test_tyrell_masks[i]])]))

#     if l == 3:
#         c_3_amount += 1
#         z_3 += np.squeeze(l_model.predict([np.array([test_spec_data[i]]), np.array([test_serum_masks[i]]),np.array([test_diva_masks[i]]), np.array([test_tyrell_masks[i]])]))

#     if l == 4:
#         c_4_amount += 1
#         z_4 += np.squeeze(l_model.predict([np.array([test_spec_data[i]]), np.array([test_serum_masks[i]]),np.array([test_diva_masks[i]]), np.array([test_tyrell_masks[i]])]))

# z_0 = z_0 / c_0_amount
# z_1 = z_1 / c_1_amount
# z_2 = z_2 / c_2_amount
# z_3 = z_3 / c_3_amount
# z_4 = z_4 / c_4_amount

# np.save("z_0",z_0)
# np.save("z_1",z_1)
# np.save("z_2",z_2)
# np.save("z_3",z_3)
# np.save("z_4",z_4)

z_0 = np.load("z_0.npy",allow_pickle=True)
z_1 = np.load("z_1.npy",allow_pickle=True)
z_2 = np.load("z_2.npy",allow_pickle=True)
z_3= np.load("z_3.npy",allow_pickle=True)
z_4 = np.load("z_4.npy",allow_pickle=True)

plt.imshow(np.array([z_0,z_1,z_2,z_3,z_4]))
plt.show()

print("Done!!")

exit()

# print(audio_cluster.cluster_centers_)

_,serum_p,diva_p,tyrell_p = m.predict([test_spec_data, test_serum_masks,test_diva_masks, test_tyrell_masks])

s_class_err = 0
d_class_err = 0
t_class_err = 0
s_n_class_err = 0
d_n_class_err = 0
t_n_class_err = 0

for i in range(len(serum_p)):
    if test_serum_masks[i][0]: 
        # predict_audio =  generate_audio(one_hot.decode(one_hot.predict(serum_p[i], one_hot.serum_oh), one_hot.serum_oh), "serum")
        # test_audio = generate_audio(one_hot.decode(test_serum_params[i], one_hot.serum_oh), "serum")
        dist = class_acuracy(one_hot.predict(serum_p[i], one_hot.serum_oh), test_serum_params[i], one_hot.serum_oh)
        
        s_class_err += dist
        s_n_class_err += 1

    if test_diva_masks[i][0]:
        # predict_audio = generate_audio(one_hot.decode(one_hot.predict(diva_p[i], one_hot.diva_oh), one_hot.diva_oh), "diva")
        # test_audio = generate_audio(one_hot.decode(test_diva_params[i], one_hot.diva_oh), "diva")
        dist = class_acuracy(one_hot.predict(diva_p[i], one_hot.diva_oh), test_diva_params[i], one_hot.diva_oh)

        d_class_err += dist
        d_n_class_err += 1


    if test_tyrell_masks[i][0]:
        # predict_audio = generate_audio(one_hot.decode(one_hot.predict(tyrell_p[0], one_hot.tyrell_oh), one_hot.tyrell_oh), "tyrell")
        # test_audio = generate_audio(one_hot.decode(test_tyrell_params[0], one_hot.tyrell_oh), "tyrell")
        dist = class_acuracy(one_hot.predict(tyrell_p[i], one_hot.tyrell_oh), test_tyrell_params[i], one_hot.tyrell_oh)

        t_class_err += dist
        t_n_class_err += 1

print("serum")
print(s_class_err/s_n_class_err)

print("diva")
print(d_class_err/d_n_class_err)

print("tyrell")
print(t_class_err/t_n_class_err)

# print(serum_p.shape)
# print(diva_p.shape)
# print(tyrell_p.shape)

# print(class_acuracy(test_serum_params[20], one_hot.predict(serum_p[0], one_hot.serum_oh), one_hot.serum_oh))
# print(class_acuracy(test_diva_params[10], one_hot.predict(diva_p[0], one_hot.diva_oh), one_hot.diva_oh))

# predict_audio =  generate_audio(one_hot.decode(one_hot.predict(serum_p[0], one_hot.serum_oh), one_hot.serum_oh), "serum")
# test_audio = generate_audio(one_hot.decode(test_serum_params[10], one_hot.serum_oh), "serum")
# wavfile.write("serum_predict.wav", SAMPLING_RATE, predict_audio)
# wavfile.write("serum_truth.wav", SAMPLING_RATE, test_audio)
# predict_audio = generate_audio(one_hot.decode(one_hot.predict(diva_p[0], one_hot.diva_oh), one_hot.diva_oh), "diva")
# test_audio = generate_audio(one_hot.decode(test_diva_params[70], one_hot.diva_oh), "diva")
# wavfile.write("diva_predict.wav", SAMPLING_RATE, predict_audio)
# wavfile.write("diva_truth.wav", SAMPLING_RATE, test_audio)
# predict_audio = generate_audio(one_hot.decode(one_hot.predict(tyrell_p[0], one_hot.tyrell_oh), one_hot.tyrell_oh), "tyrell")
# test_audio = generate_audio(one_hot.decode(test_tyrell_params[40], one_hot.tyrell_oh), "tyrell")
# wavfile.write("tyrell_predict.wav", SAMPLING_RATE, predict_audio)
# wavfile.write("tyrell_truth.wav", SAMPLING_RATE, test_audio)

# print(predict_audio)

# print("THE FINAL THING!!")

#print evaluation on test set
loss, loss1,loss2,loss3,loss4 = m.evaluate([test_spec_data, test_serum_masks,test_diva_masks,test_tyrell_masks],[test_spec_data, test_serum_params,test_diva_params, test_tyrell_params],2)
print("model loss = " + str(loss) + "\n model spectrogram loss = "+ str(loss1) + "\n model serum_synth_param loss = "+ str(loss2) + "\n model diva_synth_param loss = "+ str(loss3) + "\n model tyrell_synth_param loss = "+ str(loss4))


