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
from scipy.io import wavfile

#load data
print("Loading Data...")
train_data = ds.melParamData("train","data")
test_data = ds.melParamData("test","data")
validation_data = ds.melParamData("validation","data")
print("Done!")


#directory for finding checkpoints
checkpoint_path = "saved_models/"+ sys.argv[1] + "/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

#get latest model
latest = tf.train.latest_checkpoint(checkpoint_dir)

#create autoencoder model
m = model.get_model[sys.argv[1]]

#load stored weights
m.load_weights(latest)

#compile model
m.compile(optimizer='adam', loss=losses.MeanSquaredError())


for i in range(len(test_data.get_params())):

    #get prediction
    _,params_pred = m.predict(test_data.get_mels()[[i]])

    #get ground truth
    params_truth = np.squeeze(test_data.get_params()[i])
    error = np.zeros_like(params_truth)
    #make error metric     
    error += (params_pred[0] - params_truth)**2

error = error / len(test_data.get_params())

#path to plugin
plugin_path = "data generation/Serum.vst"
SAMPLING_RATE = 44100

#create renderman engine with plugin loaded
engine = dd.RenderEngine(SAMPLING_RATE, 512)
engine.set_bpm(120)
synth = engine.make_plugin_processor("Serum", plugin_path)
engine.load_graph([(synth, [])])
count = 0

error_dic = {param['name']:error[i] for i,param in enumerate(synth.get_plugin_parameters_description())}
error_dic = dict(sorted(error_dic.items(), key=lambda item: item[1]))

for k in error_dic.keys():
    print(k + ": " + str(error_dic[k]))

exit()

# #print evaluation on training set
# loss, loss1,loss2 = m.evaluate(train_data.get_mels(),[train_data.get_mels(),train_data.get_params()])
# print("training model loss = " + str(loss) + "\n training model spectrogram loss = "+ str(loss1) + "\n training model synth_param loss = "+ str(loss2))

# #print evaluation on test set
# loss, loss1,loss2 = m.evaluate(test_data.get_mels(),[test_data.get_mels(),test_data.get_params()])
# print("test model loss = " + str(loss) + "\n test model spectrogram loss = "+ str(loss1) + "\n test model synth_param loss = "+ str(loss2))

# #evaluate output of multiple layers
# get_all_layer_outputs = K.function([m.layers[0].input], {l.name:l.output  for l in m.layers[1:]})

# output_dict = get_all_layer_outputs(test_data.get_mels()[0])

# for k in output_dict.keys():
#     print(k)
#     print(output_dict[k])

#path to plugin
plugin_path = "data generation/Serum.vst"
SAMPLING_RATE = 44100

#create renderman engine with plugin loaded
engine = dd.RenderEngine(SAMPLING_RATE, 512)
engine.set_bpm(120)
synth = engine.make_plugin_processor("Serum", plugin_path)
engine.load_graph([(synth, [])])
count = 0

#get 10 random test predictiions
for i in range(10):

    #make directory to store outputs of model
    if not os.path.isdir("output/"+ sys.argv[1] + "/test/example_" + str(i)):
        os.makedirs("output/"+ sys.argv[1] + "/test/example_" + str(i))

    #set where outputs are saved
    output_dir = "output/"+ sys.argv[1] + "/test/example_" +str(i)

    index = np.random.randint(0,test_data.get_mels().shape[0])

    #get prediction
    spectogram,params = m.predict(test_data.get_mels()[[index]])

    #evaluate reconstruction test file
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    img = librosa.display.specshow(test_data.get_mels()[index], y_axis='mel', x_axis='time', ax=ax[0])

    ax[0].set(title='Mel-Frequency Spectrogram Reconstruction ' + str(i))
    ax[0].label_outer()

    librosa.display.specshow(np.squeeze(spectogram), y_axis='mel', x_axis='time', ax=ax[1])
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    plt.savefig(output_dir + "/reconstruction_" + str(i) + ".png")
    plt.clf()

    #ground trun parameters
    test_synth = test_data.get_params()[index]

    #generate ground truth audio from synth parameters
    for j in range(len(test_synth)):
        synth.set_parameter(j,test_synth[j])
            
    #play new note
    synth.clear_midi()
    synth.add_midi_note(60, 255,0.25,3)
    
    engine.render(5)
    audio = engine.get_audio()
    audio = audio[0] + audio[1]

    #plot audio
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLING_RATE,)
    mel_spec = librosa.power_to_db(mel_spec,ref=np.max)
    mel_spec = mel_spec - np.min(mel_spec)
    mel_spec = mel_spec / np.max(mel_spec)

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    img = librosa.display.specshow(mel_spec, y_axis='mel', x_axis='time', ax=ax)
    ax.set(title='Mel-frequency power spectrogram')
    ax.label_outer()
    fig.colorbar(img, ax=ax, format = "%+2.f dB")

    wavfile.write(output_dir + "/truth_audio_" + str(i) + ".wav", SAMPLING_RATE, audio.transpose())

    #save plots
    plt.savefig(output_dir + "/truth_param_spec_" + str(i) + ".png")
    plt.clf()


    # synth.set_patch(test_synth)
    # synth.render_patch()
    # audio = synth.get_audio()
    # audio.plot_spectrogram()
    # audio.save(output_dir + "/truth_audio_" + str(i) + ".wav")

    # #show plots
    # plt.savefig(output_dir + "/truth_param_spec_" + str(i) + ".png")
    # plt.clf()

    #generate ground truth audio from synth parameters
    for j in range(len(np.squeeze(params))):
        synth.set_parameter(j,np.squeeze(params)[j])
            
    #play new note
    synth.clear_midi()
    synth.add_midi_note(60, 255,0.25,3)
    
    engine.render(5)
    audio = engine.get_audio()
    audio = audio[0] + audio[1]

    #plot audio
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLING_RATE,)
    mel_spec = librosa.power_to_db(mel_spec,ref=np.max)
    mel_spec = mel_spec - np.min(mel_spec)
    mel_spec = mel_spec / np.max(mel_spec)

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    img = librosa.display.specshow(mel_spec, y_axis='mel', x_axis='time', ax=ax)
    ax.set(title='Mel-frequency power spectrogram')
    ax.label_outer()
    fig.colorbar(img, ax=ax, format = "%+2.f dB")

    wavfile.write(output_dir + "/predict_audio_" + str(i) +".wav", SAMPLING_RATE, audio.transpose())

    #save plots
    plt.savefig(output_dir + "/predict_param_spec_" + str(i) + ".png")
    plt.clf()


#get 10 random train predictiions
for i in range(10):

    #make directory to store outputs of model
    if not os.path.isdir("output/"+ sys.argv[1] + "/train/example_" + str(i)):
        os.makedirs("output/"+ sys.argv[1] + "/train/example_" + str(i))

    #set where outputs are saved
    output_dir = "output/"+ sys.argv[1] + "/train/example_" +str(i)

    index = np.random.randint(0,train_data.get_mels().shape[0])

    #get prediction
    spectogram,params = m.predict(train_data.get_mels()[[index]])

    #evaluate reconstruction test file
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    img = librosa.display.specshow(train_data.get_mels()[index], y_axis='mel', x_axis='time', ax=ax[0])

    ax[0].set(title='Mel-Frequency Spectrogram Reconstruction ' + str(i))
    ax[0].label_outer()

    librosa.display.specshow(np.squeeze(spectogram), y_axis='mel', x_axis='time', ax=ax[1])
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    plt.savefig(output_dir + "/reconstruction_" + str(i) + ".png")
    plt.clf()

    #ground trun parameters
    train_synth = train_data.get_params()[index]

    #generate ground truth audio from synth parameters
    for j in range(len(train_synth)):
        synth.set_parameter(j,train_synth[j])
            
    #play new note
    synth.clear_midi()
    synth.add_midi_note(60, 255,0.25,3)
    
    engine.render(5)
    audio = engine.get_audio()
    audio = audio[0] + audio[1]

    #plot audio
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLING_RATE,)
    mel_spec = librosa.power_to_db(mel_spec,ref=np.max)
    mel_spec = mel_spec - np.min(mel_spec)
    mel_spec = mel_spec / np.max(mel_spec)

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    img = librosa.display.specshow(mel_spec, y_axis='mel', x_axis='time', ax=ax)
    ax.set(title='Mel-frequency power spectrogram')
    ax.label_outer()
    fig.colorbar(img, ax=ax, format = "%+2.f dB")

    wavfile.write(output_dir + "/truth_audio_" + str(i) + ".wav", SAMPLING_RATE, audio.transpose())

    #save plots
    plt.savefig(output_dir + "/truth_param_spec_" + str(i) + ".png")
    plt.clf()


    # synth.set_patch(test_synth)
    # synth.render_patch()
    # audio = synth.get_audio()
    # audio.plot_spectrogram()
    # audio.save(output_dir + "/truth_audio_" + str(i) + ".wav")

    # #show plots
    # plt.savefig(output_dir + "/truth_param_spec_" + str(i) + ".png")
    # plt.clf()

    #generate ground truth audio from synth parameters
    for j in range(len(np.squeeze(params))):
        synth.set_parameter(j,np.squeeze(params)[j])
            
    #play new note
    synth.clear_midi()
    synth.add_midi_note(60, 255,0.25,3)
    
    engine.render(5)
    audio = engine.get_audio()
    audio = audio[0] + audio[1]

    #plot audio
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLING_RATE,)
    mel_spec = librosa.power_to_db(mel_spec,ref=np.max)
    mel_spec = mel_spec - np.min(mel_spec)
    mel_spec = mel_spec / np.max(mel_spec)

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    img = librosa.display.specshow(mel_spec, y_axis='mel', x_axis='time', ax=ax)
    ax.set(title='Mel-frequency power spectrogram')
    ax.label_outer()
    fig.colorbar(img, ax=ax, format = "%+2.f dB")

    wavfile.write(output_dir + "/predict_audio_" + str(i) +".wav", SAMPLING_RATE, audio.transpose())

    #save plots
    plt.savefig(output_dir + "/predict_param_spec_" + str(i) + ".png")
    plt.clf()

    # #generate predict audio from synth parameters
    # synth.set_patch(np.squeeze(params))
    # synth.render_patch()
    # audio = synth.get_audio()
    # audio.plot_spectrogram()
    # audio.save(output_dir + "/predict_audio_" + str(i) +".wav")

    # #show plots
    # plt.savefig(output_dir + "/predict_param_spec_" + str(i) + ".png")
    # plt.clf()
