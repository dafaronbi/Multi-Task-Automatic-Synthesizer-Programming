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

#print evaluation on test set
loss, loss1,loss2 = m.evaluate(train_data.get_mels(),[train_data.get_mels(),train_data.get_params()])
print("training model loss = " + str(loss) + "\n training model spectrogram loss = "+ str(loss1) + "\n training model synth_param loss = "+ str(loss2))

#print evaluation on test set
loss, loss1,loss2 = m.evaluate(test_data.get_mels(),[test_data.get_mels(),test_data.get_params()])
print("test model loss = " + str(loss) + "\n test model spectrogram loss = "+ str(loss1) + "\n test model synth_param loss = "+ str(loss2))

#get 10 random predictiions
for i in range(10):

    #make directory to store outputs of model
    if not os.path.isdir("output/"+ sys.argv[1] + "/example_" + str(i)):
        os.makedirs("output/"+ sys.argv[1] + "/example_" + str(i))

    #set where outputs are saved
    output_dir = "output/"+ sys.argv[1] + "/example_" +str(i)

    index = np.random.randint(0,test_data.get_mels().shape[0])

    #get prediction
    spectogram,params = m.predict(test_data.get_mels()[[index]])

    #evaluate reconstruction of 30th test file
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    img = librosa.display.specshow(test_data.get_mels()[index], y_axis='mel', x_axis='time', ax=ax[0])

    ax[0].set(title='Mel-Frequency Spectrogram Reconstruction ' + str(i))
    ax[0].label_outer()

    librosa.display.specshow(np.squeeze(spectogram), y_axis='mel', x_axis='time', ax=ax[1])
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    plt.savefig(output_dir + "/reconstruction_" + str(i) + ".png")
    plt.clf()

    test_synth = test_data.get_params()[index]

    #create serum synthesizer object
    synth = spiegelib.synth.SynthVST("/Library/Audio/Plug-Ins/Components/Serum.component")

    #generate ground truth audio from synth parameters
    synth.set_patch(test_synth)
    synth.render_patch()
    audio = synth.get_audio()
    audio.plot_spectrogram()
    audio.save(output_dir + "/truth_audio_" + str(i) + ".wav")

    #show plots
    plt.savefig(output_dir + "/truth_param_spec_" + str(i) + ".png")
    plt.clf()

    #generate predict audio from synth parameters
    synth.set_patch(np.squeeze(params))
    synth.render_patch()
    audio = synth.get_audio()
    audio.plot_spectrogram()
    audio.save(output_dir + "/predict_audio_" + str(i) +".wav")

    #show plots
    plt.savefig(output_dir + "/predict_param_spec_" + str(i) + ".png")
    plt.clf()
