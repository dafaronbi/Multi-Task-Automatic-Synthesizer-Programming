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

#load data
print("Loading Data...")
train_data = ds.melParamData("train","data")
test_data = ds.melParamData("test","data")
validation_data = ds.melParamData("validation","data")
print("Done!")




# print(serum_param_dic)
# print(test_synth)

#directory for finding checkpoints
checkpoint_path = "new_models4/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

#get latest model
latest = tf.train.latest_checkpoint(checkpoint_dir)

#create autoencoder model
autoencoder = model.autoencoder3(64,train_data.get_mels()[:10,...,np.newaxis].shape,train_data.get_params().shape)

#load stored weights
autoencoder.load_weights(latest)

#compile model
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

#print evaluation on test set
loss, loss1,loss2 = autoencoder.evaluate(train_data.get_mels(),[train_data.get_mels(),train_data.get_params()])
print("training model loss = " + str(loss) + "\n training model spectrogram loss = "+ str(loss1) + "\n training model synth_param loss = "+ str(loss2))

#print evaluation on test set
loss, loss1,loss2 = autoencoder.evaluate(test_data.get_mels(),[test_data.get_mels(),test_data.get_params()])
print("test model loss = " + str(loss) + "\n test model spectrogram loss = "+ str(loss1) + "\n test model synth_param loss = "+ str(loss2))

#get prediction
spectogram,params = autoencoder.predict(test_data.get_mels()[[30]])

#evaluate reconstruction of 30th test file
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
img = librosa.display.specshow(test_data.get_mels()[30], y_axis='mel', x_axis='time', ax=ax[0])

ax[0].set(title='Mel-Frequency Spectrogram Reconstruction')
ax[0].label_outer()

librosa.display.specshow(np.squeeze(spectogram), y_axis='mel', x_axis='time', ax=ax[1])
fig.colorbar(img, ax=ax, format="%+2.f dB")


# print("Ground Truth Parameters:" + str(test_data.get_params()[500]-params))
# print("Predicted Parameters" + str(params))

plt.show()

test_synth = test_data.get_params()[30]

#create serum synthesizer object
synth = spiegelib.synth.SynthVST("/Library/Audio/Plug-Ins/Components/Serum.component")

#generate ground truth audio from synth parameters
synth.set_patch(test_synth)
synth.render_patch()
audio = synth.get_audio()
audio.plot_spectrogram()
audio.save("test_audio.wav")

#show plots
plt.show()

#generate predict audio from synth parameters
synth.set_patch(np.squeeze(params))
synth.render_patch()
audio = synth.get_audio()
audio.plot_spectrogram()
audio.save("predict_audio.wav")

#show plots
plt.show()
