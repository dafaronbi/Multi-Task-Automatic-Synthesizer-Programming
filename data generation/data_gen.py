import numpy as np
import os
import librosa
import librosa.display
import spiegelib
import matplotlib.pyplot as plt

#create class for mel featres
class melSpectrogramFeatures(spiegelib.features.features_base.FeaturesBase):

    def get_features(self, audio):
        features = librosa.feature.melspectrogram(y=audio.get_audio(), sr=self.sample_rate,)
        return features

#create class for raw audio
class audioFeatures(spiegelib.features.features_base.FeaturesBase):

    def get_features(self, audio):
        return audio.get_audio()


#load synth to use
synth = spiegelib.synth.SynthVST("/Library/Audio/Plug-Ins/Components/Serum.component")

#make class for extracting mel spectrogram features
melSpec = melSpectrogramFeatures()

#class instance for getting audio features
audioFet = audioFeatures()

#setup location for dataset generation
output_location = "/Volumes/USB30FD/Synth_DataSet"

#setup generator
# generator = spiegelib.DatasetGenerator(synth,melSpec,output_folder=".",save_audio=True)
# generator.generate(2, file_prefix="mel_")
#generate audio
# generator.generate(3000, file_prefix="train_1_")
# generator.generate(3000, file_prefix="train_2_")
# generator.generate(3000, file_prefix="train_3_")
# generator.generate(2000, file_prefix="test_")

mels = np.load("mel_features.npy")

fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
S_dB = librosa.power_to_db(mels[0],ref=np.max)
img = librosa.display.specshow(S_dB, y_axis='mel', x_axis='time', ax=ax)
ax.set(title='Mel-frequency power spectrogram')
ax.label_outer()
fig.colorbar(img, ax=ax, format = "%+2.f dB")
plt.show()