import numpy as np
from scipy.io import wavfile
import os
import librosa
import matplotlib.pyplot as plt
import librosa.display
import dawdreamer as dd
import random


if __name__ == "__main__":
    #load data of diva parameters and get one data point
    data = np.load("tyrell_params.npy", allow_pickle=True)


    #path to plugin
    plugin_path = "TyrellN6.vst"

    SAMPLING_RATE = 44100

    #create renderman engine with plugin loaded
    engine = dd.RenderEngine(SAMPLING_RATE, 512)
    engine.set_bpm(120)
    synth = engine.make_plugin_processor("Tyrell", plugin_path)
    engine.load_graph([(synth, [])])
    count = 0

    asp_data = list()
    for i,d_point in enumerate(data):

        for note in range(12):

            for j in range(len(d_point)):
                synth.set_parameter(j,d_point[j])

            #play new note
            synth.clear_midi()
            synth.add_midi_note(60+ note,255,0.25,3)
            
            engine.render(5)
            audio = engine.get_audio()

            #combine to mon
            audio = audio[0] + audio[1]
            rms = np.sqrt(np.mean(audio**2)) 
            if rms > 0.01:
                count += 1
                print(count)

                #create and normalize mel spectrogram
                mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLING_RATE,)
                mel_spec = librosa.power_to_db(mel_spec,ref=np.max)
                mel_spec = mel_spec - np.min(mel_spec)
                mel_spec = mel_spec / np.max(mel_spec)

                asp_data.append(np.array([np.array(mel_spec),np.array(d_point)]))
                # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
                # img = librosa.display.specshow(mel_spec, y_axis='mel', x_axis='time', ax=ax)
                # ax.set(title='Mel-frequency power spectrogram')
                # ax.label_outer()
                # fig.colorbar(img, ax=ax, format = "%+2.f dB")
                # wavfile.write('audio ' + str(i) + '.' + str(note) + '.wav', SAMPLING_RATE, audio.transpose())
                # plt.show()

    np.save("asp_data_tyrell", np.array(asp_data))
    print(np.array(asp_data))
    print("Done!!")
