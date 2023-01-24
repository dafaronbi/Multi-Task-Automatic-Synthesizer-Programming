import numpy as np
import one_hot
import librosa
import librosa.display
import dawdreamer as dd
from scipy.io import wavfile

#sample rate for geneating audio
SAMPLING_RATE = 44100

if __name__ == "__main__":

    s_params = np.load("../test_serum_params.npy",allow_pickle=True)
    d_params = np.load("../test_diva_params.npy",allow_pickle=True)
    t_params = np.load("../test_tyrell_params.npy",allow_pickle=True)
    print(np.load("serum_params_oh.npy",allow_pickle=True).shape)
    print(np.load("diva_params_oh.npy",allow_pickle=True).shape)
    print(np.load("tyrell_params_oh.npy",allow_pickle=True).shape)

    mel_output = []

    noise_levles = [0,0.2,0.5]

    for i in range(len(s_params)):
        for nl in noise_levles:
            plugin_path = ""
            if not np.all(s_params[i]==0):
                print("serum")
                #one hot decode data
                param = one_hot.decode(s_params[i], one_hot.serum_oh)

                #path to plugin
                plugin_path = "../data generation/Serum.vst"

                
            if not np.all(d_params[i]==0):
                print("diva")
                #one hot decode data
                param = one_hot.decode(d_params[i], one_hot.diva_oh)

                #path to plugin
                plugin_path = "../data generation/Diva.vst"

            if not np.all(t_params[i]==0):
                print("tyrell")
                #one hot decode data
                param = one_hot.decode(t_params[i], one_hot.tyrell_oh)

                #path to plugin
                plugin_path = "../data generation/TyrellN6.vst"

            #create renderman engine with plugin loaded
            engine = dd.RenderEngine(SAMPLING_RATE, 512)
            engine.set_bpm(120)
            synth = engine.make_plugin_processor("Synth", plugin_path)
            engine.load_graph([(synth, [])])
            for j in range(len(np.squeeze(param))):
                synth.set_parameter(j,param[j])
                
            #play new note
            synth.clear_midi()
            synth.add_midi_note(60, 255,0.25,3)
            
            engine.render(5)


            audio = engine.get_audio()
            audio = audio[0] + audio[1]

            #add noise
            noise = np.random.normal(0, .1, size=len(audio))

            audio += nl *noise

            del engine

            # wavfile.write("test" + str(nl) + ".wav", SAMPLING_RATE, audio.transpose())

            #create and normalize mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLING_RATE,)
            mel_spec = librosa.power_to_db(mel_spec,ref=np.max)
            mel_spec = mel_spec - np.min(mel_spec)
            mel_spec = mel_spec / np.max(mel_spec)
            


    np.save("../noise_mels",mel_output)

