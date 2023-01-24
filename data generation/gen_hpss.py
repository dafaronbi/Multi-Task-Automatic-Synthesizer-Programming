import numpy as np
import librosa
import dawdreamer as dd
import sys
sys.path.append("..")
from data import one_hot
from scipy.io import wavfile

SAMPLING_RATE = 44100

def generate_audio(params, synth):
    plugin_path = ""
    if synth == "serum":
        #path to plugin
        plugin_path = "Serum.vst"

        
    if synth == "diva":
        #path to plugin
        plugin_path = "Diva.vst"

    if synth == "tyrell":
        #path to plugin
        plugin_path = "TyrellN6.vst"

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

if __name__ == "__main__":
    #load data
    print("Loading Data...")
    test_spec_data = np.load("../test_spec.npy",allow_pickle=True)
    test_serum_params = np.load("../test_serum_params.npy",allow_pickle=True)
    test_serum_masks = np.load("../test_serum_masks.npy",allow_pickle=True)
    test_diva_params = np.load("../test_diva_params.npy",allow_pickle=True)
    test_diva_masks = np.load("../test_diva_masks.npy",allow_pickle=True)
    test_tyrell_params = np.load("../test_tyrell_params.npy",allow_pickle=True)
    test_tyrell_masks = np.load("../test_tyrell_masks.npy",allow_pickle=True)
    print("Done!")

    h_labels = np.load("../test_h_labels.npy",allow_pickle=True)
    i = len(h_labels)
    while i < len(test_serum_params):

        h_labels = np.load("../test_h_labels.npy",allow_pickle=True)
        i = len(h_labels)

        audio = []
        accepted = True
        if test_serum_masks[i][0] == 1:

            serum = one_hot.predict(np.squeeze(test_serum_params[i]),one_hot.serum_oh)
            serum = one_hot.decode(serum,one_hot.serum_oh)
            try:
                audio = generate_audio(serum, "serum") 
            except:
                print("PROBLEM SERUM")
                accepted = False

        if test_diva_masks[i][0] == 1:
            diva = one_hot.predict(np.squeeze(test_diva_params[i]),one_hot.diva_oh)
            diva = one_hot.decode(diva,one_hot.diva_oh)

            try:
                audio = generate_audio(diva, "diva") 
            except:
                print("PROBLEM DIVA")
                accepted = False

        if test_tyrell_masks[i][0] == 1:
            tyrell = one_hot.predict(np.squeeze(test_tyrell_params[i]),one_hot.tyrell_oh)
            tyrell = one_hot.decode(tyrell,one_hot.tyrell_oh)

            try:
                audio = generate_audio(tyrell, "tyrell") 
            except:
                print("PROBLEM TYRELL")
                accepted = False

        if accepted:
            #perform hpss
            D = librosa.stft(audio)
            H,P = librosa.decompose.hpss(D, margin=3.0)
            R = D - (H+P)
            d_mag = np.mean(np.abs(D))
            h_mag = np.mean(np.abs(H))
            p_mag = np.mean(np.abs(P))
            r_mag = np.mean(np.abs(R))
            add_mag = np.mean(np.abs(H+P+R))

            h_per = round(h_mag/d_mag,2)*100
            p_per = round(p_mag/d_mag,2)*100
            r_per = round(r_mag/d_mag,2)*100
            add_per = round(add_mag/d_mag,2)*100

            # print(h_per)
            # print(p_per)
            # print(r_per)
            # print(add_per)

            l_value = 20

            if h_per > 20:
                l_value = 40

            if h_per > 40:
                l_value = 60
            
            if h_per > 60:
                l_value = 80

            if h_per > 80:
                l_value = 100

            h_labels = np.append(h_labels,[l_value])
        
            print(h_labels)
            test_spec_data = np.save("../test_h_labels.npy", np.array(h_labels))

