import sys
sys.path.append("..")
from data import one_hot
import librosa
import dawdreamer as dd

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
