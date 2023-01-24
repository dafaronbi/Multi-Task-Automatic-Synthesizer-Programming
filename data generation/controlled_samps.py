import numpy as np
import sys
sys.path.append("..")
from scipy.io import wavfile
import librosa
import dawdreamer as dd

SAMPLING_RATE = 44100

def low_pass(s_type):
    if s_type == "serum":
        #path to plugin
        plugin_path = "serum.vst"

        #create renderman engine with plugin loaded
        engine = dd.RenderEngine(SAMPLING_RATE, 512)
        engine.set_bpm(120)
        synth = engine.make_plugin_processor("Serum", plugin_path)
        engine.load_graph([(synth, [])])

        synth.load_preset("serum_base.fxb")

        synth.set_parameter(216,1)
        
        s_lowpass_spec = []

        for filter_cut in np.arange(0,1,0.01):

            synth.set_parameter(45,filter_cut)

            #play new note
            synth.clear_midi()
            synth.add_midi_note(60, 255,0.25,3)
            
            engine.render(5)

            audio = engine.get_audio()
            audio = audio[0] + audio[1]

            audio = audio.transpose()

            #generate spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLING_RATE,)
            mel_spec = librosa.power_to_db(mel_spec,ref=np.max)
            mel_spec = mel_spec - np.min(mel_spec)
            mel_spec = mel_spec / np.max(mel_spec)

            s_lowpass_spec.append(mel_spec)


        np.save("s_lowpass_con", np.array(s_lowpass_spec))
        
        for i,param in enumerate(synth.get_plugin_parameters_description()):
            print(str(i) + " : "+param['name'] + " : "+ param['text']  )

    if s_type == "diva":
        #path to plugin
        plugin_path = "diva.vst"

        #create renderman engine with plugin loaded
        engine = dd.RenderEngine(SAMPLING_RATE, 512)
        engine.set_bpm(120)
        synth = engine.make_plugin_processor("Diva", plugin_path)
        engine.load_graph([(synth, [])])

        synth.load_preset("diva_base.fxb")
        
        d_lowpass_spec = []

        for filter_cut in np.arange(0,1,0.01):

            synth.set_parameter(148,filter_cut)

            #play new note
            synth.clear_midi()
            synth.add_midi_note(60, 255,0.25,3)
            
            engine.render(5)

            audio = engine.get_audio()
            audio = audio[0] + audio[1]

            audio = audio.transpose()

            #generate spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLING_RATE,)
            mel_spec = librosa.power_to_db(mel_spec,ref=np.max)
            mel_spec = mel_spec - np.min(mel_spec)
            mel_spec = mel_spec / np.max(mel_spec)

            d_lowpass_spec.append(mel_spec)


        np.save("d_lowpass_con", np.array(d_lowpass_spec))

        for i,param in enumerate(synth.get_plugin_parameters_description()):
            print(str(i) + " : "+param['name'] + " : "+ param['text']  )

        # wavfile.write("audio_out.wav", SAMPLING_RATE, audio)
    
    if s_type == "tyrell":
        #path to plugin
        plugin_path = "TyrellN6.vst"

        #create renderman engine with plugin loaded
        engine = dd.RenderEngine(SAMPLING_RATE, 512)
        engine.set_bpm(120)
        synth = engine.make_plugin_processor("tyrell", plugin_path)
        engine.load_graph([(synth, [])])

        synth.load_preset("tyrell_base.fxb")
        
        t_lowpass_spec = []

        for filter_cut in np.arange(0,1,0.01):

            synth.set_parameter(62,filter_cut)

            #play new note
            synth.clear_midi()
            synth.add_midi_note(60, 255,0.25,3)
            
            engine.render(5)

            audio = engine.get_audio()
            audio = audio[0] + audio[1]

            audio = audio.transpose()

            #generate spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLING_RATE,)
            mel_spec = librosa.power_to_db(mel_spec,ref=np.max)
            mel_spec = mel_spec - np.min(mel_spec)
            mel_spec = mel_spec / np.max(mel_spec)

            t_lowpass_spec.append(mel_spec)


        np.save("t_lowpass_con", np.array(t_lowpass_spec))

        synth.set_parameter(62,0.7)

        for i,param in enumerate(synth.get_plugin_parameters_description()):
            print(str(i) + " : "+param['name'] + " : "+ param['text']  )

def high_pass(s_type):
    if s_type == "serum":
        #path to plugin
        plugin_path = "serum.vst"

        #create renderman engine with plugin loaded
        engine = dd.RenderEngine(SAMPLING_RATE, 512)
        engine.set_bpm(120)
        synth = engine.make_plugin_processor("Serum", plugin_path)
        engine.load_graph([(synth, [])])

        synth.load_preset("serum_hp_base.fxb")

        synth.set_parameter(216,1)
        
        s_lowpass_spec = []

        for filter_cut in np.arange(0,1,0.01):

            synth.set_parameter(45,filter_cut)

            #play new note
            synth.clear_midi()
            synth.add_midi_note(60, 255,0.25,3)
            
            engine.render(5)

            audio = engine.get_audio()
            audio = audio[0] + audio[1]

            audio = audio.transpose()

            #generate spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLING_RATE,)
            mel_spec = librosa.power_to_db(mel_spec,ref=np.max)
            mel_spec = mel_spec - np.min(mel_spec)
            mel_spec = mel_spec / np.max(mel_spec)

            s_lowpass_spec.append(mel_spec)


        np.save("s_highpass_con", np.array(s_lowpass_spec))
        
        for i,param in enumerate(synth.get_plugin_parameters_description()):
            print(str(i) + " : "+param['name'] + " : "+ param['text']  )

    if s_type == "diva":
        #path to plugin
        plugin_path = "diva.vst"

        #create renderman engine with plugin loaded
        engine = dd.RenderEngine(SAMPLING_RATE, 512)
        engine.set_bpm(120)
        synth = engine.make_plugin_processor("Diva", plugin_path)
        engine.load_graph([(synth, [])])

        synth.load_preset("diva_hp_base.fxb")
        
        d_lowpass_spec = []

        for filter_cut in np.arange(0,1,0.01):

            synth.set_parameter(148,filter_cut)

            #play new note
            synth.clear_midi()
            synth.add_midi_note(60, 255,0.25,3)
            
            engine.render(5)

            audio = engine.get_audio()
            audio = audio[0] + audio[1]

            audio = audio.transpose()

            #generate spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLING_RATE,)
            mel_spec = librosa.power_to_db(mel_spec,ref=np.max)
            mel_spec = mel_spec - np.min(mel_spec)
            mel_spec = mel_spec / np.max(mel_spec)

            d_lowpass_spec.append(mel_spec)


        np.save("d_highpass_con", np.array(d_lowpass_spec))

        for i,param in enumerate(synth.get_plugin_parameters_description()):
            print(str(i) + " : "+param['name'] + " : "+ param['text']  )

        # wavfile.write("audio_out.wav", SAMPLING_RATE, audio)
    
    if s_type == "tyrell":
        #path to plugin
        plugin_path = "TyrellN6.vst"

        #create renderman engine with plugin loaded
        engine = dd.RenderEngine(SAMPLING_RATE, 512)
        engine.set_bpm(120)
        synth = engine.make_plugin_processor("tyrell", plugin_path)
        engine.load_graph([(synth, [])])

        synth.load_preset("tyrell_hp_base.fxb")
        
        t_lowpass_spec = []

        for filter_cut in np.arange(0,1,0.01):

            synth.set_parameter(62,filter_cut)

            #play new note
            synth.clear_midi()
            synth.add_midi_note(60, 255,0.25,3)
            
            engine.render(5)

            audio = engine.get_audio()
            audio = audio[0] + audio[1]

            audio = audio.transpose()

            #generate spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLING_RATE,)
            mel_spec = librosa.power_to_db(mel_spec,ref=np.max)
            mel_spec = mel_spec - np.min(mel_spec)
            mel_spec = mel_spec / np.max(mel_spec)

            t_lowpass_spec.append(mel_spec)


        np.save("t_highpass_con", np.array(t_lowpass_spec))

        synth.set_parameter(62,0.7)

        for i,param in enumerate(synth.get_plugin_parameters_description()):
            print(str(i) + " : "+param['name'] + " : "+ param['text']  )
    
def attack(s_type):
    if s_type == "serum":
        #path to plugin
        plugin_path = "serum.vst"

        #create renderman engine with plugin loaded
        engine = dd.RenderEngine(SAMPLING_RATE, 512)
        engine.set_bpm(120)
        synth = engine.make_plugin_processor("Serum", plugin_path)
        engine.load_graph([(synth, [])])

        synth.load_preset("serum_base.fxb")

        synth.set_parameter(216,1)
        
        s_lowpass_spec = []

        for attack in np.arange(0,0.5,0.005):

            synth.set_parameter(35,attack)

            #play new note
            synth.clear_midi()
            synth.add_midi_note(60, 255,0.25,3)
            
            engine.render(5)

            audio = engine.get_audio()
            audio = audio[0] + audio[1]

            audio = audio.transpose()

            #generate spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLING_RATE,)
            mel_spec = librosa.power_to_db(mel_spec,ref=np.max)
            mel_spec = mel_spec - np.min(mel_spec)
            mel_spec = mel_spec / np.max(mel_spec)

            s_lowpass_spec.append(mel_spec)


        np.save("s_attack_con", np.array(s_lowpass_spec))

        for i,param in enumerate(synth.get_plugin_parameters_description()):
            print(str(i) + " : "+param['name'] + " : "+ param['text']  )

    if s_type == "diva":
        #path to plugin
        plugin_path = "diva.vst"

        #create renderman engine with plugin loaded
        engine = dd.RenderEngine(SAMPLING_RATE, 512)
        engine.set_bpm(120)
        synth = engine.make_plugin_processor("Diva", plugin_path)
        engine.load_graph([(synth, [])])

        synth.load_preset("diva_base.fxb")
        
        d_lowpass_spec = []

        for attack in np.arange(0,1,0.01):

            synth.set_parameter(33,attack)

            #play new note
            synth.clear_midi()
            synth.add_midi_note(60, 255,0.25,3)
            
            engine.render(5)

            audio = engine.get_audio()
            audio = audio[0] + audio[1]

            audio = audio.transpose()

            #generate spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLING_RATE,)
            mel_spec = librosa.power_to_db(mel_spec,ref=np.max)
            mel_spec = mel_spec - np.min(mel_spec)
            mel_spec = mel_spec / np.max(mel_spec)

            d_lowpass_spec.append(mel_spec)


        np.save("d_attack_con", np.array(d_lowpass_spec))

        for i,param in enumerate(synth.get_plugin_parameters_description()):
            print(str(i) + " : "+param['name'] + " : "+ param['text']  )

    
    if s_type == "tyrell":
        #path to plugin
        plugin_path = "TyrellN6.vst"

        #create renderman engine with plugin loaded
        engine = dd.RenderEngine(SAMPLING_RATE, 512)
        engine.set_bpm(120)
        synth = engine.make_plugin_processor("tyrell", plugin_path)
        engine.load_graph([(synth, [])])

        synth.load_preset("tyrell_base.fxb")
        
        t_lowpass_spec = []

        for attack in np.arange(0,1,0.01):

            synth.set_parameter(18,attack)

            #play new note
            synth.clear_midi()
            synth.add_midi_note(60, 255,0.25,3)
            
            engine.render(5)

            audio = engine.get_audio()
            audio = audio[0] + audio[1]

            audio = audio.transpose()

            #generate spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLING_RATE,)
            mel_spec = librosa.power_to_db(mel_spec,ref=np.max)
            mel_spec = mel_spec - np.min(mel_spec)
            mel_spec = mel_spec / np.max(mel_spec)

            t_lowpass_spec.append(mel_spec)


        np.save("t_attack_con", np.array(t_lowpass_spec))

        synth.set_parameter(62,0.7)

        for i,param in enumerate(synth.get_plugin_parameters_description()):
            print(str(i) + " : "+param['name'] + " : "+ param['text']  )

        

def release(s_type):
    if s_type == "serum":
        #path to plugin
        plugin_path = "serum.vst"

        #create renderman engine with plugin loaded
        engine = dd.RenderEngine(SAMPLING_RATE, 512)
        engine.set_bpm(120)
        synth = engine.make_plugin_processor("Serum", plugin_path)
        engine.load_graph([(synth, [])])

        synth.load_preset("serum_base.fxb")

        synth.set_parameter(216,1)
        
        s_lowpass_spec = []

        for release in np.arange(0,0.5,0.005):

            synth.set_parameter(39,release)

            #play new note
            synth.clear_midi()
            synth.add_midi_note(60, 255,0.25,3)
            
            engine.render(5)

            audio = engine.get_audio()
            audio = audio[0] + audio[1]

            audio = audio.transpose()

            #generate spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLING_RATE,)
            mel_spec = librosa.power_to_db(mel_spec,ref=np.max)
            mel_spec = mel_spec - np.min(mel_spec)
            mel_spec = mel_spec / np.max(mel_spec)

            s_lowpass_spec.append(mel_spec)


        np.save("s_release_con", np.array(s_lowpass_spec))

        for i,param in enumerate(synth.get_plugin_parameters_description()):
            print(str(i) + " : "+param['name'] + " : "+ param['text']  )

    if s_type == "diva":
        #path to plugin
        plugin_path = "diva.vst"

        #create renderman engine with plugin loaded
        engine = dd.RenderEngine(SAMPLING_RATE, 512)
        engine.set_bpm(120)
        synth = engine.make_plugin_processor("Diva", plugin_path)
        engine.load_graph([(synth, [])])

        synth.load_preset("diva_base.fxb")
        
        d_lowpass_spec = []
        synth.set_parameter(1,0)

        for release in np.arange(0,0.5,0.005):

            synth.set_parameter(36,release)

            #play new note
            synth.clear_midi()
            synth.add_midi_note(60, 255,0.25,3)
            
            engine.render(5)

            audio = engine.get_audio()
            audio = audio[0] + audio[1]

            audio = audio.transpose()

            #generate spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLING_RATE,)
            mel_spec = librosa.power_to_db(mel_spec,ref=np.max)
            mel_spec = mel_spec - np.min(mel_spec)
            mel_spec = mel_spec / np.max(mel_spec)

            d_lowpass_spec.append(mel_spec)


        np.save("d_release_con", np.array(d_lowpass_spec))

        for i,param in enumerate(synth.get_plugin_parameters_description()):
            print(str(i) + " : "+param['name'] + " : "+ param['text']  )

    if s_type == "tyrell":
        #path to plugin
        plugin_path = "TyrellN6.vst"

        #create renderman engine with plugin loaded
        engine = dd.RenderEngine(SAMPLING_RATE, 512)
        engine.set_bpm(120)
        synth = engine.make_plugin_processor("tyrell", plugin_path)
        engine.load_graph([(synth, [])])

        synth.load_preset("tyrell_base.fxb")
        
        t_lowpass_spec = []

        for release in np.arange(0,0.5,0.005):

            synth.set_parameter(22,release)

            #play new note
            synth.clear_midi()
            synth.add_midi_note(60, 255,0.25,3)
            
            engine.render(5)

            audio = engine.get_audio()
            audio = audio[0] + audio[1]

            audio = audio.transpose()

            #generate spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLING_RATE,)
            mel_spec = librosa.power_to_db(mel_spec,ref=np.max)
            mel_spec = mel_spec - np.min(mel_spec)
            mel_spec = mel_spec / np.max(mel_spec)

            t_lowpass_spec.append(mel_spec)


        np.save("t_release_con", np.array(t_lowpass_spec))

        synth.set_parameter(62,0.7)

        for i,param in enumerate(synth.get_plugin_parameters_description()):
            print(str(i) + " : "+param['name'] + " : "+ param['text']  )
    

if __name__ == "__main__":
    if "low_pass" in sys.argv:
        low_pass("serum")
        low_pass("diva")
        low_pass("tyrell")

    if "high_pass" in sys.argv:
        high_pass("serum")
        high_pass("diva")
        high_pass("tyrell")

    if "attack" in sys.argv:
        attack("serum")
        attack("diva")
        attack("tyrell")

    if "release" in sys.argv:
        # release("serum")
        # release("diva")
        release("tyrell")
