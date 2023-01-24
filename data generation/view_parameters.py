import numpy as np
from scipy.io import wavfile
import os
import librosa
import matplotlib.pyplot as plt
import librosa.display
import dawdreamer as dd

#folder where diva data is stored
data_folder = "Serum Presets"

serum_oh = [0,0,0,9,25,0,16,0,0,0,0,0,0,0,0,0,9,25,0,16,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,3,0,0,0,16,0,0,3,0,2,
0,0,0,0,0,2,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,2,2,0,0,3,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,
0,0,0,8,2,0,0,2,2,2,2,2,2,2,2,2,2,0,0,0,0,0,0,6,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

diva_oh =[0,2,2,0,8,6,5,2,0,0,0,0,0,2,0,0,3,0,4,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,
0,2,2,2,0,0,0,0,0,0,3,0,2,2,2,0,0,4,8,0,0,0,0,0,0,0,0,4,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,5,0,0,0,0,0,0,0,0,0,2,0,0,0,0,4,6,6,23,0,23,0,23,0,23,0,2,2,2,2,2,2,2,2,2,3,2,0,0,2,
2,2,2,2,2,4,4,4,2,0,23,0,23,0,4,0,0,2,0,23,0,0,5,0,0,23,0,23,0,0,0,2,2,2,4,0,23,0,23,0,
23,0,0,0,2,23,0,23,0,0,0,0,0,5,3,0,0,0,2,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
3,0,0,0,0,0,0,0,0,4,5,3,0,0,0,2,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,
0,0,0,0,0,4,0,4,0,6,4,0,0,2,0,0,0,0,0,0,0,0,2,2,0,0,0,0,7,7,2]

tyrell_oh =[0,18,0,18,0,18,0,18,0,8,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,4,0,
0,0,0,0,18,0,0,0,0,18,0,0,18,0,0,0,0,6,3,0,0,0,0,0,0,0,2,3,18,0,18,0,0,0,0,18,0,0,18,3,
0,0,0,4,8,0,0,0,0,0,4,8,0,0,0,0,0]

serum_labels = ["vol","vol","osc","osc","osc","osc","osc","osc","osc","osc","osc","osc","osc","osc","vol","osc","osc","osc",
"osc","osc","osc","osc","osc","osc","osc","osc","osc","vol","osc","osc","osc","osc","osc","vol","osc","env","env","env","env",
"env","fil","fil","fil","fil","fil","fil","fil","fil","fil","fil","fil","env","env","env","env","env","env","env","env","env","env",
"lfo", "lfo", "lfo", "lfo","pitch", "pitch","mod","mod","mod","mod","env","env","env","env","env","env","env","env","env","pitch","fx",
"fx","fx","fx","fx","fx","fx","fil","fil","fil","fil","fil","fil","fil","fil","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx",
"fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx",
"fx","fx","fx","fx","fx","fx","fil","fil","fil","fil","fil","fil","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx",
"fil","fil","fx","osc","osc","pitch","pitch","osc","osc","osc","osc","osc","osc","osc","osc","osc","osc","osc","osc","mod","mod","mod",
"mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod",
"mod","mod","mod","mod","mod","mod","osc","osc","osc","osc","fil","mod","mod","mod","mod","mod","vol","lfo","lfo","lfo","lfo","pitch",
"mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod",
"mod","mod","mod","mod","mod","mod","mod","mod","mod","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","fil","fx","fx","fx","fx","lfo","lfo",
"lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","fx","fx","fx","fx","fx","fx","fx","fx","fil","fx","lfo",
"lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo"]

diva_labels = ['vol',"fx","fx","fx","pitch","pitch","pitch","pitch","pitch","pitch","pitch","pitch","pitch","pitch","pitch","pitch","pitch",
"pitch","fx","fx","pitch","fil","pitch","osc","env","mod","mod","mod","mod","mod","mod","mod","mod","env","env","env","env","env","env","env",
"env","env","env","env","env","env","env","env","env","env","env","env","env","env","env","lfo", "lfo", "lfo", "lfo", "lfo", "lfo", "lfo", "lfo",
"lfo", "lfo", "lfo", "lfo", "lfo", "lfo", "lfo", "lfo", "lfo", "lfo", "lfo", "lfo","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod",
"osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc",
"osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc","osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc",
"osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc","fil","fil", "fil","fil","fil","fil","fil","fil","fil","fil",
"fil", "fil", "fil", "fil", "fil", "fil", "fil", "fil", "fil", "fil", "fil", "fil", "fil", "fil", "fil", "fil", "fil", "vol", "vol","vol", "vol", "vol",
"vol","vol","vol","vol","fx","fx", "fx", "fx", "fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx",
"fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx",
"fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx",
"fx","osc","osc","osc","fil","fil","fil","fil","fx","lfo","lfo","fx","fx","fx","fx","osc","osc","osc"]

tyrell_labels = ["vol", "mod","mod","mod","mod","mod","mod","mod","mod","fx","fx","fx","fx","fx","fx","fx","fx","fx",
"env", "env", "env", "env", "env", "env", "env", "env","env","env","env","env","env","env","env","env","lfo","lfo","lfo","lfo",
"osc","osc","osc","osc","osc","osc","osc","osc","osc","osc","osc","osc","osc","osc","osc","osc","vol","vol","vol","vol","osc","osc"
,"fil","fil","fil","fil","fil","fil","fil","fil","fil","fil","mod","mod","mod","mod","fx","fx","fx","fx","lfo","lfo","lfo","lfo","lfo"
,"lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo"]

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

if __name__ == "__main__":
    #path to plugin
    plugin_path = "tyrellN6.vst"

    labels = tyrell_labels
    oh = tyrell_oh

    SAMPLING_RATE = 44100

    #create renderman engine with plugin loaded
    engine = dd.RenderEngine(SAMPLING_RATE, 512)
    engine.set_bpm(120)
    synth = engine.make_plugin_processor("Serum", plugin_path)
    engine.load_graph([(synth, [])])
    count = 0

    #get list of fxp files
    # fxp_files = getListOfFiles(data_folder)

    prev = 0

    # for fxp_file in fxp_files:

    #     synth.load_preset(fxp_file)
    #     print(float(synth.get_plugin_parameters_description()[299]['text']))
    #     if float(synth.get_plugin_parameters_description()[299]['text']) != prev:
    #         prev = float(synth.get_plugin_parameters_description()[299]['text'])
    #         print("CHANGED!!!!")
    for i,param in enumerate(synth.get_plugin_parameters_description()):
        print(str(i+1) + " : "+param['name'] + " : "+ param['text'] + " : " + labels[i] + ":" + str(oh[i]) )
    all = []
    all.extend(serum_labels)
    all.extend(diva_labels)
    all.extend(tyrell_labels)
    

    print(set(all))
    print(len(set(all)))
