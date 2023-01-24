import numpy as np
import os
import dawdreamer as dd
from scipy.io import wavfile

#folder where diva data is stored
data_folder = "tyrellN6 presets"

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

#get list of fxb files
fxb_files = getListOfFiles(data_folder)

#array to store all parameters
all_data = list()
preset_name = list()


 #create renderman engine with plugin loaded
engine = dd.RenderEngine(44100, 512)
engine.set_bpm(120)
synth = engine.make_plugin_processor("Tyrell", "TyrellN6.vst")
count = 0
for fxb_file in fxb_files:
    print(count)
    count += 1
    size = synth.get_plugin_parameter_size()

    synth.load_preset(fxb_file)
    preset_name.append(fxb_file.split('/')[-1])

    #array containing parameters of current preset
    params = list()
    for i in range(size):
        params.append(synth.get_parameter(i))
    
    #make into numpy array
    params = np.array(params)
    
    all_data.append(params)

np.save("tyrell_params",np.array(all_data))
np.save("tyrell_preset_name", np.array(preset_name))
