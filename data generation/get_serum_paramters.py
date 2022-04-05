import numpy as np
import os
import dawdreamer as dd

#folder where diva data is stored
data_folder = "Serum Presets"

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

#get list of fxp files
fxp_files = getListOfFiles(data_folder)

#array to store all parameters
all_data = list()

 #create renderman engine with plugin loaded
engine = dd.RenderEngine(44100, 512)
engine.set_bpm(120)
synth = engine.make_plugin_processor("Serum", "Serum.vst")
count = 0
for fxp_file in fxp_files:
    print(count)
    count += 1
    size = synth.get_plugin_parameter_size()

    synth.load_preset(fxp_file)

    #array containing parameters of current preset
    params = list()
    for i in range(size):
        params.append(synth.get_parameter(i))
    
    #make into numpy array
    params = np.array(params)
    
    all_data.append(params)

np.save("serum_params",np.array(all_data))