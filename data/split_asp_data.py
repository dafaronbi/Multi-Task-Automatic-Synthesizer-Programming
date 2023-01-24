import numpy as np
import sys

synth = sys.argv[1]

asp_data = np.load("asp_data_" + synth +".npy",allow_pickle=True)

mels = list()
params = list()

for item in asp_data:
    mels.append(item[0])
    params.append(item[1])

mels = np.array(mels)
params = np.array(params)

print(mels.shape)
print(params.shape)
np.save(synth +"_mels.npy", mels)
np.save(synth +"_params.npy", params)