import numpy as np
import os

#load every file
for file in os.listdir("../raw"):
    data = np.load("../raw/"+ file,allow_pickle=True)
    # print(data["audio"])
    # print(data["chars"])
    print(data["param"])