import one_hot
import numpy as np


if __name__ == "__main__":
    data = np.load("/vast/df2322/asp_data/fixed_data/original/asp_data.npy",allow_pickle=True)

    params = data[:,1]
    synths = data[:,4]

    params_one_hot = []

    for i,p in enumerate(params):
        if synths[i] == "serum":
            encoded = one_hot.encode(p, one_hot.serum_oh)
            params_one_hot.append(encoded)

        if synths[i] == "diva":
            encoded = one_hot.encode(p, one_hot.diva_oh)
            params_one_hot.append(encoded)

        if synths[i] == "serum":
            encoded = one_hot.encode(p, one_hot.tyrell_oh)
            params_one_hot.append(encoded)

    data[:,1] = np.array(params_one_hot,dtype=object)

    np.save("/vast/df2322/asp_data/fixed_data/original/asp_data_oh.npy",data)