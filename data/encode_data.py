import one_hot
import numpy as np


if __name__ == "__main__":
    serum_p = np.load("serum_params.npy",allow_pickle=True)
    diva_p = np.load("diva_params.npy",allow_pickle=True)
    tyrell_p = np.load("tyrell_params.npy",allow_pickle=True)

    serum_one_hot = []

    for p in serum_p:
        encoded = one_hot.encode(p, one_hot.serum_oh)
        serum_one_hot.append(encoded)

    np.save("serum_params_oh",np.array(serum_one_hot))

    diva_one_hot = []

    for p in diva_p:
        encoded = one_hot.encode(p, one_hot.diva_oh)
        diva_one_hot.append(encoded)

    np.save("diva_params_oh",np.array(diva_one_hot))

    tyrell_one_hot = []

    for p in tyrell_p:
        encoded = one_hot.encode(p, one_hot.tyrell_oh)
        tyrell_one_hot.append(encoded)

    np.save("tyrell_params_oh",np.array(tyrell_one_hot))