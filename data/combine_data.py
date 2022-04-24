import numpy as np

if __name__ == "__main__":
    serum_m = np.load("serum_mels.npy",allow_pickle=True)
    diva_m = np.load("diva_mels.npy",allow_pickle=True)
    tyrell_m = np.load("tyrell_mels.npy",allow_pickle=True)

    serum_p = np.load("serum_params_oh.npy",allow_pickle=True)
    diva_p = np.load("diva_params_oh.npy",allow_pickle=True)
    tyrell_p = np.load("tyrell_params_oh.npy",allow_pickle=True)

    final_out = []

    for i,spec in enumerate(serum_m):
        joined = [spec, np.concatenate((serum_p[i],np.zeros(diva_p.shape[-1]),np.zeros(tyrell_p.shape[-1])))]
        final_out.append(joined)

    for i,spec in enumerate(diva_m):
        joined = [spec, np.concatenate((np.zeros(serum_p.shape[-1]),diva_p[i],np.zeros(tyrell_p.shape[-1])))]
        final_out.append(joined)

    for i,spec in enumerate(tyrell_m):
        joined = [spec, np.concatenate((np.zeros(serum_p.shape[-1]),np.zeros(diva_p.shape[-1]),tyrell_p[i]))]
        final_out.append(joined)

    np.save("all_data", np.array(final_out))