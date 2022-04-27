import numpy as np

if __name__ == "__main__":
    serum_m = np.load("serum_mels.npy",allow_pickle=True)
    diva_m = np.load("diva_mels.npy",allow_pickle=True)
    tyrell_m = np.load("tyrell_mels.npy",allow_pickle=True)

    serum_p = np.load("serum_params_oh.npy",allow_pickle=True)
    diva_p = np.load("diva_params_oh.npy",allow_pickle=True)
    tyrell_p = np.load("tyrell_params_oh.npy",allow_pickle=True)

    final_out_spec = []
    serum_out_params = []
    serum_mask = []
    diva_out_params = []
    diva_mask = []
    tyrell_out_params = []
    tyrell_mask = []

    for i,spec in enumerate(serum_m):
        final_out_spec.append(spec)
        serum_out_params.append(serum_p[i])
        serum_mask.append(np.ones_like(serum_p[i]))
        diva_out_params.append(np.zeros(diva_p.shape[-1]))
        diva_mask.append(np.zeros(diva_p.shape[-1]))
        tyrell_out_params.append(np.zeros(tyrell_p.shape[-1]))
        tyrell_mask.append(np.zeros(tyrell_p.shape[-1]))

    for i,spec in enumerate(diva_m):
        final_out_spec.append(spec)
        serum_out_params.append(np.zeros(serum_p.shape[-1]))
        serum_mask.append(np.zeros(serum_p.shape[-1]))
        diva_out_params.append(diva_p[i])
        diva_mask.append(np.ones_like(diva_p[i]))
        tyrell_out_params.append(np.zeros(tyrell_p.shape[-1]))
        tyrell_mask.append(np.zeros(tyrell_p.shape[-1]))

    for i,spec in enumerate(tyrell_m):
        final_out_spec.append(spec)
        serum_out_params.append(np.zeros(serum_p.shape[-1]))
        serum_mask.append(np.zeros(serum_p.shape[-1]))
        diva_out_params.append(np.zeros(diva_p.shape[-1]))
        diva_mask.append(np.zeros(diva_p.shape[-1]))
        tyrell_out_params.append(tyrell_p[i])
        tyrell_mask.append(np.ones_like(tyrell_p[i]))

    np.save("all_data_mels", np.array(final_out_spec))
    np.save("all_data_serum_params", np.array(serum_out_params))
    np.save("all_data_serum_masks", np.array(serum_mask))
    np.save("all_data_diva_params", np.array(diva_out_params))
    np.save("all_data_diva_masks", np.array(diva_mask))
    np.save("all_data_tyrell_params", np.array(tyrell_out_params))
    np.save("all_data_tyrell_masks", np.array(tyrell_mask))