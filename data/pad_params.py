import numpy as np

serum_p = np.load("serum_params.npy",allow_pickle=True)
diva_p = np.load("diva_params.npy",allow_pickle=True)
tyrell_p = np.load("tyrell_params.npy",allow_pickle=True)

print(serum_p.shape)
print(diva_p.shape)
print(tyrell_p.shape)

max_size = 315

padded_serum = []
reg_size = serum_p.shape[-1]
for p in serum_p:
    padded_serum.append(np.pad(p,(0,315-reg_size),'constant',constant_values=(0,0)))

padded_serum = np.array(padded_serum)
print(padded_serum.shape)

padded_diva = []
reg_size = diva_p.shape[-1]
for p in diva_p:
    padded_diva.append(np.pad(p,(0,315-reg_size),'constant',constant_values=(0,0)))

padded_diva = np.array(padded_diva)
print(padded_diva.shape)

padded_tyrell = []
reg_size = tyrell_p.shape[-1]
for p in tyrell_p:
    padded_tyrell.append(np.pad(p,(0,315-reg_size),'constant',constant_values=(0,0)))

padded_tyrell = np.array(padded_tyrell)
print(padded_tyrell.shape)


np.save("padded_serum.npy", padded_serum)
np.save("padded_diva.npy", padded_diva)
np.save("padded_tyrell.npy", padded_tyrell)