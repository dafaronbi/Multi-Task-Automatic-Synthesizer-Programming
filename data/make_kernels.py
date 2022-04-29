import numpy as np

s_mask = np.load("/vast/df2322/asp_data/all_data_serum_masks.npy", allow_pickle=True)
d_mask = np.load("/vast/df2322/asp_data/all_data_diva_masks.npy", allow_pickle=True)
t_mask = np.load("/vast/df2322/asp_data/all_data_tyrell_masks.npy", allow_pickle=True)


serum_kernel = np.load("serum_kernel.npy", allow_pickle=True)
diva_kernel = np.load("diva_kernel.npy", allow_pickle=True)
tyrell_kernel = np.load("tyrell_kernel.npy", allow_pickle=True)

length = len(s_mask)

param_kernel = []

for i in range(length):

    if s_mask[i][0] == 1:
        param_kernel.append(serum_kernel)
    
    if d_mask[i][0] == 1:
        param_kernel.append(diva_kernel)

    if t_mask[i][0] == 1:
        param_kernel.append(tyrell_kernel)

param_kernel = np.array(param_kernel)

print(param_kernel.shape)

np.save("param_kernels", param_kernel)
