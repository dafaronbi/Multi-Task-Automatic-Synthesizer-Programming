import numpy as np
import pickle

s_mask = np.load("/vast/df2322/asp_data/all_data_serum_masks.npy", allow_pickle=True)
d_mask = np.load("/vast/df2322/asp_data/all_data_diva_masks.npy", allow_pickle=True)
t_mask = np.load("/vast/df2322/asp_data/all_data_tyrell_masks.npy", allow_pickle=True)
s_param = np.load("/vast/df2322/asp_data/all_data_serum_params.npy", allow_pickle=True)
d_param = np.load("/vast/df2322/asp_data/all_data_diva_params.npy", allow_pickle=True)
t_param = np.load("/vast/df2322/asp_data/all_data_tyrell_params.npy", allow_pickle=True)


serum_kernel = np.load("serum_kernel.npy", allow_pickle=True)
diva_kernel = np.load("diva_kernel.npy", allow_pickle=True)
tyrell_kernel = np.load("tyrell_kernel.npy", allow_pickle=True)

length = len(s_mask)
print(l)

param_kernel = []
synth_parameters = []

for i in range(length):

    if s_mask[i][0] == 1:
        param_kernel.append(serum_kernel)
        synth_parmeters.append(s_param[i])

    
    if d_mask[i][0] == 1:
        param_kernel.append(diva_kernel)
        synth_parmeters.append(d_param[i])

    if t_mask[i][0] == 1:
        param_kernel.append(tyrell_kernel)
        synth_parmeters.append(t_param[i])


with open("/vast/df2322/asp_data/all_data_param_kernels.pkl", 'wb') as handle:
    pickle.dump(param_kernel, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("/vast/df2322/asp_data/all_data_param_together.pkl", 'wb') as handle:
    pickle.dump(synth_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

