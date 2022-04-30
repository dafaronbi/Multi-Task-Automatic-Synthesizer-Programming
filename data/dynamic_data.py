import numpy as np
import pickle

if __name__ == "__main__":

    batch_size = 32
    print("Loading data...")
    with open('/vast/df2322/asp_data/all_data_param_together.pkl', 'rb') as handle:
        params = pickle.load(handle)

    with open('/vast/df2322/asp_data/all_data_param_kernels.pkl', 'rb') as handle:
        kernels = pickle.load(handle)

    mels = np.load("/vast/df2322/asp_data/all_data_mels.npy", allow_pickle=True)
    print("DONE!!!")

    all_mels = []
    all_params = []
    all_kernels = []

    batch_mel = []
    batch_param = []
    batch_kernels = []

    past_size = len(params[0])
    for i in range(len(params)):
        #divide data by branch
        if i % batch_size == 0 and i != 0:
            #add last batch
            all_mels.append(batch_mel)
            all_params.append(batch_param)
            all_kernels.append(batch_kernels)

            print(batch_mel)
            print(batch_param)
            print(len(batch_mel))

            #start new batch
            batch_mel = [mels[i]]
            batch_param = [params[i]]
            batch_kernels = [kernels[i]]
            past_size = len(params[i])
            continue

        #make sure all batches have the same synthesizer
        if past_size != len(params[i]):
            batch_mel = [mels[i]]
            batch_param = [params[i]]
            batch_kernels = [kernels[i]]
            past_size = len(params[i])
            print("lenght error")
            print(len(batch_mel))
            continue
        
        batch_mel.append(mels[i])
        batch_param.append(params[i])
        batch_kernels.append(kernels[i])
        past_size = len(params[i])

