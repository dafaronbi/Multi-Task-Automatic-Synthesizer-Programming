#import needed modules
import numpy as np
import ds
import os
import tensorflow as tf
from tensorflow.keras import losses
import model
import sys
from data import all_data
import random
import pickle


def main():

    #load in data
    # with open("/vast/df2322/asp_data/dynamic_mels.pkl", 'rb') as handle:
    #     mels = pickle.load(handle)
    # with open("/vast/df2322/asp_data/dynamic_params.pkl", 'rb') as handle:
    #     params = pickle.load(handle)
    # with open("/vast/df2322/asp_data/dynamic_kernels.pkl", 'rb') as handle:
    #     kernels = pickle.load(handle)

    train_spec_data = np.load("/vast/df2322/asp_data/fixed_data/expanded/train_mels.npy",allow_pickle=True)
    train_params = np.load("/vast/df2322/asp_data/fixed_data/expanded/train_params.npy",allow_pickle=True)
    train_synth = np.load("/vast/df2322/asp_data/fixed_data/expanded/train_synth.npy",allow_pickle=True)

    valid_spec_data = np.load("/vast/df2322/asp_data/fixed_data/expanded/valid_mels.npy",allow_pickle=True)
    valid_params = np.load("/vast/df2322/asp_data/fixed_data/expanded/valid_params.npy",allow_pickle=True)
    valid_synth = np.load("/vast/df2322/asp_data/fixed_data/expanded/valid_synth.npy",allow_pickle=True)

    test_spec_data = np.load("/vast/df2322/asp_data/fixed_data/expanded/test_mels.npy",allow_pickle=True)
    test_params = np.load("/vast/df2322/asp_data/fixed_data/expanded/test_params.npy",allow_pickle=True)
    test_synth = np.load("/vast/df2322/asp_data/fixed_data/expanded/test_synth.npy",allow_pickle=True)

    train_kernels = []
    valid_kernels =[]
    test_kernels =[]

    # print("here")
    # for s in train_synth:
    #     if s == "serum":
    #         train_kernels.append(np.full((1024,480),0))

    #     if s == "diva":
    #         train_kernels.append(np.full((1024,759),1))

    #     if s == "tyrell":
    #         train_kernels.append(np.full((1024,327),2))
    # print("done")
    # for s in valid_synth:
    #     if s == "serum":
    #         valid_kernels.append(np.full((1024,480),0))

    #     if s == "diva":
    #         valid_kernels.append(np.full((1024,759),1))

    #     if s == "tyrell":
    #         valid_kernels.append(np.full((1024,327),2))
    # print('done')
    # for s in test_synth:
    #     if s == "serum":
    #         test_kernels.append(np.full((1024,480),0))

    #     if s == "diva":
    #         test_kernels.append(np.full((1024,759),1))

    #     if s == "tyrell":
    #         test_kernels.append(np.full((1024,327),2))

    # print("here2")

    m_size = len(train_spec_data)
    print(m_size)
    split_size = int(m_size/12)
    print(split_size)

    # r_mels,r_params,r_kernels = zip(*random.sample(list(zip(mels,params,kernels)), m_size))

    train_mels = np.array(np.split(train_spec_data, split_size, axis=0))          #r_mels[:m_size - m_size//5]
    train_params = np.array(np.split(train_params, split_size, axis=0))                                                      #r_params[:m_size - m_size//5]
    # train_kernels = np.array(np.split(train_kernels, split_size, axis=0))                                                    #r_kernels[:m_size - m_size//5]

    m_size = len(valid_spec_data)
    print(m_size)
    split_size = int(m_size/12)
    print(split_size)

    valid_mels = np.array(np.split(valid_spec_data, split_size, axis=0)) 
    valid_params = np.array(np.split(valid_params, split_size, axis=0)) 
    # valid_kernels = np.array(np.split(valid_kernels, split_size, axis=0))

    m_size = len(test_spec_data)
    print(m_size)
    split_size = int(m_size/12)
    print(split_size)

    test_mels = np.array(np.split(test_spec_data, split_size, axis=0)) 
    test_params = np.array(np.split(test_params, split_size, axis=0)) 
    # test_kernels = np.array(np.split(test_kernels, split_size, axis=0))

    print("here3")

    m_size = len(train_mels)

    #use synth dataloader
    train_data_load = all_data.SynthDataGenerator(len(train_mels),train_mels,train_params)
    valid_data_load = all_data.SynthDataGenerator(len(valid_mels),valid_mels,valid_params)
    test_data_load = all_data.SynthDataGenerator(len(test_mels),test_mels,test_params)

    # with open("/vast/df2322/asp_data/dynamic/test_dynamic_mels.pkl", 'wb') as handle:
    #     pickle.dump(test_mels, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open("/vast/df2322/asp_data/dynamic/test_dynamic_params.pkl", 'wb') as handle:
    #     pickle.dump(test_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open("/vast/df2322/asp_data/dynamic/test_dynamic_kernels.pkl", 'wb') as handle:
    #     pickle.dump(test_kernels, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    print(m_size)

    #batch_size
    batch_size = 12

    #number of batches in one epoch
    batches_epoch = m_size 

    print(batches_epoch)

    #warmup amount
    warmup_it = 100*batches_epoch

    #list GPUs that tensor flow can use
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs:", len(physical_devices))

    #define shapes
    l_dim = 64
    i_dim = (1, 128, 431, 1)

    #make directory to save model if not already made
    if not os.path.isdir("saved_models/dynamic"):
        os.makedirs("saved_models/dynamic")

    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = "saved_models/dynamic/cp-{epoch:04d}.ckpt"

    #epoch size
    epochs= 500

    #save freq is every 100 epochs
    save_freq = batches_epoch*100

    # Create a callback that saves the model's weights every 50 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq=save_freq)

    #create model
    m = model.dynamic_vae(64, i_dim, 0, model.optimizer, warmup_it,0)

    #view summary of model
    m.summary()

    #compile model
    m.compile(optimizer=model.optimizer, loss=losses.MeanSquaredError())

    #update learning rate
    m.optimizer.lr.assign(1e-4)

    #train model
    m.fit(train_data_load, epochs=epochs, batch_size=batch_size, callbacks=[cp_callback])

    #print evaluation on test set
    loss, loss1,loss2 = m.evaluate(test_data_load,2)
    print("model loss = " + str(loss) + "\n model spectrogram loss = "+ str(loss1) + "\n model synth_param loss = "+ str(loss2))

if __name__ == "__main__":
    main()