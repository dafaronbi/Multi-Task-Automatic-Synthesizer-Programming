#import needed modules
import numpy as np
import ds
import os
import tensorflow as tf
from tensorflow.keras import losses
import model
import sys


def main():

    #load in data
    spec_data = np.load("/vast/df2322/asp_data/all_data_mels.npy",allow_pickle=True)
    serum_params = np.load("/vast/df2322/asp_data/all_data_serum_params.npy",allow_pickle=True)
    serum_masks = np.load("/vast/df2322/asp_data/all_data_serum_masks.npy",allow_pickle=True)
    diva_params = np.load("/vast/df2322/asp_data/all_data_diva_params.npy",allow_pickle=True)
    diva_masks = np.load("/vast/df2322/asp_data/all_data_diva_masks.npy",allow_pickle=True)
    tyrell_params = np.load("/vast/df2322/asp_data/all_data_tyrell_params.npy",allow_pickle=True)
    tyrell_masks = np.load("/vast/df2322/asp_data/all_data_tyrell_masks.npy",allow_pickle=True)

    m_size = len(spec_data)

    #create splits for training validation and test data
    all_data_indices = np.random.choice(m_size,m_size,replace=False)
    train_indices = all_data_indices[:m_size - m_size//5]
    valid_indices = all_data_indices[m_size - m_size//5: m_size - m_size//10]
    test_indices = all_data_indices[m_size - m_size//10:]

    train_spec_data = spec_data[train_indices]
    train_serum_params = serum_params[train_indices]
    train_serum_masks = serum_masks[train_indices]
    train_diva_params = diva_params[train_indices]
    train_diva_masks = diva_masks[train_indices]
    train_tyrell_params = tyrell_params[train_indices]
    train_tyrell_masks = tyrell_masks[train_indices]

    valid_spec_data = spec_data[valid_indices]
    valid_serum_params = serum_params[valid_indices]
    valid_serum_masks = serum_masks[valid_indices]
    valid_diva_params = diva_params[valid_indices]
    valid_diva_masks = diva_masks[valid_indices]
    valid_tyrell_params = tyrell_params[valid_indices]
    valid_tyrell_masks = tyrell_masks[valid_indices]

    test_spec_data = spec_data[test_indices]
    test_serum_params = serum_params[test_indices]
    test_serum_masks = serum_masks[test_indices]
    test_diva_params = diva_params[test_indices]
    test_diva_masks = diva_masks[test_indices]
    test_tyrell_params = tyrell_params[test_indices]
    test_tyrell_masks = tyrell_masks[test_indices]
    


    print(spec_data.shape)
    print(serum_params.shape)
    print(serum_masks.shape)
    print(diva_params.shape)
    print(diva_masks.shape)
    print(tyrell_params.shape)
    print(tyrell_masks.shape)


    
    print(m_size)
    #parameter input for dynamic filters
    v_dims = 4

    #batch_size
    batch_size = 32

    #number of batches in one epoch
    batches_epoch = m_size // batch_size

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
    if not os.path.isdir("saved_models/vst_multi"):
        os.makedirs("saved_models/vst_multi")

    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = "saved_models/vst_multi/cp-{epoch:04d}.ckpt"

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
    m = model.vae_multi(64, i_dim, serum_params.shape[-1], diva_params.shape[-1], tyrell_params.shape[-1], model.optimizer, warmup_it)

    #view summary of model
    m.summary()

    #compile model
    m.compile(optimizer=model.optimizer, loss=losses.MeanSquaredError())

    #update learning rate
    m.optimizer.lr.assign(1e-4)

    #train model
    m.fit([traiin_spec_data, train_serum_masks, train_diva_masks,train_tyrell_masks],[train_spec_data, train_serum_params, train_diva_params, train_tyrell_params], epochs=epochs, batch_size=batch_size, callbacks=[cp_callback])

    #print evaluation on test set
    loss, loss1,loss2,loss3,loss4 = m.evaluate([test_spec_data, test_serum_masks,test_diva_masks,test_tyrell_masks],[test_spec_data, test_serum_params,test_diva_params, test_tyrell_params],2)
    print("model loss = " + str(loss) + "\n model spectrogram loss = "+ str(loss1) + "\n model synth_param loss = "+ str(loss2))

if __name__ == "__main__":
    main()