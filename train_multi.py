#import needed modules
import numpy as np
import ds
import os
import tensorflow as tf
from tensorflow.keras import losses
import model
import sys


def main():

    spec_data = np.load("/vast/df2322/asp_data/all_data_mels.npy",allow_pickle=True)
    serum_params = np.load("/vast/df2322/asp_data/all_data_serum_params.npy",allow_pickle=True)
    serum_masks = np.load("/vast/df2322/asp_data/all_data_serum_masks.npy",allow_pickle=True)
    diva_params = np.load("/vast/df2322/asp_data/all_data_diva_params.npy",allow_pickle=True)
    diva_masks = np.load("/vast/df2322/asp_data/all_data_diva_masks.npy",allow_pickle=True)
    tyrell_params = np.load("/vast/df2322/asp_data/all_data_tyrell_params.npy",allow_pickle=True)
    tyrell_masks = np.load("/vast/df2322/asp_data/all_data_tyrell_masks.npy",allow_pickle=True)


    m_size = len(spec_data)
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

    #dictionary of losses
    get_loss = {"ae": losses.MeanSquaredError(),"ae2": losses.MeanSquaredError(),"ae3" : losses.MeanSquaredError(),"vae": losses.MeanSquaredError(),"dynamic_vae": losses.MeanSquaredError(),"vae_flow": losses.MeanSquaredError()}

    #create model
    m = model.vae_multi(64, i_dim, serum_params.shape[-1], diva_params.shape[-1], tyrell_params.shape[-1], m.optimizer, warmup_it)

    #view summary of model
    m.summary()

    #compile model
    m.compile(optimizer=model.optimizer, loss=get_loss[sys.argv[1]])

    #update learning rate
    m.optimizer.lr.assign(1e-4)

    #train model
    m.fit([spec_data, serum_masks,diva_masks,tyrell_masks],[serum_params,diva_params, tyrell_params], epochs=epochs, batch_size=batch_size, callbacks=[cp_callback])

    #print evaluation on test set
    loss, loss1,loss2,loss3,loss4 = m.evaluate([spec_data, serum_masks,diva_masks,tyrell_masks],[serum_params,diva_params, tyrell_params],2)
    print("model loss = " + str(loss) + "\n model spectrogram loss = "+ str(loss1) + "\n model synth_param loss = "+ str(loss2))

if __name__ == "__main__":
    main()