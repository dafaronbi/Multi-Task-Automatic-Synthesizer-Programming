#import needed modules
import numpy as np
import ds
import os
import tensorflow as tf
from tensorflow.keras import losses
import model
import sys
import metrics


def main():

    train_spec_data = np.load("/vast/df2322/asp_data/fixed_data/expanded/train_mels.npy",allow_pickle=True)
    train_serum_params = np.load("/vast/df2322/asp_data/fixed_data/expanded/train_serum_params.npy",allow_pickle=True)
    train_serum_masks = np.load("/vast/df2322/asp_data/fixed_data/expanded/train_serum_mask.npy",allow_pickle=True)
    train_diva_params = np.load("/vast/df2322/asp_data/fixed_data/expanded/train_diva_params.npy",allow_pickle=True)
    train_diva_masks = np.load("/vast/df2322/asp_data/fixed_data/expanded/train_diva_mask.npy",allow_pickle=True)
    train_tyrell_params = np.load("/vast/df2322/asp_data/fixed_data/expanded/train_tyrell_params.npy",allow_pickle=True)
    train_tyrell_masks = np.load("/vast/df2322/asp_data/fixed_data/expanded/train_tyrell_mask.npy",allow_pickle=True)
    train_synth  = np.load("/vast/df2322/asp_data/fixed_data/expanded/train_synth.npy",allow_pickle=True)

    valid_spec_data = np.load("/vast/df2322/asp_data/fixed_data/expanded/valid_mels.npy",allow_pickle=True)
    valid_serum_params = np.load("/vast/df2322/asp_data/fixed_data/expanded/valid_serum_params.npy",allow_pickle=True)
    valid_serum_masks = np.load("/vast/df2322/asp_data/fixed_data/expanded/valid_serum_mask.npy",allow_pickle=True)
    valid_diva_params = np.load("/vast/df2322/asp_data/fixed_data/expanded/valid_diva_params.npy",allow_pickle=True)
    valid_diva_masks = np.load("/vast/df2322/asp_data/fixed_data/expanded/valid_diva_mask.npy",allow_pickle=True)
    valid_tyrell_params = np.load("/vast/df2322/asp_data/fixed_data/expanded/valid_tyrell_params.npy",allow_pickle=True)
    valid_tyrell_masks = np.load("/vast/df2322/asp_data/fixed_data/expanded/valid_tyrell_mask.npy",allow_pickle=True)
    valid_synth  = np.load("/vast/df2322/asp_data/fixed_data/expanded/valid_synth.npy",allow_pickle=True)

    test_spec_data = np.load("/vast/df2322/asp_data/fixed_data/expanded/test_mels.npy",allow_pickle=True)
    test_serum_params = np.load("/vast/df2322/asp_data/fixed_data/expanded/test_serum_params.npy",allow_pickle=True)
    test_serum_masks = np.load("/vast/df2322/asp_data/fixed_data/expanded/test_serum_mask.npy",allow_pickle=True)
    test_diva_params = np.load("/vast/df2322/asp_data/fixed_data/expanded/test_diva_params.npy",allow_pickle=True)
    test_diva_masks = np.load("/vast/df2322/asp_data/fixed_data/expanded/test_diva_mask.npy",allow_pickle=True)
    test_tyrell_params = np.load("/vast/df2322/asp_data/fixed_data/expanded/test_tyrell_params.npy",allow_pickle=True)
    test_tyrell_masks = np.load("/vast/df2322/asp_data/fixed_data/expanded/test_tyrell_mask.npy",allow_pickle=True)
    test_synth  = np.load("/vast/df2322/asp_data/fixed_data/expanded/test_synth.npy",allow_pickle=True)

    # np.save("/vast/df2322/asp_data/multi/test_spec",test_spec_data)
    # np.save("/vast/df2322/asp_data/multi/test_serum_params",test_serum_params)
    # np.save("/vast/df2322/asp_data/multi/test_serum_masks",test_serum_masks)
    # np.save("/vast/df2322/asp_data/multi/test_diva_params",test_diva_params)
    # np.save("/vast/df2322/asp_data/multi/test_diva_masks",test_diva_masks)
    # np.save("/vast/df2322/asp_data/multi/test_tyrell_params",test_tyrell_params)
    # np.save("/vast/df2322/asp_data/multi/test_tyrell_masks",test_tyrell_masks)
    
    train_index = np.where(train_synth ==  "diva")
    valid_index = np.where(valid_synth ==  "diva")
    test_index = np.where(test_synth ==  "diva")

    train_spec_data = train_spec_data[train_index]
    train_params = train_diva_params[train_index]
    valid_spec_data = valid_spec_data[valid_index]
    valid_params = valid_diva_params[valid_index]
    test_spec_data = test_spec_data[test_index]
    test_params = test_diva_params[test_index]

    m_size = len(train_spec_data)

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
    i_dim = (1, 128, 431, 1)

    #make directory to save model if not already made
    if not os.path.isdir("/vast/df2322/asp_data/saved_models/vst_diva"):
        os.makedirs("/vast/df2322/asp_data/saved_models/vst_diva")

    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = "/vast/df2322/asp_data/saved_models/vst_diva/cp-{epoch:04d}.ckpt"

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
    m = model.vae_diva(64, i_dim, train_serum_params.shape[-1], train_diva_params.shape[-1], train_tyrell_params.shape[-1], model.optimizer, warmup_it)

    #view summary of model
    m.summary()

    #compile model
    m.compile(optimizer=model.optimizer, loss=losses.MeanSquaredError())

    #update learning rate
    m.optimizer.lr.assign(1e-4)

    #train model
    m.fit([train_spec_data],[train_spec_data, train_params], epochs=epochs, batch_size=batch_size, callbacks=[cp_callback])

    #print evaluation on test set
    loss, loss1,loss2 = m.evaluate([test_spec_data],[test_spec_data, test_params],2)
    print("model loss = " + str(loss) + "\n model spectrogram loss = "+ str(loss1) + "\n model synth_param loss = "+ str(loss2))

if __name__ == "__main__":
    main()