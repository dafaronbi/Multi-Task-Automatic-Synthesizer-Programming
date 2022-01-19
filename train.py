#import needed modules
import numpy as np
import ds
import tensorflow as tf
from tensorflow.keras import losses
import model

#load data
train_data = ds.melParamData("train","data")
test_data = ds.melParamData("test","data")
validation_data = ds.melParamData("validation","data")

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "saved_models/cp-{epoch:04d}.ckpt"

#epoch size
epochs=500

#batch_size
batch_size = 8

#number of batches in one epoch
batches_epoch = ds.melParamData("train","data").get_mels().shape[0] // batch_size

#save freq is every 10 epochs
save_freq = batches_epoch*10

# Create a callback that saves the model's weights every 50 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=save_freq)

#create auto encoder model
autoencoder = model.vae(64,train_data.get_mels()[...,np.newaxis].shape,train_data.get_params().shape)

#view summary of model
autoencoder.summary()

#compile model
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

#update learning rate
autoencoder.optimizer.lr.assign(1e-3)

#train model
autoencoder.fit(train_data.get_mels()[...,np.newaxis],[train_data.get_mels(),train_data.get_params()], epochs=epochs, batch_size=batch_size, callbacks=[cp_callback])

#print evaluation on test set
loss, loss1,loss2 = autoencoder.evaluate(test_data.get_mels(),[test_data.get_mels(),test_data.get_params()],2)
print("model loss = " + str(loss) + "\n model spectrogram loss = "+ str(loss1) + "\n model synth_param loss = "+ str(loss2))