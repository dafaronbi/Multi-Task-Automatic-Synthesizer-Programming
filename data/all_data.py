import tensorflow.keras as tfk
import numpy as np

class SynthDataGenerator(tfk.utils.Sequence):
    'Generates data for Keras'
    '''
    nbatches (int): total number of batches in dataset
    spectrograms (list of np.arrays): list of numpy arrays with the spectrograms of each dataset
        [ [nbatches, [batch_size, nmels, time]], [...], [...] ]
    synth_params (list of np.arrays): list of numpy arrays with the synthesizer parameters for each spectrogram
        [ [nbatches, [batch_size, nparams], [...], [...] ]
    synth_feats: list of numpy arrays with the features that describe each synthesizer    
        [ [nbatches, [1, nparams, nfeats]], [...], [...] ]
    '''

    def __init__(self, nbatches, spectrograms, synth_params, synth_feats):
        'Initialization'
        self.nbatches_per_epoch = nbatches
        self.spectrograms = spectrograms
        self.synth_params = synth_params
        self.synth_feats = synth_feats
        self.nbatches_per_synth = [len(specs) for specs in self.spectrograms]
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __get_synth_number_from_index(self, index):
        add = 0
        index_modulo = index
        for isynth, nbps in enumerate(self.nbatches_per_synth):          
          add += nbps
          if index < add:
            return isynth, index_modulus
          else:
            index_modulus = index - (add - nbps)
            
    def __getitem__(self, index):
        'Generate one batch of data'   
        synth_number, modulus = self.__get_synth_number_from_index(index)

        # Generate data
        X, y = self.__data_generation(synth_number, modulus)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nbatches_per_epoch)
        
    def __data_generation(self, index, modulus):
        'Generates data containing batch_size samples' # X : (n_samples, ndim)      
        spec = self.spectrograms[index][modulus]
        synth_params = self.synth_params[index][modulus]
        synth_feats = self.synth_feats[index][modulus]
        
        return (spec, synth_feats), (spec, synth_params)