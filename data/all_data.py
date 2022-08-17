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

    def __init__(self, nbatches, spectrograms, synth_params):
        'Initialization'
        self.nbatches_per_epoch = nbatches
        self.spectrograms = spectrograms
        self.synth_params = synth_params
        # self.synth_feats = synth_feats
        self.nbatches_per_synth = [len(specs) for specs in self.spectrograms]
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.nbatches_per_epoch

    def __get_synth_number_from_index(self, index):
        add = 0
        index_modulo = index
        for isynth, nbps in enumerate(self.nbatches_per_synth):          
          add += nbps
          if index < add:
            return isynth, index_modulo
          else:
            index_modulo = index - (add - nbps)
            
    def __getitem__(self, index):

        # Generate data
        X, y = self.__data_generation(index)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nbatches_per_epoch)
        
    def __data_generation(self, index):
        'Generates data containing batch_size samples' # X : (n_samples, ndim)    

        spec = np.array(self.spectrograms[index])
        synth_params = np.array(list(self.synth_params[index]),dtype=float)

        synth_feats = np.array([])

        if len(synth_params[0]) == 480:
            synth_feats = np.full((1, 480,1024),0)

        if len(synth_params[0]) == 759:
            synth_feats = np.full((1, 759,1024),1)

        if len(synth_params[0]) == 327:
            synth_feats = np.full((1, 327,1024),2)

        # synth_feats = np.swapaxes(np.array(self.synth_feats[index]),1,2)[[0]]

        # debug_bias = np.zeros((32,synth_params.shape[-1]))
        # debug_decode = np.zeros((32,1,1,synth_params.shape[-1]))

        # print(spec.shape)
        # print(synth_params.shape)
        # print(synth_feats.shape)
        return (spec, synth_feats), (spec, synth_params)