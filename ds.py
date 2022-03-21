#import needed classes
import numpy as np

class melParamData():
    """"Data set class to load MelSepctrogram with synth parameter labels"""
    def __init__(self, set_type, dir):
        """
        Constructor: create instance of class
        :param set_type (string): data set type (train, test, validation)
        :param dir (string): data directory (directory where data is stored)
        """
        #set class variables
        self.set_type = set_type
        self.dir = dir

        #load in data
        if set_type == "train":
            self.mels = np.load(dir + "/asp_data_mels.npy")[:25000]
            self.params = np.load(dir + "/asp_data_params.npy")[:25000]

        if set_type == "test":
            self.mels = np.load(dir + "/asp_data_mels.npy")[25000:27000]
            self.params = np.load(dir + "/asp_data_params.npy")[25000:27000]

        if set_type == "validation":
            self.mels = np.load(dir + "/asp_data_mels.npy")[27000:]
            self.params = np.load(dir + "/asp_data_params.npy")[27000:]

        # #delete elements where spectrogram is zero
        # zero_locations = np.where(np.min(self.mels, axis=(1,2))==0.0)[0]
        # self.mels = np.delete(self.mels, zero_locations, axis=0)
        # self.params = np.delete(self.params, zero_locations, axis=0)


        # #normalize mel spectrogram
        # for i,mel in enumerate(self.mels):
        #     self.mels[i] = (mel - np.min(mel))
        #     self.mels[i] = self.mels[i] / np.max(self.mels[i])

        #set input and output size
        self.input_size = self.mels[0].shape
        self.output_size = self.params[0].shape


    def __len__(self):
        """
        __len__: get the lenght of data set
        :return: The number of samples in data set
        """
        return self.mels.shape[0]

    def __getitem__(self, idx):
        """
        __getitem__: get a specific item from [] index
        :param idx: index
        :return: sample at specified index
        """

        #get data
        mel = self.mels[idx]
        params = self.params[idx]

        return mel,params

    def get_mels(self):
        """
        get_mels: return mel spectrogram feature array
        :return: numpy array of mel spectrogram features
        """
        return self.mels

    def get_params(self):
        """
        get_params: return synthesizer parameter labels
        :return: numpy array of parameter labels
        """
        return self.params