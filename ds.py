#import needed classes
from ctypes import sizeof
import numpy as np

class melParamData():
    """"Data set class to load MelSepctrogram with synth parameter labels"""
    def __init__(self, set_type, synth):
        """
        Constructor: create instance of class
        :param set_type (string): data set type (train, test, validation)
        :param dir (string): data directory (directory where data is stored)
        """
        #set class variables
        self.set_type = set_type
        self.synth = synth

        total_size = len(np.load("data/" + synth + "_params.npy"))
        total_size_32factor = 32*((total_size)//32)

        #load in data
        if set_type == "train":
            self.mels = np.load("data/" + synth + "_mels.npy")[:total_size_32factor - total_size_32factor//5]
            self.params = np.load("data/" + synth + "_params.npy")[:total_size_32factor - total_size_32factor//5]

        if set_type == "test":
            self.mels = np.load("data/" + synth + "_mels.npy")[total_size_32factor - total_size_32factor//5:total_size_32factor - total_size_32factor//10]
            self.params = np.load("data/" + synth + "_params.npy")[total_size_32factor - total_size_32factor//5:total_size_32factor - total_size_32factor//10]

        if set_type == "validation":
            self.mels = np.load("data/" + synth + "_mels.npy")[total_size_32factor - total_size_32factor//10:]
            self.params = np.load("data/" + synth + "_params.npy")[total_size_32factor - total_size_32factor//10:]

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