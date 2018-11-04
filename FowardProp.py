#FowardProp.py
import activation as a
import convolution as co
import numpy as np
def feedForward(self, inputs):
    #convolution layer one
    freature_maps = co.convoluteOne(self, inputs) #gets the filtered receptive field
    max_pools = co.poolOne(feature_maps) #reduce the size of the image using max pooling
    fully_connected = max_pools.reshape((40,))
    # preparing for the activation, gets all the inputs not including the bias
    for i in tange(self.input - 1): #-1 skips the bias
        self.aIn[i] = fully_connected[i]: #the last element (bias) sits there unchanged at the end in self.ai
    sum = np.dot(self.wIn.T, self.aIn)#sum of all the weights to each layer
    self.ah = a.tanh(sum) #runs the layers through the activation
    
    # output dot products and activations
    sum = np.dot(self.wOut.T, self.aHid)#sum of the layers to each input
    self.aOut = a.sigmoid(sum) #runs the output through the activation (think about adding options to change this)
    return self.aOut #returns the activation summed outputs
    
