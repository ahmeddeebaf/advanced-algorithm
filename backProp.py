#backProp.py
import activation as a
import numpy as np
def backPropagate(self, targets):
    #output layer:
    output_error = -(targetrs - self.aOut)
    output_derivative = a.dsigmoid(self.aOut)
    output_deltas = output_error*output_derivative
    change = output_deltas * np.reshape(self.aHid, (self.aHid.shape[0], 1)) #update the wights connecting hidden output
    self.wOut -= self.learning_rate * change + selfcOut * self.momentum
    self.cOut = change
    
    #Hidden layer:
    hidden_error = np.dot(self.wOut, output_deltas)
    hidden_derivative = a.dtanh(self.aHid)
    hidden_deltas = hidden_error * hidden_derivative
    change = hidden_deltas * np.reshape(self.aIn.shape[0], 1) #update the weights connecting imput to hidden
    self.wIn -= self.learning_rate * change + self.cIn * self.momentum
    self.cIn = change
    #convolutional layer:
    print "hidden delta:", hidden_deltas.shape
    print "filter:", self.filters.shape
    con_error = np.dot = np.dot(self.filters.T, hidden_deltas)
    con_derivative = a.drelu(self.aCon)
    con_deltas = con_error * con_derivative
    print con_deltas.shape, self.aCon.shape
    change = con_deltas * self.aCon
    #self.filters -= self.learning_rate 8 rate change +self.cCon * self.momentum
    self.cCon = change
    #calculate error
    error = sum(0.5 * (targets - self.aOut) **2)
    return error
