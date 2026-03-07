import torch 
import torch.nn as nn
import numpy as np 

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x=np.array([2.0,1.0,0.1])
outputs=softmax(x)
print(outputs)

x=torch.tensor([2.0,1.0,0.1])
outputs=torch.softmax(x, dim=0)
print(outputs)

def cross_entropy(actual, predicted):
    loss=-np.sum(actual*np.log(predicted))
    return loss 

Y=np.array([1,0,0])

Y_pred_good=np.array([0.7,0.2,0.1])
Y_pred_bad=np.array([0.1,0.3,0.6])
l1=cross_entropy(Y, Y_pred_good)
l2=cross_entropy(Y, Y_pred_bad)
print(f"{l1:.4f}")
print(f"{l2:.4f}")

loss=nn.CrossEntropyLoss()

Y=torch.tensor([0])
# nsamples x nclasses
Y_pred_good=torch.tensor([[2.0,1.0,0.1]])
Y_pred_bad=torch.tensor([[0.5,2.0,0.3]])

