import torch
from sklearn.feature_extraction import image
import numpy as np


class SelfAttention(torch.nn.Module):
    def __init__(self, inputDimension,qDimension,kDimension):
        super(SelfAttention, self).__init__()
        self.qDimension = qDimension
        self.kDimension = kDimension
        self.q = torch.nn.Linear(inputDimension, qDimension)
        self.k = torch.nn.Linear(inputDimension, kDimension)
        self.inputDimension = inputDimension
    def forward(self, input):
        q=self.q(input)
        k=self.k(input)
        attention=torch.matmul(q,k.t())
        attention=torch.softmax(attention,dim=1)
        return attention

class Controller(torch.nn.Module):
    def __init__(self,input,output):
        super(Controller,self).__init__()
        self.controller=torch.nn.LSTM(input,output)
        self.hidden=torch.zeros(15)
    def forward(self,input):
        output,self.hidden=self.controller(input,self.hidden)
        return output

class AgentNetwork(torch.nn.Module):
    inputDimension=0
    qDimension=0
    kDimension=0
    nOfPatches=0
    stride=0
    patchesDim=0
    layers=[]

    def center():
        pass

    def __init__(self,imageDimension=(64,64,3),qDimension=32,kDimension=32,nOfPatches=16,stride=4,patchesDim=16,firstBests=8,getFeatures=center):
        super(AgentNetwork,self).__init__()
        self.imageDimension = imageDimension
        self.stride = stride
        self.getFeatures = getFeatures
        self.firstBests = firstBests
        self.qDimension = qDimension
        self.kDimension = kDimension
        self.nOfPatches = (self.imageDimension[0]/self.stride)**2
        self.patchesDim = self.stride**2
        self.controller=Controller(self.featuresDimension(),15)
        self.attention=SelfAttention(self.patchesDim*self.imageDimension[2],self.qDimension, self.kDimension)
        self.layers.append(self.attention)
        self.layers.append(self.controller)
        self.f=AgentNetwork.center

    def forward(self):
        pass

    def getOutput(self,input):
        #n s s c
        self.patches=self.getPatches(input,self.stride)
        attention=self.attention(self.patches.view(self.nOfPatches,-1))
        bestPatches,indices=self.getBestPatches(attention)
        features=self.getFeatures(bestPatches,indices)
        actions=self.controller(bestPatches,self.firstBests)
        output=self.selectAction(actions)
        return output

    
    def getPatches(self,obs,stride):
        patches = image.extract_patches_2d(obs, (stride,stride))
        return patches
    def featuresDimension(self):
        return int(2*(64/(self.slide))**2)
    def selectAction(self,actions):
        return torch.argmax(actions)

    def getBestPatches(self,attention):
        #attention nof patches**2
        patchesAttention=attention.sum(dim=0)
        sorted,indices=patchesAttention.sort(descending=True)
        bests=sorted[:self.bests]
        indices=indices[:self.bests]
        return bests,indices

    def getParameters(self):
        pass
    
    def loadparameters(self):
        pass
    def saveparameters(self):
        pass


agent=AgentNetwork()