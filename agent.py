import torch
from skimage.util import view_as_windows
import numpy as np
from torchsummary import summary


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
        self.controller=torch.nn.LSTM(input_size=input,hidden_size=15,num_layers=1)
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

    def __init__(self,imageDimension=(64,64,3),qDimension=32,kDimension=32,nOfPatches=16,stride=4,patchesDim=16,firstBests=8,f=center):
        super(AgentNetwork,self).__init__()
        self.imageDimension = imageDimension
        self.stride = stride
        self.firstBests = firstBests
        self.qDimension = qDimension
        self.kDimension = kDimension
        self.xPatches=int(self.imageDimension[0]/self.stride)
        self.nOfPatches = int((self.imageDimension[0]/self.stride)**2)
        self.patchesDim = self.stride**2
        self.controller=Controller(self.featuresDimension(),15)
        self.attention=SelfAttention(self.patchesDim*self.imageDimension[2],self.qDimension, self.kDimension)
        self.layers.append(self.attention)
        self.layers.append(self.controller)
        self.f=f
        self.obsExample=np.load("observation.npy")
        

    def forward(self):
        pass

    def getOutput(self,input):
        self.patches=self.getPatches(input,self.stride)
        self.patches=torch.reshape(self.patches,[self.nOfPatches,-1])
        attention=self.attention(self.patches)
        print(attention.shape)
        bestPatches,indices,patchesAttention=self.getBestPatches(attention)
        features=self.getFeatures(bestPatches,indices,patchesAttention)
        actions=self.controller(features)
        output=self.selectAction(actions)
        return output

    def getFeatures(self,bestPatches,indices,patchesAttention):
        positions=[]
        for i in indices:
            row=i%self.stride
            column=i-row*self.stride
            positions.append((row,column))
        features=torch.tensor([self.f(i) for i in positions])
        return features


    def getPatches(self,obs,stride):
        lists=[]
        for i in range(0,64,stride):
            for j in range(0,64,stride):
                patc=obs[i:i+stride,j:j+stride,:]
                lists.append(patc)
        patches=np.stack(lists)
        return torch.tensor(patches,dtype=torch.float)

    def featuresDimension(self):
        return int(2*self.firstBests)
    def selectAction(self,actions):
        return torch.argmax(actions)

    def getBestPatches(self,attention):
        #attention nof patches**2
        patchesAttention=attention.sum(dim=0)
        sorted,indices=patchesAttention.sort(descending=True)
        bests=sorted[:self.bests]
        indices=indices[:self.bests]
        return bests,indices,patchesAttention

    def getParameters(self):
        pass
    
    def loadparameters(self):
        pass
    def saveparameters(self):
        pass


agent=AgentNetwork()
summary(agent)
print(agent)
agent.getOutput(agent.obsExample)