import torch
import numpy as np
from torchsummary import summary
from torch import nn



class SelfAttention(torch.nn.Module):
    def __init__(self, inputDimension,qDimension,kDimension):
        super(SelfAttention, self).__init__()
        self.qDimension = qDimension
        self.kDimension = kDimension
        self.q = torch.nn.Linear(inputDimension, qDimension)
        self.k = torch.nn.Linear(inputDimension, kDimension)
        self.inputDimension = inputDimension
        # torch.nn.init.xavier_uniform(self.q.weight)
        # torch.nn.init.xavier_uniform(self.k.weight)
    def forward(self, input):
        input=input.double()
        q=self.q(input)
        k=self.k(input)
        attention=torch.matmul(q,k.t())
        attention=torch.softmax(attention,dim=1)
        return attention

class Controller(torch.nn.Module):
    def __init__(self,input,output):
        super(Controller,self).__init__()
        self.controller=torch.nn.LSTM(input_size=input,hidden_size=15,num_layers=1)
        self.hidden=(torch.zeros(1,15).double(),torch.zeros(1,15).double())
        self.fc=torch.nn.Linear(15,output)

    def forward(self,input):
        output,self.hidden=self.controller(input.view(1,-1).double(),self.hidden)
        output=self.fc(output).squeeze()
        output=torch.softmax(output,dim=0)
        return output


class AgentNetwork(torch.nn.Module):
    inputDimension=0
    qDimension=0
    kDimension=0
    nOfPatches=0
    stride=0
    patchesDim=0
    layers=[]

    def center(x,y,stride):
        move=stride/2
        return [x+move,y+move]

    def __init__(self,imageDimension=(64,64,3),qDimension=10,kDimension=10,nOfPatches=16,stride=4,patchesDim=16,firstBests=8,f=center):
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
        #self.removeGrad()
        

    def forward(self):
        pass

    def getOutput(self,input):
        self.patches=self.getPatches(input,self.stride)
        reshapedPatches=torch.reshape(self.patches,[self.nOfPatches,-1])
        attention=self.attention(reshapedPatches)
        bestPatches,indices,patchesAttention=self.getBestPatches(attention)
        #print(bestPatches,indices,patchesAttention,sep="\n\n\n")
        features=self.getFeatures(bestPatches,indices,patchesAttention)
    
        actions=self.controller(features)
        #print(actions)
        output=self.selectAction(actions)
        
        return output

    def getFeatures(self,bestPatches,indices,patchesAttention):
        positions=[]
        indices=indices.tolist()
        for i in indices:
            row=int(i/self.xPatches)
            column=i%self.xPatches
            positions.append((row,column))
        features=[self.f(row,column,self.stride) for row,column in positions]
        features=torch.tensor(features)
        return features.reshape(-1)


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
        return torch.argmax(actions).reshape(1)

    def getBestPatches(self,attention):
        #attention nof patches**2
        patchesAttention=attention.sum(dim=0)
        sorted,indices=patchesAttention.sort(descending=True)
        bests=sorted[0:self.firstBests]
        indices=indices[0:self.firstBests]
        return bests,indices,patchesAttention

    def getparameters(self):
        result=[]
        for params in self.parameters():
            a=params.data.reshape(-1)
            result.append(a)
        result=torch.concat(result,0).numpy()
        return result

    def loadparameters(self,parameters):
        parameters=torch.tensor(parameters).double()
        conta=0
        for params in self.parameters():
            shape=params.data.shape
            avanti=torch.prod(torch.tensor(shape)).numpy()
            dati=parameters[conta:conta+avanti].reshape(shape)
            params.data=dati
            conta+=avanti
        self.double()


    def saveModel(self):
        #torch.save(self, "./parameters.pt")
        torch.save(self.state_dict(), "./parameters.pt")
    def loadModel(self):
        # self=torch.load("./parameters.pt")
        # self.eval()
        model = AgentNetwork(kDimension=10,qDimension=10)
        model.load_state_dict(torch.load("./parameters.pt"))
        model.eval()
    def removeGrad(self):
        for params in self.parameters():
            params.requires_grad=False

if __name__ == '__main__':
    agent=AgentNetwork(kDimension=10,qDimension=10)
    summary(agent)
    print(agent)
    i=0

    agent.loadparameters([0 for i in range(3200)])
        
    
    #agent.getOutput(agent.obsExample)