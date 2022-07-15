import torch
import numpy as np
from torchsummary import summary
from torch import nn
torch.set_default_tensor_type(torch.cuda.FloatTensor)
import random


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
        attention=attention/((input.shape[1])**0.5)
        attention=torch.softmax(attention,dim=1)
        return attention
class MLPController(torch.nn.Module):
    def __init__(self,input,output):
        super(MLPController,self).__init__()
        self.fc=torch.nn.Linear(input,output)
        
    def forward(self,input):

        output=self.fc(input.double())
        output=torch.softmax(output,dim=0)
        return output
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
    

    def center(self,x,y,stride):
        move=stride/2
        return [(x+move)/self.imageDimension[0],(y+move)/self.imageDimension[1]]

    def positionAndColor(self,x,y,stride,patch):
        move=stride/2
        xaxis=(x+move)/self.imageDimension[0]
        yaxis=(y+move)/self.imageDimension[1]

    def __init__(self,imageDimension=(64,64,3),qDimension=1,kDimension=1,nOfPatches=16,stride=4,patchesDim=16,firstBests=10,f=center,threshold=0.3,color=True):
        super(AgentNetwork,self).__init__()
        self.imageDimension = imageDimension
        self.stride = stride
        self.render=False
        self.color=color
        self.threshold = threshold
        self.firstBests = firstBests
        self.qDimension = qDimension
        self.kDimension = kDimension
        self.xPatches=int(self.imageDimension[0]/self.stride)
        self.nOfPatches = int((self.imageDimension[0]/self.stride)**2)
        self.patchesDim = self.stride**2
        self.controller=MLPController(self.featuresDimension(),15)
        self.attention=SelfAttention(self.patchesDim*self.imageDimension[2],self.qDimension, self.kDimension)
        self.layers.append(self.attention)
        self.layers.append(self.controller)
        self.f=f
        self.obsExample=np.load("observation.npy")
        self.removeGrad()
        

    def forward(self):
        pass

    def getOutput(self,input):
        self.patches=self.getPatches(input,self.stride)
        reshapedPatches=torch.reshape(self.patches,[self.nOfPatches,-1])
        attention=self.attention(reshapedPatches)
        bestPatches,indices,patchesAttention=self.getBestPatches(attention)
        #print(bestPatches,indices,patchesAttention,sep="\n\n\n")
        # features=self.getFeatures(bestPatches,indices,patchesAttention)
        if self.color:
            features=self.getFeaturesAndColors(bestPatches,indices,patchesAttention)
        else:
            features=self.getFeatures(bestPatches,indices,patchesAttention)
        actions=self.controller(features)
        #print(actions)
        
        output=self.selectAction(actions)
        if self.render:
            
            # print("patches",self.patches)
            # print("reshapedPatches",reshapedPatches)
            # print("attention",attention)
            # print("bestPatches",bestPatches)
            # print("indices",indices)
            # print("patchesAttention",patchesAttention)
            # print("features",features*self.imageDimension[0])
            # print("actions",actions)
            # print("output",output)
            pass
        return output

    def getFeatures(self,bestPatches,indices,patchesAttention):
        positions=[]
        indices=indices.tolist()
        for i in indices:
            row=int(i/self.xPatches)
            column=i%self.xPatches
            positions.append((row,column))
        features=torch.tensor(positions)/16
        
        return features.reshape(-1)

    def getFeaturesAndColors(self,bestPatches,indices,patchesAttention):
        positions=[]
        indices=indices.tolist()
        for i in indices:
            row=int(i/self.imageDimension[0])
            column=i%self.imageDimension[1]
            color=int(self.patches[i].mean()/255)
            positions.append((row,column,color))
        features=[[*self.f(self,row,column,self.stride,),color] for row,column,color in positions]
        features=torch.tensor(features)
        return features.reshape(-1)

    def getPatches(self,obs,stride):
        lists=[]
        for i in range(0,64,stride):
            for j in range(0,64,stride):
                patc=obs[i:i+stride,j:j+stride,:]
                lists.append(patc)
        patches=np.stack(lists)
        ret=torch.tensor(patches,dtype=torch.float)
        
        return ret 

    def featuresDimension(self):
        if self.color:
            return int(3*self.firstBests)
        return int(2*self.firstBests)
    def selectAction(self,actions):
        selected=torch.argmax(actions).reshape(1)
        if actions[selected]>self.threshold:
            return selected 
        return torch.tensor([4])
        # return selected

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
        result=torch.concat(result,0).to("cpu").numpy()
        return result

    def loadparameters(self,parameters):
        parameters=torch.tensor(parameters).double()
        parameters.cuda()
        conta=0
        for params in self.parameters():
            shape=params.data.shape
            avanti=torch.prod(torch.tensor(shape).to("cpu")).numpy()
            dati=parameters[conta:conta+avanti].reshape(shape)
            params.data=dati
            conta+=avanti
        self.double()


    def saveModel(self,val):
        #torch.save(self, "./parameters.pt")
        torch.save(self.state_dict(), "./"+val+".pt")
        torch.save(self.state_dict(), "./parameters.pt")
    def loadModel(path):
        # self=torch.load("./parameters.pt")
        # self.eval()
        model = AgentNetwork(kDimension=1,qDimension=1,color=False,firstBests=10)
        model.load_state_dict(torch.load(path))
       # model.eval()
        return model
    def removeGrad(self):
        for params in self.parameters():
            params.requires_grad=False

if __name__ == '__main__':
    agent=AgentNetwork(kDimension=1,qDimension=1,color=False)
    
    print(len(agent.getparameters()))
    i=0

    agent.loadparameters([random.random() for i in range(3200)])
        
    
    o=agent.getOutput(agent.obsExample)
    print(o)
    