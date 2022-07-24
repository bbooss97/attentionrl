import torch
import numpy as np
from torchsummary import summary
from torch import nn
torch.set_default_tensor_type(torch.cuda.FloatTensor)
import random
import time

class FeatureExtractor(torch.nn.Module):
    def __init__(self,x,y):
        super(FeatureExtractor, self).__init__()
        self.fc1=torch.nn.Linear(x,5)
        self.relu=torch.nn.ReLU()
        self.fc2=torch.nn.Linear(5,y)
    def forward(self,x):
        x=x.double()
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        return x

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
        attention=torch.einsum('bij,bjk->bik', q, k.reshape(input.shape[0],self.kDimension,225))
        attention=attention/((input.shape[2])**0.5)
        attention=torch.softmax(attention,dim=2)
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
class MLPController(torch.nn.Module):
    def __init__(self,input,output):
        super(MLPController,self).__init__()
        self.fc=torch.nn.Linear(input,20)
        self.fc1=torch.nn.Linear(20,output)
        self.fc2=torch.nn.Linear(output,output)

    def forward(self,input):
        #print(self.fc(input.double()))
        output=nn.Sigmoid()(self.fc(input.double()))
        output=nn.Sigmoid()(self.fc1(output))
        output=self.fc2(output)
        output=torch.softmax(output,dim=1)
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

    def __init__(self,imageDimension=(64,64,3),qDimension=3,kDimension=3,nOfPatches=16,stride=4,patchesDim=16,firstBests=8,f=center,threshold=0.33,color=True,num=1,render=False):
        super(AgentNetwork,self).__init__()
        self.imageDimension = imageDimension
        self.stride = stride
        self.render=render
        self.num=num
        self.color=color
        self.threshold = threshold
        self.firstBests = firstBests
        self.qDimension = qDimension
        self.kDimension = kDimension
        self.xPatches=int(self.imageDimension[0]/self.stride)
        self.nOfPatches = int((self.imageDimension[0]/self.stride)**2)
        self.patchesDim = self.stride**2
        self.controller=MLPController(self.featuresDimension(),15)
        self.attention=SelfAttention(147,self.qDimension, self.kDimension)
        self.layers.append(self.attention)
        self.layers.append(self.controller)
        self.f=f
        self.obsExample=torch.tensor(np.load("parallelObs.npy"))
        self.removeGrad()
        self.featureExtractor=FeatureExtractor(49*3,3)
        

    def forward(self):
        pass

    def getOutput(self,input):
        input=input/255
        self.patches=self.getPatches(input,self.stride)

        attention=self.attention(self.patches)

        bestPatches,indices,patchesAttention=self.getBestPatches(attention)

        if self.color:
            features=self.getFeaturesAndColors(bestPatches,indices,patchesAttention)
        else:
            #features=self.getFeatures(bestPatches,indices,patchesAttention)
            features=self.getFeautresFromExtractor(bestPatches,indices,patchesAttention)
        actions=self.controller(features)
        
        output=self.selectAction(actions)
        if self.render:
            print(indices)
            print(features)

            pass
        return output

    def getFeatures(self,bestPatches,indices,patchesAttention):
        col=(indices%25).int()
        row=(indices/25).int()
        if self.render:
            print(row,col)
        features=torch.cat((row*4+4,col*4+4),1)/64
        return features
    
    def getFeautresFromExtractor(self,bestPatches,indices,patchesAttention):
        col=(indices%25).int()
        row=(indices/25).int()
        
        a=torch.outer(torch.arange(0,self.num),torch.ones(self.firstBests)).reshape(-1).long()
        b=indices.reshape(-1).long()
        
        
        
        selected=self.patches[a,b].reshape(self.num,self.firstBests,-1)
        
        extracted=self.featureExtractor(selected).reshape(self.num,-1)
        
        if self.render:
            print(row,col)
        features=torch.cat((row*4+4,col*4+4),1)/64
        features=torch.cat((features,extracted),1)
        
        return features

    def getFeaturesAndColors(self,bestPatches,indices,patchesAttention):
        indices=indices.tolist()
        res=[]
        for i in range(len(indices)):
            positions=[]
            for j in range(len(indices[i])):
                row=int(indices[i][j]/self.imageDimension[0])
                column=indices[i][j]%self.imageDimension[1]
                #print(self.patches.shape)
                color=list(self.patches[i][indices[i][j]].reshape(16,3).mean(axis=0)/255)
                positions.append((row,column,color))
            features=[[*self.f(self,row,column,self.stride,),*color] for row,column,color in positions]
            res.append(features)
        res=torch.tensor(res).reshape(self.num,-1)
        return res


    def getPatches(self,obs,stride):
        obs=torch.einsum("abcd->adbc",obs)
        unfold=nn.Unfold(kernel_size=(7,7),stride=4)
        obs=unfold(obs)
        obs=obs.transpose(1,2)
        # print(obs.shape)
        return obs


    def featuresDimension(self):
        if self.color:
            return int(5*self.firstBests)
        return int(2*self.firstBests+self.firstBests*3)
    def selectAction(self,actions):
        selected=torch.argmax(actions,axis=1).reshape(-1)
        return selected


    def getBestPatches(self,attention):
        #attention nof patches**2
        patchesAttention=attention.sum(dim=1)
        sorted,indices=patchesAttention.sort(descending=True,dim=1)
        bests=sorted[:,0:self.firstBests]
        indices=indices[:,0:self.firstBests]
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
    def loadModel(self,path):
        # self=torch.load("./parameters.pt")
        # self.eval()
        self.load_state_dict(torch.load(path))
       # model.eval()
        return self
    def removeGrad(self):
        for params in self.parameters():
            params.requires_grad=False

if __name__ == '__main__':
    agent=AgentNetwork(num=100,color=False)
    
    print(len(agent.getparameters()))
    i=0

    agent.loadparameters([float(1) for i in range(3021)])
        
    
    o=agent.getOutput(agent.obsExample)
    print(o)
    