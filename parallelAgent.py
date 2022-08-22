#this contains the parallel agent used to play the game in the vectorized environment

import torch
import numpy as np
from torch import nn
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

#in the original paper for every generated parameters there is a container that executes num games and take the average of those
#i not having at disposal the cloud used the gpu to parallelize the num games 
#i also wrote a script where i create an agent for every parameter generated in multiprocessing
#and for every of them i execute num games in parallel
#my gpu has not that much dram 3gb so i could only use 5 agents together considering the parallel environments observation
# so its not that useful in my case but with a gpu with more memory it could speed things up 


#all the network components are batched :
#i have num games and for each game i output a single action
#in this way i use the gpu to parallelize the network operations for the num games




#module to extract automatic features from a patch 
class FeatureExtractor(torch.nn.Module):
    def __init__(self,x,y):
        super(FeatureExtractor, self).__init__()
        #linear layers with relu activation
        self.fc1=torch.nn.Linear(x,5)
        self.relu=torch.nn.ReLU()
        self.fc2=torch.nn.Linear(5,y)
    def forward(self,x):
        x=x.double()
        x=self.fc1(x)
        x=self.relu(x)
        x=torch.nn.Sigmoid()(self.fc2(x))
        #output a sigmoid of the fetures extracted
        return x

#self attention module to otain the attention map from the patches
class SelfAttention(torch.nn.Module):
    def __init__(self, inputDimension,qDimension,kDimension):
        super(SelfAttention, self).__init__()
        self.qDimension = qDimension
        self.kDimension = kDimension
        #query and key linear layers
        self.q = torch.nn.Linear(inputDimension, qDimension)
        self.k = torch.nn.Linear(inputDimension, kDimension)
        self.inputDimension = inputDimension

    def forward(self, input):
        input=input.double()
        #get query and keys
        #shape num, number of patches, qDimension or kDimension
        q=self.q(input)
        k=self.k(input)
        #transpose k for self attention
        kTrasposed=torch.einsum("abc->acb",k)
        #batched matrix multiplication
        #shape num , number of patches, number of patches in this case 225
        attention=torch.einsum('bij,bjk->bik', q,kTrasposed)
        #scaling factor sqrt of dimension of key vector like in normal self attention
        attention=attention/((input.shape[2])**0.5)
        #softmax along the last dimension
        #shape num , number of patches, number of patches in this case 225
        attention=torch.softmax(attention,dim=2)
        return attention


class LstmController(torch.nn.Module):
    def __init__(self,input,output,num):
        super().__init__()
        self.num=num
        #initialize hidden state
        self.init_hidden()
        #lstm layer with sequence length 1
        self.lstm = nn.LSTM(input, 15, 1, batch_first=True)
        #linear layers after lstm
        self.fc1=torch.nn.Linear(15,15)
        self.fc2 = nn.Linear(15, output)
        

    def forward(self, x):
        self.lstm.flatten_parameters()
        #get output of lstm and hidden state give n batched input(num) of sequence lenght 1 and hidden state
        #x initially has shape num,number of input that depends on the function getFeaturesDimension
        output, self.hidden = self.lstm(x.view(self.num,1,-1).double(), self.hidden)
        #get output from the linear layers and softmax it to then select the action to do
        output = torch.nn.functional.relu(self.fc1(output).squeeze(dim=1))
        output=self.fc2(output)
        output = torch.nn.functional.softmax(output, dim=1)
        #output has dimension num,output
        return output
        
    def init_hidden(self):
        #initialize hidden state with batches sequence lenghts and 15 neurons lstm
        hidden = torch.zeros(1,self.num, 15,requires_grad=False).double()
        cell = torch.zeros(1, self.num, 15,requires_grad=False).double()
        self.hidden=hidden,cell
    def resetLstmState(self,indexes):
        self.hidden[0][:,indexes,:]=0
        self.hidden[1][:,indexes,:]=0

#this is a mlp controller in case we dont want to use lstm 
class MLPController(torch.nn.Module):
    def __init__(self,input,output):
        super(MLPController,self).__init__()
        self.fc=torch.nn.Linear(input,20)
        self.fc1=torch.nn.Linear(20,output)
        self.fc2=torch.nn.Linear(output,output)

    def forward(self,input):
        output=nn.Sigmoid()(self.fc(input.double()))
        output=nn.Sigmoid()(self.fc1(output))
        output=self.fc2(output)
        output=torch.softmax(output,dim=1)
        #shape num,outputDim(15)
        return output

class AgentNetwork(torch.nn.Module):

    
    def __init__(self,imageDimension=(64,64,3),qDimension=6,kDimension=6,extractorOutput=1,firstBests=5,threshold=0.33,color=False,num=1,render=False,useLstm=True):
        super(AgentNetwork,self).__init__()
        #qDimension and kDimension are the dimension of the query and key vectors in the self attention module
        #extractorOutput is the dimension of the output of the feature extractor that automatically detects the features of the patches beyond the positions coordinates
        #stride is the stride of the patches that is 4 as in the paper and the patches dimension is 7x7
        #firstBests is the number of best patches to consider to give as input to the controller
        #threshold is the threshold to do an action otherwise do nothing 4
        #color is a boolean that tells if we want to take color features beyond the positions features
        #num is the number of games to play in parallel
        #render is a boolean that tells if we want to render the game to see it in action when it plays (used in the testAgent script)
        #useLstm is a boolean that tells if we want to use lstm otherwise we use the mlp as controller
        self.imageDimension = imageDimension
        self.stride = 4
        self.extractorOutput = extractorOutput
        self.render=render
        self.num=num
        self.color=color
        self.threshold = threshold
        self.firstBests = firstBests
        self.qDimension = qDimension
        self.kDimension = kDimension
        self.useLstm=useLstm
        if self.useLstm==True:
            self.controller=LstmController(self.featuresDimension(),15,num)
        else:
            self.controller=MLPController(self.featuresDimension(),15)
        self.attention=SelfAttention(147,self.qDimension, self.kDimension)
        #i remove the gradient from the network because i use cmaes to update the weights and there is a leak in memory with the lstm if i dont remove it
        self.removeGrad()
        self.featureExtractor=FeatureExtractor(49*3,self.extractorOutput)

    def forward(self):
        pass

    def getOutput(self,input):
        #i put the input in cuda because i use the gpu
        input=torch.tensor(input)
        # normalize pixels colors
        input=input/255
        #get the patches from the input
        self.patches=self.getPatches(input,self.stride)
        #get the attention map from the patches
        attention=self.attention(self.patches)
        #get the best patches,their indices and the patches attention 
        bestPatches,indices,patchesAttention=self.getBestPatches(attention)
        #if color i use as features the positions and the color of the patches
        #if extractorOutput is 0 i use the positions and the automatically detected features from the patches
        #otherwise i only use the positions as features for the controller
        if self.color:
            features=self.getFeaturesAndColors(bestPatches,indices,patchesAttention)
        elif self.extractorOutput==0:
            features=self.getFeatures(bestPatches,indices,patchesAttention)
        else:
            features=self.getFeautresFromExtractor(bestPatches,indices,patchesAttention)
        #get the output actions softmaxed that are also batched so num,outputDim(15)
        actions=self.controller(features)
        #select the best action in this case the argmax but i could use a probabilistic approach based on the softmaxed output
        output=self.selectAction(actions)
        if self.render:
            self.drawAttentionMap(indices)
        return output

    def drawAttentionMap(self,indices):
        col=(indices%15).int()
        row=(indices/15).int()
        r=row.tolist()[0]
        c=col.tolist()[0]
        m=[[0 for i in range(15)]for j in range(15)]
        for i in range(len(r)):
            m[r[i]][c[i]]=i+1
        for i in m:
            print()
            for j in i:
                print("{:3d}".format(j),end=" ")
        print("\n\n")
    #get the positions features of the best patches
    def getFeatures(self,bestPatches,indices,patchesAttention):
        #get the col and row indices of the best patches
        col=(indices%15).int()
        row=(indices/15).int()
        #get center and normalize it
        features=torch.cat((row*4+4,col*4+4),1)/64
        return features
    #get the positions and the color features of the best patches
    #the color is a 3 values where each of them is the mean of the corresponding rgb color in the patch
    def getFeautresFromExtractor(self,bestPatches,indices,patchesAttention):
        col=(indices%15).int()
        row=(indices/15).int()
        #those are operations to get the selected patches in a vectorized way without loops for the gpu
        #i essentially index the patches and then extract from the patches the features 
        a=torch.outer(torch.arange(0,self.num),torch.ones(self.firstBests)).reshape(-1).long()
        b=indices.reshape(-1).long()
        selected=self.patches[a,b].reshape(self.num,self.firstBests,-1)
        extracted=self.featureExtractor(selected).reshape(self.num,-1)
        
        #get center and features extracted as input for the controller
        features=torch.cat(((row*4+4)/64,(col*4+4)/64),1)
        features=torch.cat((features,extracted),1)
        return features

    #this gets positions center and mean of every values for the color
    def getFeaturesAndColors(self,bestPatches,indices,patchesAttention):
        col=(indices%15).int()
        row=(indices/15).int()
        #those are operations to get the selected patches in a vectorized way without loops for the gpu
        #i essentially index the patches and then extract from the patches the features 
        a=torch.outer(torch.arange(0,self.num),torch.ones(self.firstBests)).reshape(-1).long()
        b=indices.reshape(-1).long()
        selected=self.patches[a,b]
        selected=selected.reshape(self.num,self.firstBests,49,3)
        extracted=selected.mean(2).reshape(self.num,-1)
        features=torch.cat(((row*4+4)/64,(col*4+4)/64),1)
        #get center and concat the color features
        features=torch.cat((features,extracted),1)
        return features

    #this fuction allows to get the patches from the input
    #the patches are of dimension 7x7 and have a stride of 4
    def getPatches(self,obs,stride):
        #obs initially num,64,64,3 then num,3,64,64
        obs=torch.einsum("abcd->adbc",obs)
        unfold=nn.Unfold(kernel_size=(7,7),stride=4)
        obs=unfold(obs)
        obs=obs.transpose(1,2)
        #now obs has shape num,numpatches**2,7*7*3
        return obs

    #this function gives the input dimension of the controller
    #if color then (2+3)*firstBests 2 positions 3 colors
    def featuresDimension(self):
        if self.color:
            return int(5*self.firstBests)
        elif self.extractorOutput==0:
            #if no extractor output then 2*firstbests for the positions coordinate
            return int(2*self.firstBests)
        #else the (2+extractedoutput)*firstbests 2 is the positions
        return int(2*self.firstBests+self.firstBests*self.extractorOutput)

    #this function gets the actions from the softmaxed output of the controller in this case its a argmax 
    #i could have done also something probabilistic with the softmaxed output
    def selectAction(self,actions):
        #actions inithially has shape( num,outputDim(15))
        selected=torch.argmax(actions,axis=1).reshape(-1)
        #if >threshold then i do an actions otherwise i do nowthing(action 4)
        if self.threshold>0:
            for i in range(selected.shape[0]):
                if actions[i][selected[i]]<self.threshold:
                    selected[i]=4        
        #selectted has now shape num where for each num(game) i have the action to do 
        return selected
    #this function allows to get the firstbest patches and their indices from the attention map
    def getBestPatches(self,attention):
        patchesAttention=attention.sum(dim=1)
        sorted,indices=patchesAttention.sort(descending=True,dim=1)
        bests=sorted[:,0:self.firstBests]
        indices=indices[:,0:self.firstBests]
        return bests,indices,patchesAttention

    #this function allows to get the parameters of the entire agent network 
    def getparameters(self):
        result=[]
        for params in self.parameters():
            a=params.data.reshape(-1)
            result.append(a)
        result=torch.concat(result,0).to("cpu").numpy()
        return result
    #this function is used to load to the agent the parameters given from the cmaes algorithm
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
        # if self.useLstm:
        #     self.controller.init_hidden()

    #this function allows to save the model to a path
    def saveModel(self,path):
        torch.save(self.state_dict(), "./"+path+".pt")
        torch.save(self.state_dict(), "./parameters.pt")
    #this function allows to load the model from a path
    def loadModel(self,path):
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(path))
        else:
            self.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
        return self
    #this is used to remove the autograd because i dont use it having the cmaes algorithm to optimize the parameters
    #morover i had to add this because otherwise the lstm layer gets a computatioonal graph that continues to increase and in the end saturates the memory
    def removeGrad(self):
        for params in self.parameters():
            params.requires_grad=False
        torch.autograd.set_grad_enabled(False)

if __name__ == '__main__':
    agent=AgentNetwork(num=100,extractorOutput=0,color=False,useLstm=True)
    numberOfParameters=len(agent.getparameters())
    agent.loadparameters([float(1) for i in range(numberOfParameters)])
    obsExample=torch.tensor(np.load("parallelObs.npy"))
    o=agent.getOutput(obsExample)
    print(o)
    