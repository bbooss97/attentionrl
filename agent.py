import torch



class SelfAttention(torch.nn):
    def __init__(self, inputDimension,qDimension,kDimension):
        super(SelfAttention, self).__init__()
        self.qDimension = qDimension
        self.kDimension = kDimension
        self.q = torch.nn.Linear(inputDimension, qDimension)
        self.k = torch.nn.Linear(qDimension, kDimension)
        self.inputDimension = inputDimension
    def forward(self, input):
        q=self.q(input)
        k=self.k(input)
        attention=torch.matmul(q,k.t())
        attention=torch.softmax(attention,dim=1)
        return attention

class Controller():
    def __init__(self,input,output):
        self.controller=torch.nn.LSTM(input,output)
        self.hidden=torch.zeros(1,1,1)
    def forward(self,input):
        output,self.hidden=self.controller(input,self.hidden)
        return output

class AgentNetwork(torch.nn):
    inputDimension=0
    qDimension=0
    kDimension=0
    patches=0
    stride=0
    layers=[]
    def __init__(self,imageDimension,slide):
        self.imageDimension = imageDimension
        self.slide = slide
        self.controller=Controller(AgentNetwork.featuresDimension(),15)
        self.attention=SelfAttention(self.inputDimension,self.qDimension, self.kDimension)
        self.layers.append(self.selfattention)
        self.layers.append(self.controller)
        self.f=AgentNetwork.center

    def forward(self):
        pass

    def getOutput(self,input):
        patches=AgentNetwork.getPatches(input,self.stride)
        attention=self.attention(patches)
        actions=self.controller(attention)
        output=torch.argmax(actions)
        return output

    def center():
        pass
    def getPatches():
        pass
    def featuresDimension():
        pass

    def getParameters(self):
        pass
    
    def loadparameters(self):
        pass
    def saveparameters(self):
        pass