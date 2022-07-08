import torch

def center():
    pass
def getPatches():
    pass


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
        
class AgentNetwork(torch.nn):
    def __init__(self,imageDimension,slide):
        self.imageDimension = imageDimension
        self.slide = slide
        self.controller=torch.nn.LSTM()
        self.attention=SelfAttention()
        self.f=center()

    def forward(self):
        pass

    def getOutput(self,input):
        patches=getPatches(input,self.stride)
        attention=self.attention(patches)
        actions=self.controller(attention)
        output=torch.argmax(actions)
        return output

    def getParameters(self):
        pass
    
    def loadparameters(self):
        pass
    def saveparameters(self):
        pass