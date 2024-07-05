import torch
import math
from Play import Play

maxTimes=100000
BatchSize=3000

matrixSize=8
layers=400
features=512

Play_lr=2e-4
Adam_lr=2e-4
weight_decay=0

class Block(torch.nn.Module):
    def __init__(self,in_features,out_features,gain,**kwargs):
        super(Block, self).__init__()
        self.Linear=torch.nn.Linear(in_features,out_features,bias=True)
        with torch.no_grad():
            torch.nn.init.orthogonal_(self.Linear.weight,gain)
            self.Linear.bias.fill_(0)
    def forward(self,input):
        return self.Linear(input).asinh()

#This network architecture is also my original work and follows the same open source license,
#including its extension to any/every dimensional convolution and replacement of any/every activation functions.        
class AccessNet(torch.nn.Module):
    def __init__(self,in_features,out_features,hidden_layers,hidden_features,**kwargs):
        super(AccessNet, self).__init__()
        self.inLayer=torch.nn.Linear(in_features,hidden_features,bias=True)
        self.Blocks=torch.nn.Sequential() 
        for t in range(hidden_layers):
            self.Blocks.add_module('hiddenLayer{0}'.format(t),Block(hidden_features,hidden_features,1/math.sqrt(hidden_layers)))
        self.outLayer=torch.nn.Linear(hidden_features,out_features,bias=True)
        
    def forward(self,input):
        step=self.inLayer(input)
        outputAccess=torch.zeros_like(step)
        for block in self.Blocks:
            step=block(step)
            outputAccess=outputAccess+step
        output=self.outLayer(outputAccess)
        return output

moduel1=AccessNet(matrixSize*matrixSize,matrixSize*matrixSize,layers,features)
moduel2=AccessNet(matrixSize*matrixSize,matrixSize*matrixSize,layers,features)
optimizer1 = torch.optim.Adam(moduel1.parameters(),Adam_lr,weight_decay=weight_decay)
optimizer2 = Play(moduel2.parameters(),lr=Play_lr,weight_decay=weight_decay)

device = "cuda" if torch.cuda.is_available() else "cpu"

moduel1.to(device)
moduel2.to(device)
eye=torch.eye(matrixSize,device=device).unsqueeze(0)

import visdom
wind = visdom.Visdom(env="Optimizer Test", use_incoming_socket=False)
    
wind.line([[float('nan'),float('nan')]],[0],win = 'loss',opts = dict(title = 'log(loss)/log(batchs)',legend = ['Adam','Play']))

print(moduel1)
    
for time in range(maxTimes):
    moduel1.zero_grad()
    input=torch.rand(BatchSize,matrixSize*matrixSize,device=device)*2-1
    output=moduel1(input)
    input=input.view(BatchSize,matrixSize,matrixSize)
    output=output.view(BatchSize,matrixSize,matrixSize)
    loss1=(input@output-eye).abs().mean()
    loss1.backward()
    optimizer1.step()
    
    moduel2.zero_grad()
    input=input.view(BatchSize,matrixSize*matrixSize)
    output=moduel2(input)
    input=input.view(BatchSize,matrixSize,matrixSize)
    output=output.view(BatchSize,matrixSize,matrixSize)
    loss2=(input@output-eye).abs().mean()
    loss2.backward()
    optimizer2.step(loss=loss2)
    
    wind.line([[float(loss1.log()),float(loss2.log())]],[math.log(time+1)],win = 'loss',update = 'append')
    #wind.line([[float(loss1),float(loss2)]],[time],win = 'loss',update = 'append')
