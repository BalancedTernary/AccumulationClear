import math
import torch
from torch.optim.optimizer import Optimizer, required



class AccumulationClear(Optimizer):
    def __init__(self, params, lr=1e-4, resistance=None, lr_supplement=None, weight_decay=0, decay_flexibility=1):
        #parameter checks
        if lr<=0: #学习速率
            raise ValueError(f'Invalid 学习速率: {lr}')
        if resistance is not None and (resistance<0 or resistance>=1): #阻力
            raise ValueError(f'Invalid 刹车力度: {resistance}')
        if lr_supplement is not None and (lr_supplement<0): #学习率下限
            raise ValueError(f'Invalid 学习率补充: {lr_supplement}')
        if weight_decay<0 or weight_decay>=1: #权重衰减
            raise ValueError(f'Invalid 权重衰减: {weight_decay}')
        if decay_flexibility<0: #权重衰减柔性
            raise ValueError(f'Invalid 权重衰减柔性: {decay_flexibility}')

        if lr_supplement is None:
            lr_supplement=lr

        defaults = dict(lr=lr, resistance=resistance, lr_supplement=lr_supplement, weight_decay=weight_decay, decay_flexibility=decay_flexibility)
        super(AccumulationClear, self).__init__(params,defaults)

        self.lr = lr
        self.resistance = resistance
        self.lr_supplement=lr_supplement
        self.weight_decay = weight_decay
        self.decay_flexibility=decay_flexibility

    def __setstate__(self, state):
        super(AccumulationClear, self).__setstate__(state)

    def step(self, closure=None, loss=None):
        if closure is not None:
            loss = closure()
            loss.backward()
        if loss is not None and not loss.isfinite():
            return
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    p.data[~torch.isfinite(p.data)]=0
                    if p.grad is None:
                        continue
                    #p.data[~torch.isfinite(p.grad.data)]=0
                    p.grad.data[~torch.isfinite(p.grad.data)]=0
                    if p.grad.data.is_sparse:
                        raise RuntimeError('AccumulationClear does not support sparse gradients')
                    
                    state = self.state[p]  #get state dict for this param
                    if len(state) == 0:   #if first time to run...init dictionary with our desired entries
                        state['speed'] = torch.zeros_like(p.grad.data)
                    
                    
                    #d=-p.grad.data*self.lr*(p.data.abs()+1)
                    d=-p.grad.data*self.lr*(p.data.abs()+self.lr_supplement)
                    if self.resistance is None:
                        state['speed'][state['speed']*d<0]=float('nan')
                    else:
                        state['speed'][(state['speed']+d)*d<0]*=self.resistance
                    d[~torch.isfinite(d)]=0
                    state['speed']+=d
                    decayMask=state['speed']*p.data>0
                    x=state['speed'][decayMask]
                    state['speed'][decayMask]+=(-2*self.weight_decay*x*(x/self.decay_flexibility).atan().abs())/math.pi
                    #state['speed'][decayMask]*=(1-self.weight_decay)
                    state['speed'][~torch.isfinite(state['speed'])]=0

                    p.data.add_(state['speed'])
                    

        return loss



