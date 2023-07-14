import math
import torch
from torch.optim.optimizer import Optimizer, required



class AccumulationClear(Optimizer):
    def __init__(self, params, lr=1e-4, resistance=None, lr_supplement=None, weight_decay=0, decay_flexibility=1, smoots_decrement=1e-2, snapshot_recovery_threshold=10, limit_snapshot_cycle=100):
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
        if smoots_decrement<0: #平滑衰减量,对loss取包络时使用,约小越平滑
            raise ValueError(f'Invalid 平滑衰减量: {smoots_decrement}')
        if snapshot_recovery_threshold<=1: #快照恢复阈值,当前平滑loss大于平滑loss的最小值的这个倍数时恢复快照
            raise ValueError(f'Invalid 快照恢复阈值: {snapshot_recovery_threshold}')
        if limit_snapshot_cycle<=1: #快照周期下限
            raise ValueError(f'Invalid 快照周期下限: {limit_snapshot_cycle}')

        if lr_supplement is None:
            lr_supplement=lr

        defaults = dict(lr=lr, resistance=resistance, lr_supplement=lr_supplement, weight_decay=weight_decay, decay_flexibility=decay_flexibility, smoots_decrement=smoots_decrement, snapshot_recovery_threshold=snapshot_recovery_threshold, limit_snapshot_cycle=limit_snapshot_cycle)
        super(AccumulationClear, self).__init__(params,defaults)

        self.lr = lr
        self.resistance = resistance
        self.lr_supplement=lr_supplement
        self.weight_decay = weight_decay
        self.decay_flexibility=decay_flexibility
        self.smoots_decrement=smoots_decrement
        self.snapshot_recovery_threshold=snapshot_recovery_threshold
        self.limit_snapshot_cycle=limit_snapshot_cycle
        self.minLoss = 0
        self.upperEnvelope = -2**15
        self.lowerEnvelope = 2**15
        self.upperEnvelopeCache = -2**15
        self.lowerEnvelopeCache = 2**15
        self.minNeutral = 2**15 #上包络与下包络的平均值的最小值
        self.t=0 #计数器,拍摄快照后清零

    def __setstate__(self, state):
        super(AccumulationClear, self).__setstate__(state)

    def step(self, closure=None, loss=None):
        if closure is not None:
            loss = closure()
            loss.backward()
        if loss is None:
            loss=1
        if loss<self.minLoss:
            self.minLoss=loss
        self.t+=1

        self.upperEnvelope-=self.smoots_decrement
        self.lowerEnvelope+=self.smoots_decrement
        if loss>self.upperEnvelope:
            self.upperEnvelope=loss
        if loss<self.lowerEnvelope: #前面不能加else,不然因为两者的自动衰减会导致值出现错误
            self.lowerEnvelope=loss
        neutral=(self.upperEnvelope+self.lowerEnvelope)/2

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
                        state['snapshot'] = p.data
                    
                    if self.t>=self.limit_snapshot_cycle and neutral<self.minNeutral: #不能在这里清零t,否则快照拍摄不完整
                        state['snapshot'] = p.data

                    #d=-p.grad.data*self.lr*(p.data.abs()+1)
                    d=-p.grad.data*self.lr*(p.data.abs()*(loss-self.minLoss)+self.lr_supplement)
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

                    if neutral>self.snapshot_recovery_threshold*self.minNeutral:
                        state['speed'] = torch.zeros_like(p.grad.data)
                        p.grad.data = torch.zeros_like(p.grad.data)
                        p.data=state['snapshot']
                        t=0
                        self.upperEnvelope = self.upperEnvelopeCache
                        self.lowerEnvelope = self.lowerEnvelopeCache
                    
        if self.t>=self.limit_snapshot_cycle and neutral<self.minNeutral:
            t=0
        if neutral<self.minNeutral:
            self.minNeutral=neutral
        self.upperEnvelopeCache = self.upperEnvelope
        self.lowerEnvelopeCache = self.lowerEnvelope

        return loss



