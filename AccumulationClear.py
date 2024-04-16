import math
import random
import sys
import torch
from torch.optim.optimizer import Optimizer, required

#此优化器的主要超参数调节方法
#1.

class AccumulationClear(Optimizer):
    def __init__(self, params, lr=1e-2, smooth=10, lr_supplement=1e-2, soft_start=None, weight_decay=0, decay_flexibility=1, smoots_decrement=1e-2, snapshot_recovery_threshold=100, limit_snapshot_cycle=100, lossNeverNegative=True, deburr=False, nap=False, eps=1e-18):
        #parameter checks
        if lr<0: #学习速率
            raise ValueError(f'Invalid 学习速率: {lr}')
        if smooth is not None and smooth<0: #平滑程度
            raise ValueError(f'Invalid 平滑程度: {smooth}')
        if lr_supplement is not None and (lr_supplement<0): #学习率补充
            raise ValueError(f'Invalid 学习率补充: {lr_supplement}')
        if soft_start is not None and (soft_start<=0): #软起动,这个参数将乘在lr上，且每运行一次开一次方，越来越接近1.初始值越小，启动越软
            raise ValueError(f'Invalid 软起动: {soft_start}')
        if weight_decay<0 or weight_decay>=1: #权重衰减
            raise ValueError(f'Invalid 权重衰减: {weight_decay}')
        if decay_flexibility<0: #权重衰减柔性
            raise ValueError(f'Invalid 权重衰减柔性: {decay_flexibility}')
        if smoots_decrement<0: #平滑衰减量,对loss取包络时使用,约小越平滑
            raise ValueError(f'Invalid 平滑衰减量: {smoots_decrement}')
        if snapshot_recovery_threshold<=1: #快照恢复阈值,当前平滑loss大于平滑loss的最小值的这个倍数时恢复快照
            raise ValueError(f'Invalid 快照恢复阈值: {snapshot_recovery_threshold}')
        if limit_snapshot_cycle<1: #快照周期下限
            raise ValueError(f'Invalid 快照周期下限: {limit_snapshot_cycle}')
        if deburr and nap: #去除毛刺和加强毛刺
            raise ValueError(f'Invalid 同时加强毛刺和去除毛刺是无效的:deburr and nap')
        if eps<=0 or eps>=1: 
            raise ValueError(f'Invalid eps:{eps}')

        if lr_supplement is None:
            lr_supplement=lr
        if soft_start is None:
            soft_start=smooth

        defaults = dict(lr=lr, smooth=smooth, lr_supplement=lr_supplement, soft_start=soft_start, weight_decay=weight_decay, decay_flexibility=decay_flexibility, smoots_decrement=smoots_decrement, snapshot_recovery_threshold=snapshot_recovery_threshold, limit_snapshot_cycle=limit_snapshot_cycle, lossNeverNegative=lossNeverNegative, deburr=deburr, eps=eps)
        super(AccumulationClear, self).__init__(params,defaults)

        self.lr = lr
        self.smooth = smooth
        self.lr_supplement=lr_supplement
        self.soft_start=soft_start
        self.weight_decay = weight_decay
        self.decay_flexibility=decay_flexibility
        self.smoots_decrement=smoots_decrement
        self.snapshot_recovery_threshold=snapshot_recovery_threshold
        self.limit_snapshot_cycle=limit_snapshot_cycle
        self.minLoss = 0
        self.maxLoss = 1
        self.upperEnvelope = -2**15
        self.lowerEnvelope = 2**15
        self.upperEnvelopeCache = -2**15
        self.lowerEnvelopeCache = 2**15
        self.minNeutral = 2**15 #上包络与下包络的平均值的最小值
        self.t=0 #计数器,拍摄快照后清零
        self.maxAbsData=1
        self.lossNeverNegative=lossNeverNegative #声明Loss值域是否非负
        #可以用新旧值差异绝对值乘以self.smooth以抑制毛刺.但对于毛糙的曲面,这样可能会妨碍对全局最优解的寻找
        self.deburr=deburr #是否去毛刺
        self.nap=nap #是否加强毛刺
        self.eps=eps
        

    def __setstate__(self, state):
        super(AccumulationClear, self).__setstate__(state)

    def step(self, closure=None, loss=None):
        if closure is not None:
            loss = closure()
            loss.backward()
        if loss is None:
            loss=torch.tensor(1)
        
        k=torch.atan(loss*math.pi/2)*2/math.pi
        b=torch.atan(loss/self.lr_supplement*math.pi/2)*2/math.pi
        if loss<0 or not self.lossNeverNegative:
            k=(k+1)/2
            b=(b+1)/2
        k=1-torch.sqrt(1-k**2)    
        b=1-torch.sqrt(1-b**2)    
        s=1-1/(self.t/self.soft_start+1)
        k=k*self.lr*s
        b=b*self.lr*s*self.lr_supplement
        

        self.t+=1

        self.upperEnvelope=self.upperEnvelope-self.smoots_decrement
        self.lowerEnvelope=self.lowerEnvelope+self.smoots_decrement
        if loss>self.upperEnvelope:
            self.upperEnvelope=loss.clone()
        if loss<self.lowerEnvelope: #前面不能加else,不然因为两者的自动衰减会导致值出现错误
            self.lowerEnvelope=loss.clone()
        neutral=(self.upperEnvelope+self.lowerEnvelope)/2

        if neutral<self.snapshot_recovery_threshold*self.minNeutral and loss.isfinite():
            if loss<self.minLoss:
                self.minLoss=loss.clone()
            if loss>self.maxLoss:
                self.maxLoss=loss.clone()
        maxAbsData0=0
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
                        state['flat'] = torch.zeros_like(p.grad.data)
                        state['absFlat'] = torch.zeros_like(p.grad.data)
                        state['snapshot'] = p.data.clone()
                    
                    if self.t>=self.limit_snapshot_cycle and neutral<self.minNeutral: #不能在这里清零t,否则快照拍摄不完整
                        state['snapshot'] = p.data.clone()

                    maxAbsData0=max(maxAbsData0,p.data.abs().max())
                    deburr=1
                    if self.deburr:
                        deburr=(state['flat']-p.grad.data).abs()
                    nap=1
                    if self.nap:
                        nap=(state['flat']-p.grad.data).abs()
                    #可以用新旧值差异绝对值乘以self.smooth以抑制毛刺.但对于毛糙的曲面,这样可能会妨碍对全局最优解的寻找
                    state['flat']=(state['flat']*self.smooth*deburr+p.grad.data*nap)/(self.smooth*deburr+nap)
                    state['flat'][~torch.isfinite(state['flat'])]=0
                    state['absFlat']=(state['absFlat']*self.smooth*deburr+p.grad.data.abs()*nap)/(self.smooth*deburr+nap)
                    state['absFlat'][~torch.isfinite(state['absFlat'])]=0
                    
                    #d=-p.grad.data*self.lr*(p.data.abs()+1)
                    #d=-p.grad.data*self.lr*((p.data.abs()/self.maxAbsData)*(loss-self.minLoss)/(self.maxLoss-self.minLoss)+self.lr_supplement)
                    #d=-p.grad.data*self.lr*((p.data.abs()/self.maxAbsData)+self.lr_supplement)
                    #d=-p.grad.data.sign()*torch.rand_like(p.grad.data)*k*self.lr*((p.data.abs()/self.maxAbsData)+self.lr_supplement)
                    rand=torch.rand_like(p.grad.data)
                    rand=1-(1-(rand**((state['absFlat'].abs()+self.eps)/(state['flat'].abs()+self.eps))))**((state['flat'].abs()+self.eps)/(state['absFlat'].abs()+self.eps))
                    rand.clamp_(min=0,max=1)
                    d=-state['flat'].sign()*rand*(k*(p.data.abs()/self.maxAbsData)+b)
                    d[~torch.isfinite(d)]=0
                    
                    #对使参数远离0的训练,进行抑制,实现权重衰减.
                    decayMask=d*p.data>0
                    x=d[decayMask]
                    d[decayMask]+=(-2*self.weight_decay*x*(x/self.decay_flexibility).atan().abs())/math.pi
                    #state['speed'][decayMask]*=(1-self.weight_decay)
                    d[~torch.isfinite(d)]=0

                    #置当前步跨越0的参数和对应的平滑梯度为0,
                    #p.data.add_(state['speed'])
                    buffer=p.data.add(d)
                    clearMask=buffer*p.data<0
                    buffer[clearMask]=0
                    p.data=buffer.data
                    state['flat'][clearMask]=0

                    if neutral>self.snapshot_recovery_threshold*self.minNeutral or not loss.isfinite():
                        state['flat'] = torch.zeros_like(p.grad.data)
                        state['absFlat'] = torch.zeros_like(p.grad.data)
                        p.grad.data = torch.zeros_like(p.grad.data)
                        p.data=state['snapshot'].clone()
                        
        self.maxAbsData=maxAbsData0                
                    
        if self.t>=self.limit_snapshot_cycle and neutral<self.minNeutral:
            self.t=0
        if neutral<self.minNeutral:
            self.minNeutral=neutral.clone()
        if neutral>self.snapshot_recovery_threshold*self.minNeutral or not loss.isfinite():
            self.t=0
            self.upperEnvelope = self.upperEnvelopeCache.clone()
            self.lowerEnvelope = self.lowerEnvelopeCache.clone()

        self.upperEnvelopeCache = self.upperEnvelope.clone()
        self.lowerEnvelopeCache = self.lowerEnvelope.clone()

        return loss



