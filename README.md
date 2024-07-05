# AccumulationClear
*此优化器是各种想法的黏合体,而Play优化器更适合真正使用
优化器,使用独特的方法为每个参数分别提供自适应更新速度.
具备非线性的"权重衰减",阻碍参数远离零,但不主动使参数靠近零,因此不会破坏优化.效果弱于正常权重衰减及L1、L2正则化.
调用优化器时传入测试损失可实现:
快照功能,在训练发散到一定程度后自动恢复过往最优参数
根据传入损失应用内置的学习率曲线,因此可能需要在传入损失前对损失值进行适当缩放.

The optimizer uses a unique method to provide an adaptive update speed for each parameter individually.
It has a nonlinear "weight decay", which prevents the parameters from moving away from zero, but does not actively make the parameters close to zero, so it will not destroy the optimization, and the effect is weaker than the normal weight decay and L1 and L2 regularization.
When the optimizer is invoked, the incoming test loss achieves:
The snapshot function automatically restores the past optimal parameters after the training divergence reaches a certain level.
The built-in learning rate curve is applied based on the incoming loss, so it may be necessary to scale the loss value appropriately before the incoming loss.
@PyTorch
