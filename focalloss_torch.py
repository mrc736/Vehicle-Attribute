from torch import nn
import torch
from torch.nn import functional as F

class focal_loss(nn.Module):
    '''
    Focal Loss : FL=-a(1-p)^r*log(p)

    '''

    def __init__(self,alpha=2,gamma=2,num_classes=5,size_average=True):
        super(focal_loss,self).__init__()
        self.alpha=alpha
        self.gamma=gamma
        self.num_classes=num_classes
        self.size_average=size_average
    def forward(self,inputs,targets):
        N=inputs.size(0)
        C=inputs.size(1)
        P=F.softmax(inputs)

        mask=inputs.data.new(N,C).fill_(0)  #生成和input一样的shape的tensor
        mask=mask.requires_grad_() #加入梯度更新计算
        ids=targets.view(-1,1) #获取目标的索引
        mask.data.scatter_(1,ids.data,1.) #使用Scatter将索引赋值为mask，生成one-hot数据
        probs=(P*mask).sum(1).view(-1,1)
        print(probs.shape)
        log_p=probs.log()  #log_p
        loss=torch.pow((1-probs),self.gamma)*log_p
        batch_loss=-self.alpha*loss.t() #loss.t()--->t的转置

        if self.size_average:
            loss=batch_loss.mean()
        else:
            loss=batch_loss.sum()
        print('focal loss值为:',loss)
        return loss

# torch.manual_seed(50)
# inputs=torch.randn(5,5,dtype=torch.float32,requires_grad=True)
# print('inputs: ',inputs)
# targets=torch.randint(5,(5,))
# print('targets: ',targets)
#
# criterion=focal_loss()
# loss=criterion(inputs,targets)
# loss.backward()



