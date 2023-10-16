import copy

import torch
import numpy as np
from src.fedalg import FedAlg
from networks.metanet import MetaNN
class SignSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2):
        defaults = dict(lr=lr)
        super(SignSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-group['lr'], torch.sign(d_p))  # weight update with the sign of gradient

        return loss
class FedSgd(FedAlg):
    def __init__(self, criterion, model, config):
        super().__init__(criterion, model, half=config.half)
    
    def client_grad(self, x, y):

        out = self.model(x)
        loss = self.criterion(out, y)
        dy_dx = torch.autograd.grad(loss, self.model.parameters())

        if self.half:
            grad = list((_.detach().half().clone() for _ in dy_dx))
        else:
            grad = list((_.detach().clone() for _ in dy_dx))

        return grad


class FedAvg(FedAlg):
    def __init__(self, criterion, model, config):
        super().__init__(criterion, model, half=config.half)
        self.fed_lr = config.fed_lr
        self.tau = config.tau0
        self.strategy = config.strategy

        self.init_state = copy.deepcopy(self.model.state_dict())
    def client_grad(self, x, y):
        net_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.fed_lr)
        # net_optimizer = SignSGD(self.model.parameters(), lr=self.fed_lr)
        
        # 如果策略是“soteria”，执行特定的计算
        if self.strategy  == "soteria":
            print("Applying Soteria strategy...")
            x.requires_grad = True  # 确保x保留梯度
            out = self.model(x)
            feature_fc1_graph = self.model.extract_feature()
            deviation_f1_target = torch.zeros_like(feature_fc1_graph)
            deviation_f1_x_norm = torch.zeros_like(feature_fc1_graph)
            print(feature_fc1_graph.shape)
            for f in range(deviation_f1_x_norm.size(1)):
                deviation_f1_target[:, f] = 1
                feature_fc1_graph.backward(deviation_f1_target, retain_graph=True)
                deviation_f1_x = x.grad.data  # 使用x的梯度
                deviation_f1_x_norm[:, f] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1) / (feature_fc1_graph.data[:, f] + 0.1)
                self.model.zero_grad()
                x.grad.data.zero_()  # 清零x的梯度
                deviation_f1_target[:, f] = 0
            deviation_f1_x_norm_sum = deviation_f1_x_norm.sum(axis=0)
            print(deviation_f1_x_norm_sum.shape)
            thresh = np.percentile(deviation_f1_x_norm_sum.flatten().cpu().numpy(), 50)
            mask = np.where(abs(deviation_f1_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)
            # 之后，你可能需要将mask或其他计算结果应用于模型或梯度

        # 执行通常的梯度计算和更新步骤
        for t in range(self.tau):
            out = self.model(x)
            risk = self.criterion(out, y)
            net_optimizer.zero_grad()
            risk.backward()
            net_optimizer.step()
            if t == 1999:
                self.init_state = copy.deepcopy(self.model.state_dict())

        grad = []
        st = self.model.state_dict()
        for w_name, w_val in st.items():
            grad.append((self.init_state[w_name] - w_val)/self.fed_lr)
        # print(grad[-2].shape)
        # print(len(grad))
        if self.strategy  == "soteria":
            grad[-1] = grad[-1] * torch.Tensor(mask).to("cuda:0")

        return grad

    
    # def client_grad(self, x, y):
        

    #     net_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.fed_lr)
    #     for t in range(self.tau):
    #         out = self.model(x)
    #         risk = self.criterion(out, y)
    #         net_optimizer.zero_grad()
    #         risk.backward()
    #         net_optimizer.step()

    #     grad = []
    #     st = self.model.state_dict()
    #     for w_name, w_val in st.items():
    #         grad.append((self.init_state[w_name] - w_val)/self.fed_lr)

    #     return grad

