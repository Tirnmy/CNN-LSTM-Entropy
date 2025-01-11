import time

import torch
from torchviz import make_dot

from mymodels import CNNLSTMEntropy

model = torch.load('./tmp/modelCNNLSTMEntropy/model_11.pth')
model.to('cuda:0')
print(model)
x = torch.randn(1, 70, 7)
e1 = torch.randn(1, 1, 7)
e2 = torch.randn(1, 1, 7)
x = x.to('cuda:0')
e1 = e1.to('cuda:0')
e2 = e2.to('cuda:0')
outputs = model(x, e1, e2)
g = make_dot(outputs)  # 实例化 make_dot
g.view(f'./tmp/modelCNNLSTMEntropy/{time.strftime("%Y%m%d%H%M%S", time.localtime())}')  # 直接在当前路径下保存 pdf 并打开
# g.render(filename='netStructure/myNetModel', view=False, format='pdf')  # 保存 pdf 到指定路径不打开