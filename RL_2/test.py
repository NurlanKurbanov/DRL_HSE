import torch
from torch import nn
from torch.distributions import Normal

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(DEVICE)
# sigma = nn.Parameter(torch.zeros(5))
# sigma = torch.exp(sigma).unsqueeze(0)
# sigma = 1
# m = torch.randint(0,5,size=(3,4))
# s = torch.randint(5,10,size=(3,4))
# d = Normal(m,s)
#
# a = torch.tensor([1,2,3, 4])
# p = d.log_prob(a)
# p = 1
# m = nn.Sequential(
#             nn.Linear(4, 256),
#             nn.ELU(),
#             nn.Linear(256, 256),
#             nn.ELU(),
#             nn.Linear(256, 6)
#         ).to(DEVICE)
#
# x = torch.randint(0,10,size=(3,4), dtype=torch.float32).to(DEVICE)
# xx = m(x)
# print(torch.get_device(xx))