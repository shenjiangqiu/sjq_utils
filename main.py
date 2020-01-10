# %%
import math
import torch
import time
# Our module!
import norm_cuda



# %%
a=torch.tensor([[1.0,3,4,5],[1,2,1,2],[1,1,1,1],[2,2,2,2]],dtype=torch.float32)
print(a)
device=torch.device('cuda')
a=a.to(device)
b=torch.tensor([[0.0,0.0],[0,0]],dtype=torch.float32)
b=b.to(device)
print(b)
norm_cuda.norm(a,b,2,2)
print(b)

# %%
