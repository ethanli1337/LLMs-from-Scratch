import torch 
import numpy as np
x=torch.empty(2,2,2,3)
y=torch.randn(2,2)
z=torch.zeros(2,2,dtype=torch.float16)
c=torch.tensor([2.5,0.1])
c=torch.rand(2,2)

c=torch.rand(2,2)
d=torch.rand(2,2)
c.add_(d)
p=torch.mul(c,d)
o=torch.rand(5,3)
v=torch.rand(4,4)
p=v.view(-1,8)

a=torch.ones(5)
b=a.numpy()
a.add_(1)
a=np.ones(5)
b=torch.from_numpy(a)

a+=1

if torch.cuda.is_available():
    device=torch.device("cuda")
    x=torch.ones(5, device=device)
    y=torch.ones(5)
    y=y.to(device)
    z=x+y
    z=z.to("cpu")

x=torch.ones(5,requires_grad=True)
print(x)