import torch 

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
print(p.size())
print(o[1,1].item())
