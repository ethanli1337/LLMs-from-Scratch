import torch 
import torch.nn as nn 

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear=nn.Linear(n_input_features,1)
    def forward(self, x):
        y_pred=torch.sigmoid(self.linear(x))
        return y_pred
    
model=Model(n_input_features=6)
learning_rate=0.01
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
print(optimizer.state_dict())

checkpoint={
    "epoch":90,
    "model_state":model.state_dict(),
    "optim_state":optimizer.state_dict()
}

# torch.save(checkpoint,"checkpoint.pth")
loaded_checkpoint=torch.load("checkpoint.pth")

# for param in model.parameters():
#     print(param)
# FILE="model.pth"
# #torch.save(model, FILE)
# torch.save(model.state_dict(),FILE)
# #model=torch.load(FILE,weights_only=False)
# #model.eval()



# loaded_model=Model(n_input_features=6)
# loaded_model.load_state_dict(torch.load(FILE))
# loaded_model.eval()
# for param in model.parameters():
#     print(param)