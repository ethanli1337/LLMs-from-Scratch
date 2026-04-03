import torch 
import torch.nn as nn

torch.save(model, PATH)

model=torch.load(PATH)
model.eval()

torch.save(model.state_dict() PATH)

model=Model(*args,**kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
