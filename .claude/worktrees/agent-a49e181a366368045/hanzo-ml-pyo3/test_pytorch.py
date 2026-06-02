import hanzo
import torch

# convert from hanzo tensor to torch tensor
t = hanzo.randn((3, 512, 512))
torch_tensor = t.to_torch()
print(torch_tensor)
print(type(torch_tensor))

# convert from torch tensor to hanzo tensor
t = torch.randn((3, 512, 512))
hanzo_tensor = hanzo.Tensor(t)
print(hanzo_tensor)
print(type(hanzo_tensor))
