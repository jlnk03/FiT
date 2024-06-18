import models
import torch
from models import FiT
import timm

Testtensor = torch.randn(50, 256, 224)
Testmask = torch.randn(50, 256)


Testtensor[:, -50:, :] = 0
Testmask[:, -50:] = False

output = FiT.select_random_tokens(Testtensor,5,Testmask)
print (output)

