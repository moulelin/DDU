import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import normalize, conv_transpose2d, conv2d
import torch
a = nn.Conv2d(3,3, 3)
print(a.parameters())
print(a._parameters)
print(a._parameters["weight"].shape)
strides = [1, 1, 2, 2]
input_sizes = 32 // np.cumprod(strides)
print(input_sizes)
v = normalize(torch.randn(20), dim=0, eps=1e-12)
print(v)