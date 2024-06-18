import torch
import torch
import timm

def select_random_tokens( x: torch.Tensor, selection: int, mask: torch.Tensor) -> torch.Tensor:
        indices = torch.randperm(x.shape[1])

        x_shuffled = x[:, indices, :]
        mask_shuffled = mask[:, indices]

        mask_sorted = torch.argsort(mask_shuffled, dim=1, descending=True)

        x_sorted = x_shuffled[:, mask_sorted, :]

        return x_sorted[:selection]

def select_random_tokens2( x: torch.Tensor, selection: int, mask: torch.Tensor) -> torch.Tensor:
        mask_lengths = torch.sum(mask, dim=1).int()
        

        
Testtensor = torch.randn(3, 256, 10)
Testmask = torch.randn(3, 256)


Testtensor[:, -50:, :] = 0
Testmask[:, -50:] = False

output = select_random_tokens(Testtensor,5,Testmask)
print (output)

