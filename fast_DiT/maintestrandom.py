import torch
import torch
import timm

def select_random_tokens( x: torch.Tensor, selection: int, mask: torch.Tensor) -> torch.Tensor:
        indices = torch.randperm(x.shape[1])

        x_shuffled = x[:, indices, :]
        mask_shuffled = mask[:, indices]

        mask_sorted = torch.argsort(mask_shuffled, dim=1, descending=True)

        x_sorted = x_shuffled[mask_sorted]

        return x_sorted[:selection]

def select_random_tokens2( x: torch.Tensor, selection: int, mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, token_dim = input.shape
        assert selection <= num_tokens

        # Create a tensor of indices for each batch
        indices = torch.argsort(torch.rand(batch_size, num_tokens), dim=1)

        # Filter out indices where mask is False
        valid_indices = indices[:, mask.bool()]

        # Select the specified number of random tokens for each batch
        selected_indices = valid_indices[:, :selection]

        # Gather the selected tokens from the input tensor
        selected_tokens = torch.gather(input, 1, selected_indices.unsqueeze(-1).repeat(1, 1, token_dim))

        return selected_tokens


Testtensor = torch.randn(50, 256, 224)
Testmask = torch.randn(50, 256)


Testtensor[:, -50:, :] = 0
Testmask[:, -50:] = False

output = select_random_tokens(Testtensor,5,Testmask)
print (output)

