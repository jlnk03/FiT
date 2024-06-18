import torch
import torch
import timm

# def select_random_tokens( x: torch.Tensor, selection: int, mask: torch.Tensor) -> torch.Tensor:
#         indices = torch.randperm(x.shape[1])

#         x_shuffled = x[:, indices, :]
#         mask_shuffled = mask[:, indices]

#         mask_sorted = torch.argsort(mask_shuffled, dim=1, descending=True)

#         x_sorted = x_shuffled[:, mask_sorted, :]

#         return x_sorted[:selection]

# def select_random_tokens2( x: torch.Tensor, selection: int, mask: torch.Tensor) -> torch.Tensor:
#         mask_lengths = torch.sum(mask, dim=1).int()
#         print(mask_lengths)
        
#         random_indices = torch.stack([torch.randint(0,length, (selection,)) for length in mask_lengths])
#         cut_x = torch.stack([torch.gather(x[i], 1, random_indices[i]) for i in range(x.shape[0])])
#         return cut_x
        
def subsample_tokens_no_loop(tokens, mask, subsample_size):
    batch_size = tokens.shape[0]
    num_relevant = mask.sum(dim=1)

    random_indices = torch.stack([torch.randperm(n)[:subsample_size] for n in num_relevant])

    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, subsample_size).to(tokens.device)
    
    subsampled_tokens = tokens[batch_indices, random_indices]

    return subsampled_tokens
        
        

        
Testtensor = torch.randn(3, 256, 10)
Testmask = torch.ones(3, 256, dtype=torch.bool)


Testtensor[:, -50:, :] = 0
Testmask[:, -50:] = False

#output = select_random_tokens2(Testtensor,5,Testmask)
#output= subsample_tokens_no_loop(Testtensor,Testmask,5)
#print (output.shape)

batch_size = 10
seq_len = 256
feature_dim = 16
tokens = torch.randn(batch_size, seq_len, feature_dim)  # Example token tensor
mask = torch.rand(batch_size, seq_len) > 0.5  # Example random mask

subsampled_output = subsample_tokens_no_loop(tokens, mask,32)
output = subsample_tokens_no_loop(Testtensor,Testmask,32)

print("subsampled=" + str(subsampled_output.shape) +"  \n")  # Should print: torch.Size([10, 32, 16])
print ("normal ="   + str(output.shape))  # Should print: torch.Size([10, 32, 16])