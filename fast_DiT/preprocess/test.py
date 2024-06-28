import torch

from fast_DiT.preprocess.iterators import ImageNetLatentIterator as O
from fast_DiT.preprocess_old.iterators import ImageNetLatentIterator as N

if __name__ == "__main__":
     latent, label, pos, mask, height, width = next(iter(O({})))
     latent_, label_, pos_, mask_, height_, width_ = next(iter(N({})))

     print(torch.all(torch.eq(latent, latent_)))
     print(latent.shape)
     print(latent_.shape)
     print(torch.all(torch.eq(label, label_)))
     print(label.shape)
     print(label_.shape)
     print(torch.all(torch.eq(pos, pos_)))
     print(pos.shape)
     print(pos_.shape)
     print(torch.all(torch.eq(mask, mask_)))
     print(mask.shape)
     print(mask_.shape)
