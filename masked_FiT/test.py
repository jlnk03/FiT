from torch.utils.data import DataLoader
import cProfile

from preprocess.iterators import ImageNetLatentIterator as N
from preprocess_new_old.iterators import ImageNetLatentIterator as NO
from preprocess_old.iterators import ImageNetLatentIterator as O


def n_dataloader():
    dataset = N({
        "latent_folder": '/storage/slurm/schnaus/gsu_2024/latentnew_train',
        "sample_size": 256,
        "patch_size": 2,
        "vae_scale": 8,
        "C": 4,
        "embed_dim": 16,
        "embed_method": "rotate"
    })
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=dataset.collate
    )
    return loader

def test_n_dataloader():
    n = iter(n_dataloader())

    for i in range(3000):
        next(n)


def o_dataloader():
    dataset = O({
        "latent_folder": '/storage/slurm/schnaus/gsu_2024/latentnew_train',
        "sample_size": 256,
        "patch_size": 2,
        "vae_scale": 8,
        "C": 4,
        "embed_dim": 16,
        "embed_method": "rotate"
    })
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    return loader


def test_o_dataloader():
    o = iter(o_dataloader())

    for i in range(3000):
        next(o)


def n_o_dataloader():
    dataset = NO({
        "latent_folder": '/storage/slurm/schnaus/gsu_2024/latentnew_train',
        "sample_size": 256,
        "patch_size": 2,
        "vae_scale": 8,
        "C": 4,
        "embed_dim": 16,
        "embed_method": "rotate"
    })
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=dataset.collate
    )
    return loader


def test_n_o_dataloader():
    n_o = iter(n_o_dataloader())

    for i in range(3000):
        next(n_o)


if __name__ == "__main__":
    cProfile.run('test_n_o_dataloader()')
    print('......................................................................')
    cProfile.run('test_o_dataloader()')
    print('......................................................................')
    cProfile.run('test_n_dataloader()')

