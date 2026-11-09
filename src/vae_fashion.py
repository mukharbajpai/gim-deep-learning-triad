"""Convolutional VAE, Fashion-MNIST; reports reconstruction and KL losses."""
import argparse
import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .common import (DATA, OUTPUTS, RESULTS, dataloader_generator, device,
                     seed_everything, seed_worker, write_metrics)


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(), nn.Flatten())
        self.mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.logvar = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.dec = nn.Sequential(nn.Unflatten(1, (64, 7, 7)),
                                 nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
                                 nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid())

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return self.dec(self.fc(z)), mu, logvar


class ConvAE(nn.Module):
    """Parameter-matched deterministic autoencoder reconstruction baseline."""
    def __init__(self, latent_dim=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(), nn.Flatten(),
                                 nn.Linear(64 * 7 * 7, latent_dim))
        self.dec = nn.Sequential(nn.Linear(latent_dim, 64 * 7 * 7),
                                 nn.Unflatten(1, (64, 7, 7)),
                                 nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
                                 nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid())

    def forward(self, x):
        return self.dec(self.enc(x))


def loss_terms(recon, x, mu, logvar):
    # BCE summed per pixel, normalized per sample. KL normalized per sample.
    rec = F.binary_cross_entropy(recon, x, reduction='sum') / x.shape[0]
    kl = -0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp()) / x.shape[0]
    return rec, kl


def run(fast=False, epochs=None, latent_dim=16, beta=1.0, seed=42):
    overall_start = time.time()
    seed_everything(seed)
    tr = datasets.FashionMNIST(DATA, train=True, transform=transforms.ToTensor(), download=True)
    te = datasets.FashionMNIST(DATA, train=False, transform=transforms.ToTensor(), download=True)
    # Fixed disjoint train/validation split from training data; test untouched.
    perm = torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed))
    ntrain, nval, ntest = (2000, 400, 500) if fast else (50000, 10000, 10000)
    train = Subset(tr, perm[:ntrain].tolist())
    val = Subset(tr, perm[ntrain:ntrain+nval].tolist())
    test = Subset(te, list(range(ntest)))
    assert set(train.indices).isdisjoint(val.indices)
    loader_kwargs = {'num_workers': 0, 'worker_init_fn': seed_worker,
                     'pin_memory': torch.cuda.is_available()}
    train_loader = DataLoader(train, batch_size=128, shuffle=True,
                              generator=dataloader_generator(seed), **loader_kwargs)
    val_loader = DataLoader(val, batch_size=256, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test, batch_size=256, shuffle=False, **loader_kwargs)
    model = ConvVAE(latent_dim).to(device())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = epochs if epochs is not None else (2 if fast else 12)
    history = []
    def evaluate(loader):
        model.eval()
        totals = torch.zeros(4)
        n = 0
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device())
                r, mu, lv = model(x)
                rec, kl = loss_terms(r, x, mu, lv)
                mse = F.mse_loss(r, x, reduction='sum') / (x.shape[0] * x[0].numel())
                totals += torch.tensor([float(rec), float(kl), float(rec + beta*kl),
                                        float(mse)]) * x.shape[0]
                n += x.shape[0]
        return [round(float(v / n), 5) for v in totals]
    for epoch in range(n_epochs):
        model.train()
        total = 0.
        for x, _ in train_loader:
            x = x.to(device())
            opt.zero_grad()
            recon, mu, lv = model(x)
            rec, kl = loss_terms(recon, x, mu, lv)
            loss = rec + beta * kl
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        validation = evaluate(val_loader)
        history.append({'epoch': epoch + 1, 'train_loss_per_image': round(total / len(train), 5),
                        'val_reconstruction_bce_per_image': validation[0],
                        'val_kl_per_image': validation[1], 'val_loss_per_image': validation[2],
                        'val_mse_per_pixel': validation[3]})
        print(history[-1])
    test_rec, test_kl, test_loss, test_mse = evaluate(test_loader)

    # Train an explicit deterministic baseline on exactly the same split.
    seed_everything(seed + 1)
    baseline_loader = DataLoader(train, batch_size=128, shuffle=True,
                                 generator=dataloader_generator(seed), **loader_kwargs)
    baseline = ConvAE(latent_dim).to(device())
    baseline_opt = torch.optim.Adam(baseline.parameters(), lr=1e-3)
    baseline_history = []
    def evaluate_baseline(loader):
        baseline.eval()
        total_bce = total_mse = 0.
        n = 0
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device())
                reconstruction = baseline(x)
                total_bce += float(F.binary_cross_entropy(
                    reconstruction, x, reduction='sum'))
                total_mse += float(F.mse_loss(reconstruction, x, reduction='sum'))
                n += x.shape[0]
        return round(total_bce / n, 5), round(total_mse / (n * 28 * 28), 5)
    for epoch in range(n_epochs):
        baseline.train()
        total = 0.
        for x, _ in baseline_loader:
            x = x.to(device())
            baseline_opt.zero_grad()
            reconstruction = baseline(x)
            loss = F.binary_cross_entropy(reconstruction, x, reduction='sum') / x.shape[0]
            loss.backward()
            baseline_opt.step()
            total += loss.item() * len(x)
        val_bce, val_mse = evaluate_baseline(val_loader)
        baseline_history.append({'epoch': epoch + 1,
                                 'train_bce_per_image': round(total / len(train), 5),
                                 'val_bce_per_image': val_bce,
                                 'val_mse_per_pixel': val_mse})
        print({'ae_baseline': baseline_history[-1]})
    baseline_test_bce, baseline_test_mse = evaluate_baseline(test_loader)

    model.eval()
    baseline.eval()
    batch_imgs, _ = next(iter(test_loader))
    with torch.no_grad():
        rec, _, _ = model(batch_imgs[:8].to(device()))
        baseline_rec = baseline(batch_imgs[:8].to(device()))
    fig, axes = plt.subplots(3, 8, figsize=(12, 5.2))
    for i in range(8):
        axes[0, i].imshow(batch_imgs[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[1, i].imshow(rec[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
        axes[2, i].imshow(baseline_rec[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
        for j in range(3): axes[j, i].axis('off')
    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('VAE')
    axes[2, 0].set_ylabel('AE baseline')
    fig.tight_layout()
    fig.savefig(RESULTS / 'vae_reconstructions.png', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([h['epoch'] for h in history], [h['train_loss_per_image'] for h in history], label='train loss')
    ax.plot([h['epoch'] for h in history], [h['val_loss_per_image'] for h in history], label='validation loss')
    ax.plot([h['epoch'] for h in baseline_history],
            [h['val_bce_per_image'] for h in baseline_history], '--', label='AE baseline val BCE')
    ax.set(xlabel='Epoch', ylabel='ELBO loss / image', title='VAE training')
    ax.legend(); fig.tight_layout(); fig.savefig(RESULTS / 'vae_training.png', dpi=170); plt.close(fig)
    torch.save({'state_dict': model.state_dict(), 'latent_dim': latent_dim}, OUTPUTS / 'fashion_vae.pt')
    torch.save({'state_dict': baseline.state_dict(), 'latent_dim': latent_dim},
               OUTPUTS / 'fashion_autoencoder_baseline.pt')
    metrics = {'experiment': 'fashion_mnist_vae', 'mode': 'fast' if fast else 'full',
               'seed': seed, 'train_size': ntrain, 'validation_size': nval,
               'test_size': ntest, 'epochs': n_epochs, 'latent_dim': latent_dim,
               'beta': beta, 'test_reconstruction_bce_per_image': test_rec,
               'test_kl_per_image': test_kl, 'test_elbo_loss_per_image': test_loss,
               'test_mse_per_pixel': test_mse,
               'deterministic_ae_baseline_seed': seed + 1,
               'deterministic_ae_baseline_test_bce_per_image': baseline_test_bce,
               'deterministic_ae_baseline_test_mse_per_pixel': baseline_test_mse,
               'split_protocol': ('seeded permutation of original 60k train; adjacent, '
                                  'non-overlapping train/validation subsets; original test untouched'),
               'compute_device': str(device()),
               'elapsed_seconds': round(time.time()-overall_start, 2),
               'history': history, 'baseline_history': baseline_history}
    write_metrics('vae', metrics)
    return metrics


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fast', action='store_true')
    p.add_argument('--epochs', type=int)
    p.add_argument('--latent-dim', type=int, default=16)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    print(json.dumps(run(a.fast, a.epochs, a.latent_dim, a.beta, a.seed), indent=2))
