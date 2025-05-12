import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ------------------ Hyperparameters ------------------
batch_size = 64
epochs = 30
lr = 1e-3
latent_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Data ------------------
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ------------------ VAE ------------------
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 8x8
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 4x4
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),  # 32x32
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(self.fc_decode(z))
        return out, mu, logvar

# ------------------ Loss ------------------
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# ------------------ Training ------------------
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for x, _ in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)
        loss = vae_loss(recon_x, x, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.4f}")

# ------------------ Visualize ------------------
model.eval()
with torch.no_grad():
    x, _ = next(iter(train_loader))
    x = x[:8].to(device)
    recon_x, _, _ = model(x)

    fig, axs = plt.subplots(2, 8, figsize=(15, 4))
    for i in range(8):
        axs[0, i].imshow(x[i].permute(1, 2, 0).cpu())
        axs[1, i].imshow(recon_x[i].permute(1, 2, 0).cpu())
        axs[0, i].axis('off')
        axs[1, i].axis('off')
    axs[0, 0].set_title('Original')
    axs[1, 0].set_title('Reconstructed')
    plt.tight_layout()
    plt.show()
    
# Save the trained VAE model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': train_loss,
}, 'vae_cifar100.pth')