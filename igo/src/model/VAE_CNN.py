import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from typing import Tuple

rng = np.random.RandomState(1234)
random_state = 42
device = "mps" if torch.backends.mps.is_available() else "cpu"
# device = "cpu"



# input_dim = 2*19*19


class CNNEncoder(nn.Module):
    """ CNNベースのエンコーダ """
    def __init__(self, z_dim, h_dim):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 19 * 19, h_dim)
        self.fc_mean = nn.Linear(h_dim, z_dim)
        self.fc_var = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        mean = self.fc_mean(x)
        log_var = self.fc_var(x)
        return mean, log_var


class CNNDecoder(nn.Module):
    """ CNNベースのデコーダ """
    def __init__(self, z_dim, h_dim):
        super(CNNDecoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, 32 * 19 * 19)
        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(16, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        # z.size(0)はバッチサイズ
        z = z.view(z.size(0), 32, 19, 19)
        z = F.relu(self.deconv1(z))
        reconstruction = torch.sigmoid(self.deconv2(z))
        return reconstruction


class VAE_CNN(nn.Module):
    """ CNNベースのVAE """
    def __init__(self, z_dim=64, h_dim=128):
        super(VAE_CNN, self).__init__()
        self.encoder = CNNEncoder(z_dim, h_dim)
        self.decoder = CNNDecoder(z_dim, h_dim)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        return self.decoder(z), mean, log_var
    

    def loss(self, x):
        # モデルのforwardパスを使用して、再構成されたデータ、平均、対数分散を取得
        recon_x, mean, log_var = self.forward(x)

        # 再構成誤差（バイナリクロスエントロピー）
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # KLダイバージェンス
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return KLD, BCE  # KLダイバージェンスと再構成誤差を返す
