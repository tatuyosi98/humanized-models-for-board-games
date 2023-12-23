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


def torch_log(x: torch.Tensor) -> torch.Tensor:
    """ torch.log(0)によるnanを防ぐ． """
    return torch.log(torch.clamp(x, min=1e-10))

class VAE_linear(nn.Module):
    """ VAEモデルの実装 """
    def __init__(self, input_dim = 2 * 19 * 19, z_dim = 2) -> None:
        """
        クラスのコンストラクタ．

        Parameters
        ----------
        z_dim : int
            VAEの潜在空間の次元数．
        """
        super().__init__()        

        # Encoder, xを入力にガウス分布のパラメータmu, sigmaを出力
        self.dense_enc1 = nn.Linear(input_dim, 200)
        self.dense_enc2 = nn.Linear(200, 200)
        self.dense_encmean = nn.Linear(200, z_dim)
        self.dense_encvar = nn.Linear(200, z_dim)

        # Decoder, zを入力にベルヌーイ分布のパラメータlambdaを出力
        self.dense_dec1 = nn.Linear(z_dim, 200)
        self.dense_dec2 = nn.Linear(200, 200)
        self.dense_dec3 = nn.Linear(200, input_dim)

    # (以下、forwardメソッドとその他のメソッドを定義)


    def _encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        VAEのエンコーダ部分．

        Parameters
        ----------
        x : torch.Tensor ( b, c * h * w )
            Flattenされた入力画像．

        Returns
        ----------
        mean : torch.Tensor ( b, z_dim )
            エンコーダがモデリングするガウス分布の平均
        std : torch.Tensor ( b, z_dim )
            エンコーダがモデリングするガウス分布の標準偏差
        """
        x = x.view(-1, 2 * 19 * 19)
        x = F.relu(self.dense_enc1(x))
        x = F.relu(self.dense_enc2(x))
        mean = self.dense_encmean(x)
        std = F.softplus(self.dense_encvar(x))

        return mean, std

    def _sample_z(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        訓練時に再パラメータ化トリックによってガウス分布から潜在変数をサンプリングする．
        推論時はガウス分布の平均を返す．

        Parameters
        ----------
        mean : torch.Tensor ( b, z_dim )
            エンコーダがモデリングするガウス分布の平均
        std : torch.Tensor ( b, z_dim )
            エンコーダがモデリングするガウス分布の標準偏差
        """
        if self.training:
            epsilon = torch.randn(mean.shape).to(device)
            return mean + epsilon * std
        else:
            return mean

    def _decoder(self, z: torch.Tensor) -> torch.Tensor:
        """
        VAEのデコーダ部分．

        Parameters
        ----------
        z : torch.Tensor ( b, z_dim )
            潜在変数．

        Returns
        ----------
        x : torch.Tensor ( b, c * h * w )
            再構成画像．
        """
        x = F.relu(self.dense_dec1(z))
        x = F.relu(self.dense_dec2(x))
        # 出力が0~1になるようにsigmoid
        x = torch.sigmoid(self.dense_dec3(x))
        x = x.view(-1, 2, 19, 19) 

        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        順伝播．

        Parameters
        ----------
        x : torch.Tensor ( b, c * h * w )
            Flattenされた入力画像．

        Returns
        ----------
        x : torch.Tensor ( b, c * h * w )
            再構成画像．
        z : torch.Tensor ( b, z_dim )
            潜在変数．
        """
        mean, std = self._encoder(x)
        z = self._sample_z(mean, std)
        x = self._decoder(z)
        return x, z

    def loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        順伝播しつつ目的関数の計算を行う．

        Parameters
        ----------
        x : torch.Tensor ( b, c * h * w )
            Flattenされた入力画像．

        Returns
        ----------
        KL : torch.Tensor (, )
            正則化項．エンコーダ（ガウス分布）と事前分布（標準ガウス分布）のKLダイバージェンス．
        reconstruction : torch.Tensor (, )
            再構成誤差．
        """
        mean, std = self._encoder(x)

        # KL loss(正則化項)の計算. mean, stdは (batch_size , z_dim)
        # torch.sumは上式のJ(=z_dim)に関するもの. torch.meanはbatch_sizeに関するものなので,
        # 上式には書いてありません.
        KL = -0.5 * torch.mean(torch.sum(1 + torch_log(std**2) - mean**2 - std**2, dim=1))

        z = self._sample_z(mean, std)
        y = self._decoder(z)

        # reconstruction loss(負の再構成誤差)の計算. x, yともに (batch_size , 784)
        # torch.sumは上式のD(=784)に関するもの. torch.meanはbatch_sizeに関するもの.
        # バイナリー・クロスエントロピー
        reconstruction = torch.mean(torch.sum(x * torch_log(y) + (1 - x) * torch_log(1 - y), dim=1))
        # reconstruction = F.binary_cross_entropy(y, x, reduction='mean').sum(dim=1).mean()

        return KL, -reconstruction