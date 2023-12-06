import logging

import torch

import cshogi

from dataloader import HcpeDataLoader
from features import feature_to_sfen

batch_size = 1024
device = torch.device(f"cuda:0")

"""
print("Reading training data")
train_dataloader = HcpeDataLoader(files="../preprocessed_data/train.hcpe",
                                  batch_size=batch_size,
                                  device=device, 
                                  shuffle=True)


print("Reading test data")
test_dataloader = HcpeDataLoader(files="../preprocessed_data/train.hcpe",
                                 batch_size=batch_size,
                                 device=device)
"""

test_dataloader = HcpeDataLoader(files="../preprocessed_data/initial_position.hcpe",
                                 batch_size=1,
                                 device=device)

mini_batch_data = test_dataloader.data[:1]

print(mini_batch_data)

features, move_label, result = test_dataloader.mini_batch(mini_batch_data)
print(features[0][0])
sfen = feature_to_sfen(features[0])
board = cshogi.Board(sfen)
print(board)