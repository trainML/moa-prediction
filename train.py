import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold  # trainML
except:
    sys.path.append("../input/iterative-stratification")  # kaggle
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


class MoaDataset(Dataset):
    def __init__(self, features, targets, mode="train"):
        self.mode = mode
        self.data = features
        if mode == "train":
            self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == "train":
            return torch.FloatTensor(self.data[idx]), torch.FloatTensor(
                self.targets[idx]
            )
        elif self.mode == "eval":
            return torch.FloatTensor(self.data[idx]), 0


def train(
    model, device, X, Y, n_splits=10, train_test_split=0.25, batch_size=4096, epochs=50
):
    kfold = MultilabelStratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_losses = []
    val_losses = []

    for n, (tr, te) in enumerate(kfold.split(X, Y)):
        X_train, X_val = X[tr], X[te]
        y_train, y_val = Y[tr], Y[te]

        train_dataset = MoaDataset(X_train, y_train)
        val_dataset = MoaDataset(X_val, y_val)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=False
        )
        split_train_losses, split_val_losses = model.batch_gd(
            model, criterion, optimizer, train_loader, val_loader, epochs
        )
        print(
            f"Fold {n}, final train loss: {split_train_losses[epochs-1]:5.5f}, final train loss: {split_val_losses[epochs-1]:5.5f}"
        )
        train_losses += split_train_losses
        val_losses += split_val_losses

    model.save("latest_model")
