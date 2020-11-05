import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime


class MoaModel(nn.Module):
    def __init__(
        self,
        num_columns,
        model_state=None,
        layer1_outputs=2048,
        layer1_dropout=0.2,
        layer2_outputs=1048,
        layer2_dropout=0.5,
        layer3_enable=False,
        layer3_outputs=None,
        layer3_dropout=0.5,
        final_layer_dropout=0.5,
    ):
        super(MoaModel, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_columns)
        self.dropout1 = nn.Dropout(layer1_dropout)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_columns, layer1_outputs))

        self.batch_norm2 = nn.BatchNorm1d(layer1_outputs)
        self.dropout2 = nn.Dropout(layer2_dropout)
        self.dense2 = nn.utils.weight_norm(nn.Linear(layer1_outputs, layer2_outputs))

        self.layer3 = layer3_enable
        if self.layer3:
            self.batch_norm3 = nn.BatchNorm1d(layer2_outputs)
            self.dropout3 = nn.Dropout(layer3_dropout)
            self.dense3 = nn.utils.weight_norm(
                nn.Linear(layer2_outputs, layer3_outputs)
            )

        final_layer_inputs = layer3_outputs if self.layer3 else layer2_outputs
        self.batch_norm_final = nn.BatchNorm1d(final_layer_inputs)
        self.dropout_final = nn.Dropout(final_layer_dropout)
        self.dense_final = nn.utils.weight_norm(nn.Linear(final_layer_inputs, 206))

    def forward(self, X):
        X = self.batch_norm1(X)
        X = self.dropout1(X)
        X = F.relu(self.dense1(X))

        X = self.batch_norm2(X)
        X = self.dropout2(X)
        X = F.relu(self.dense2(X))

        if self.layer3:
            X = self.batch_norm3(X)
            X = self.dropout3(X)
            X = F.relu(self.dense3(X))

        X = self.batch_norm_final(X)
        X = self.dropout_final(X)
        X = F.sigmoid(self.dense_final(X))

        return X

    def _load_from_file(self, file):
        self.load_state_dict(torch.load(file))

    def save(self, file):
        torch.save(self.state_dict(), file)


def batch_gd(model, device, criterion, optimizer, train_loader, val_loader, epochs):
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    for it in range(epochs):
        t0 = datetime.now()

        model.train()
        train_loss = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item() / len(train_loader))

        train_loss = np.mean(train_loss)

        model.eval()
        val_loss = []
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss.append(loss.item() / len(val_loader))
        val_loss = np.mean(val_loss)

        train_losses[it] = train_loss
        val_losses[it] = val_loss

        dt = datetime.now() - t0
        print(
            f"Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Duration {dt}"
        )

    return train_losses, val_losses


def predict(model, device, data_loader):
    model.eval()
    preds = []

    for inputs, _ in data_loader:
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        preds.append(outputs.detach().cpu().numpy())

    preds = np.concatenate(preds)

    return preds