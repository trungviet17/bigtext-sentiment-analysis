import torch.nn.functional as F
from torchmetrics.classification import Accuracy, F1Score
import lightning as pl
import torch

class LSTMClassifierLightning(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super(LSTMClassifierLightning, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.accuracy = Accuracy(task="multiclass", num_classes=4)
        self.f1_score = F1Score(task="multiclass", num_classes=4, average='macro')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        f1 = self.f1_score(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1_score', f1, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        f1 = self.f1_score(preds, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', acc, on_epoch=True, prog_bar=True)
        self.log('val_f1_score', f1, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        f1 = self.f1_score(preds, y)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log('test_accuracy', acc, on_epoch=True, prog_bar=True)
        self.log('test_f1_score', f1, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss"
            }
        }

