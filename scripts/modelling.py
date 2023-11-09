"""TODO: add description."""

import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import torch
import torchvision

class RoadModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def training_step(self, batch, batch_idx):
        inputs, target = batch

        logit_mask = self.model(inputs.float())
        loss = self.loss_fn(logit_mask, target)

        prob_mask = logit_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), target.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=1e-3)
