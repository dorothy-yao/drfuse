import math
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchmetrics.functional.classification import multilabel_average_precision, multilabel_auroc

import lightning.pytorch as pl

from .drfuse import DrFuseModel


class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='none', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor, masks):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log())).sum() / max(1e-6, masks.sum())


class DrFuseTrainer(pl.LightningModule):
    def __init__(self, args, label_names):
        super().__init__()
        self.model = DrFuseModel(hidden_size=args.hidden_size,
                                 num_classes=len(label_names),
                                 ehr_dropout=args.dropout,
                                 ehr_n_head=args.ehr_n_head,
                                 ehr_n_layers=args.ehr_n_layers)

        self.save_hyperparameters(args)  # args goes to self.hparams

        self.pred_criterion = nn.BCELoss(reduction='none')
        self.alignment_cos_sim = nn.CosineSimilarity(dim=1)
        self.triplet_loss = nn.TripletMarginLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.jsd = JSD()

        # self.val_preds = []
        self.val_preds = {k: [] for k in ['final', 'ehr', 'cxr']}
        self.val_labels = []
        self.val_pairs = []

        self.test_preds = []
        self.test_labels = []
        self.test_pairs = []
        self.test_feats = {k: [] for k in ['feat_ehr_shared', 'feat_ehr_distinct',
                                           'feat_cxr_shared', 'feat_cxr_distinct']}
        self.test_attns = []

        self.label_names = label_names

    def _compute_masked_pred_loss(self, input, target, mask):
        return (self.pred_criterion(input, target).mean(dim=1) * mask).sum() / max(mask.sum(), 1e-6)

    def _masked_abs_cos_sim(self, x, y, mask):
        return (self.alignment_cos_sim(x, y).abs() * mask).sum() / max(mask.sum(), 1e-6)

    def _masked_cos_sim(self, x, y, mask):
        return (self.alignment_cos_sim(x, y) * mask).sum() / max(mask.sum(), 1e-6)

    def _masked_mse(self, x, y, mask):
        return (self.mse_loss(x, y).mean(dim=1) * mask).sum() / max(mask.sum(), 1e-6)

    def _disentangle_loss_jsd(self, model_output, pairs, log=True, mode='train'):
        ehr_mask = torch.ones_like(pairs)
        loss_sim_cxr = self._masked_abs_cos_sim(model_output['feat_cxr_shared'],
                                                model_output['feat_cxr_distinct'], pairs)
        loss_sim_ehr = self._masked_abs_cos_sim(model_output['feat_ehr_shared'],
                                                model_output['feat_ehr_distinct'], ehr_mask)

        jsd = self.jsd(model_output['feat_ehr_shared'].sigmoid(),
                       model_output['feat_cxr_shared'].sigmoid(), pairs)

        loss_disentanglement = (self.hparams.lambda_disentangle_shared * jsd +
                                self.hparams.lambda_disentangle_ehr * loss_sim_ehr +
                                self.hparams.lambda_disentangle_cxr * loss_sim_cxr)
        if log:
            self.log_dict({
                f'disentangle_{mode}/EHR_disinct': loss_sim_ehr.detach(),
                f'disentangle_{mode}/CXR_disinct': loss_sim_cxr.detach(),
                f'disentangle_{mode}/shared_jsd': jsd.detach(),
                'step': float(self.current_epoch)
            }, on_epoch=True, on_step=False, batch_size=pairs.shape[0])

        return loss_disentanglement

    def _compute_prediction_losses(self, model_output, y_gt, pairs, log=True, mode='train'):
        ehr_mask = torch.ones_like(model_output['pred_final'][:, 0])
        loss_pred_final = self._compute_masked_pred_loss(model_output['pred_final'], y_gt, ehr_mask)
        loss_pred_ehr = self._compute_masked_pred_loss(model_output['pred_ehr'], y_gt, ehr_mask)
        loss_pred_cxr = self._compute_masked_pred_loss(model_output['pred_cxr'], y_gt, pairs)
        loss_pred_shared = self._compute_masked_pred_loss(model_output['pred_shared'], y_gt, ehr_mask)

        if log:
            self.log_dict({
                f'{mode}_loss/pred_final': loss_pred_final.detach(),
                f'{mode}_loss/pred_shared': loss_pred_shared.detach(),
                f'{mode}_loss/pred_ehr': loss_pred_ehr.detach(),
                f'{mode}_loss/pred_cxr': loss_pred_cxr.detach(),
                'step': float(self.current_epoch)
            }, on_epoch=True, on_step=False, batch_size=y_gt.shape[0])

        return loss_pred_final, loss_pred_ehr, loss_pred_cxr, loss_pred_shared

    def _compute_and_log_loss(self, model_output, y_gt, pairs, log=True, mode='train'):
        prediction_losses = self._compute_prediction_losses(model_output, y_gt, pairs, log, mode)
        loss_pred_final, loss_pred_ehr, loss_pred_cxr, loss_pred_shared = prediction_losses

        loss_prediction = (self.hparams.lambda_pred_shared * loss_pred_shared +
                           self.hparams.lambda_pred_ehr * loss_pred_ehr +
                           self.hparams.lambda_pred_cxr * loss_pred_cxr)

        loss_prediction = loss_pred_final + loss_prediction

        loss_disentanglement = self._disentangle_loss_jsd(model_output, pairs, log, mode)

        loss_total = loss_prediction + loss_disentanglement
        epoch_log = {}

        # aux loss for attention ranking
        raw_pred_loss_ehr = F.binary_cross_entropy(model_output['pred_ehr'].data, y_gt, reduction='none')
        raw_pred_loss_cxr = F.binary_cross_entropy(model_output['pred_cxr'].data, y_gt, reduction='none')
        raw_pred_loss_shared = F.binary_cross_entropy(model_output['pred_shared'].data, y_gt, reduction='none')

        pairs = pairs.unsqueeze(1)
        attn_weights = model_output['attn_weights']
        attn_ehr, attn_shared, attn_cxr = attn_weights[:, :, 0], attn_weights[:, :, 1], attn_weights[:, :, 2]

        cxr_overweights_ehr = 2 * (raw_pred_loss_cxr < raw_pred_loss_ehr).float() - 1
        loss_attn1 = pairs * F.margin_ranking_loss(attn_cxr, attn_ehr, cxr_overweights_ehr, reduction='none')
        loss_attn1 = loss_attn1.sum() / max(1e-6, loss_attn1[loss_attn1>0].numel())

        shared_overweights_ehr = 2 * (raw_pred_loss_shared < raw_pred_loss_ehr).float() - 1
        loss_attn2 = pairs * F.margin_ranking_loss(attn_shared, attn_ehr, shared_overweights_ehr, reduction='none')
        loss_attn2 = loss_attn2.sum() / max(1e-6, loss_attn2[loss_attn2>0].numel())

        shared_overweights_cxr = 2 * (raw_pred_loss_shared < raw_pred_loss_cxr).float() - 1
        loss_attn3 = pairs * F.margin_ranking_loss(attn_shared, attn_cxr, shared_overweights_cxr, reduction='none')
        loss_attn3 = loss_attn3.sum() / max(1e-6, loss_attn3[loss_attn3>0].numel())

        loss_attn_ranking = (loss_attn1 + loss_attn2 + loss_attn3) / 3

        loss_total = loss_total + self.hparams.lambda_attn_aux * loss_attn_ranking
        epoch_log[f'{mode}_loss/attn_aux'] = loss_attn_ranking.detach()

        if log:
            epoch_log.update({
                f'{mode}_loss/total': loss_total.detach(),
                f'{mode}_loss/prediction': loss_prediction.detach(),
                'step': float(self.current_epoch)
            })
            self.log_dict(epoch_log, on_epoch=True, on_step=False, batch_size=y_gt.shape[0])

        return loss_total

    def _get_batch_data(self, batch):
        x, img, y_ehr, seq_lengths, pairs = batch
        y = torch.from_numpy(y_ehr).float().to(self.device)
        x = torch.from_numpy(x).float().to(self.device)
        img = img.to(self.device)
        pairs = torch.FloatTensor(pairs).to(self.device)
        return x, img, y, seq_lengths, pairs

    def training_step(self, batch, batch_idx):
        x, img, y, seq_lengths, pairs = self._get_batch_data(batch)
        if self.hparams.data_pair == 'paired' and self.hparams.aug_missing_ratio > 0:
            perm = torch.randperm(pairs.shape[0])
            idx = perm[:int(self.hparams.aug_missing_ratio * pairs.shape[0])]
            pairs[idx] = 0
        out = self.model(x, img, seq_lengths, pairs, 0)
        return self._compute_and_log_loss(out, y_gt=y, pairs=pairs)

    def validation_step(self, batch, batch_idx):
        x, img, y, seq_lengths, pairs = self._get_batch_data(batch)
        out = self.model(x, img, seq_lengths, pairs, 0)
        loss = self._compute_and_log_loss(out, y_gt=y, pairs=pairs, mode='val')

        pred_final =  out['pred_final']

        # self.val_preds.append(out['pred_final'])
        self.val_preds['final'].append(pred_final)
        self.val_preds['ehr'].append(out['pred_ehr'])
        self.val_preds['cxr'].append(out['pred_cxr'])
        self.val_pairs.append(pairs)
        self.val_labels.append(y)

        # return self._compute_masked_pred_loss(out['pred_final'], y, torch.ones_like(y[:, 0]))
        return self._compute_masked_pred_loss(pred_final, y, torch.ones_like(y[:, 0]))

    def on_validation_epoch_end(self):
        for name in ['final', 'ehr', 'cxr']:
            y_gt = torch.concat(self.val_labels, dim=0)
            preds = torch.concat(self.val_preds[name], dim=0)
            if name == 'cxr':
                pairs = torch.concat(self.val_pairs, dim=0)
                y_gt = y_gt[pairs==1, :]
                preds = preds[pairs==1, :]

            mlaps = multilabel_average_precision(preds, y_gt.long(), num_labels=y_gt.shape[1], average=None)
            mlroc = multilabel_auroc(preds, y_gt.long(), num_labels=y_gt.shape[1], average=None)

            if name == 'final':
                self.log('Val_PRAUC', mlaps.mean(), logger=False, prog_bar=True)
                self.log('Val_AUROC', mlroc.mean(), logger=False, prog_bar=True)

            log_dict = {
                'step': float(self.current_epoch),
                f'val_PRAUC_avg_over_dxs/{name}': mlaps.mean(),
                f'val_AUROC_avg_over_dxs/{name}': mlroc.mean(),
            }
            for i in range(mlaps.shape[0]):
                log_dict[f'val_PRAUC_per_dx_{name}/{self.label_names[i]}'] = mlaps[i]
                log_dict[f'val_AUROC_per_dx_{name}/{self.label_names[i]}'] = mlroc[i]

            self.log_dict(log_dict)

        for k in self.val_preds:
            self.val_preds[k].clear()
        self.val_pairs.clear()
        self.val_labels.clear()

    def test_step(self, batch, batch_idx):
        x, img, y, seq_lengths, pairs = self._get_batch_data(batch)
        out = self.model(x, img, seq_lengths, pairs, 0)

        pred_final =  out['pred_final']

        self.test_preds.append(pred_final)
        self.test_labels.append(y)
        self.test_pairs.append(pairs)
        self.test_attns.append(out['attn_weights'])

        for k in self.test_feats:
            self.test_feats[k].append(out[k].cpu())


    def on_test_epoch_end(self):
        y_gt = torch.concat(self.test_labels, dim=0)
        preds = torch.concat(self.test_preds, dim=0)
        pairs = torch.concat(self.test_pairs, dim=0)
        attn_weights = torch.concat(self.test_attns, dim=0)
        mlaps = multilabel_average_precision(preds, y_gt.long(), num_labels=y_gt.shape[1], average=None)
        mlroc = multilabel_auroc(preds, y_gt.long(), num_labels=y_gt.shape[1], average=None)
        self.test_results = {
            'y_gt': y_gt.cpu(),
            'preds': preds.cpu(),
            'pairs': pairs.cpu(),
            'mlaps': mlaps.cpu(),
            'mlroc': mlroc.cpu(),
            'prauc': mlaps.mean().item(),
            'auroc': mlroc.mean().item(),
            'attn_weight': attn_weights.cpu()
        }
        for k in self.test_feats:
            self.test_results[k] = torch.concat(self.test_feats[k], dim=0)
            self.test_feats[k].clear()
        self.test_labels.clear()
        self.test_preds.clear()
        self.test_pairs.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        return optimizer
