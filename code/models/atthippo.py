from attr import get_run_validators
import torch
import pytorch_lightning as pl

import torch.nn as nn
import torch.nn.functional as F

# import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
from polyode.models.ode_utils import NODE
from torchdiffeq import odeint
import numpy as np
from numpy.linalg import matrix_rank
import math

from sklearn.metrics import roc_auc_score, accuracy_score
from polyode.utils import str2bool
import time
import os

torch.autograd.set_detect_anomaly(True)


class ATThippomod(nn.Module):
    def __init__(self, Nc, r_dim, input_dim, output_dim, hidden_dim, time_num, hippo_mode, ah_compute_type,
                 p_compute_type, zpinv, Delta, delta_t, temp_mode=None, method="euler", bridge_ode=False, impute_mode=False, **kwargs):
        """
        Nc = dimension of the hippo coefficients
        input_dim = dimension of input data  # d
        output_dim = dimension of the hidden state  # D
        hidden_dim = hidden_dimension of the neural network
        time_num = number of time steps  # T

        Delta = parameter of the hippo measure
        """
        super().__init__()

        if ah_compute_type == 'p' or ah_compute_type == 'att':
            self.h = nn.Parameter(torch.randn(time_num))
        self.time_num = time_num

        # HIPPO-LegS
        # A,B div t later in ode
        if hippo_mode:
            self.A = nn.Parameter(torch.ones((Nc, Nc)), requires_grad=False)
            self.B = nn.Parameter(torch.ones(Nc, requires_grad=False), requires_grad=False)
            for n in range(Nc):
                self.B[n] = (2 * n + 1) ** 0.5
                for k in range(Nc):
                    if n > k:
                        self.A[n, k] = - ((2 * n + 1) ** 0.5) * ((2 * k + 1) ** 0.5)
                    elif n == k:
                        self.A[n, k] = - (n + 1)
                    else:
                        self.A[n, k] = 0

        self.hippo_mode = hippo_mode
        self.ah_compute_type = ah_compute_type
        self.p_compute_type = p_compute_type
        self.zpinv = zpinv
        self.temp_mode = temp_mode

        self.delta_t = delta_t
        self.method = method

        self.bridge_ode = bridge_ode
        self.Nc = Nc
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.r_dim = r_dim
        self.impute_mode = impute_mode

        self.Wv = nn.Parameter(torch.ones((output_dim, output_dim)))

        # hippo
        if hippo_mode:
            self.mlp_r = nn.Sequential(nn.Linear(r_dim + Nc + output_dim, hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, r_dim))
            self.linear = nn.Linear(r_dim, 1)
            self.hippo_out_fun = nn.Sequential(nn.Linear(r_dim + Nc + output_dim, hidden_dim), nn.ReLU(),
                                               nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                               nn.Linear(hidden_dim, input_dim))
        else:
            self.conv = nn.ModuleList()
            for i in range(2):
                self.conv.append(
                    nn.Sequential(nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
                                  nn.ReLU()))
            self.out_fun1 = nn.Sequential(nn.Linear(output_dim, hidden_dim), nn.ReLU())
            self.out_fun3 = nn.Sequential(nn.Linear(hidden_dim, input_dim))

        self.mlp_z = nn.Sequential(nn.Linear(output_dim, hidden_dim),
                                       nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        self.gru_cell = nn.GRU(input_dim, output_dim)

        self.timesteps = 0

    def dev_s(self, t, att, h1, Z, Z_pinv, Z_trans, Ap, I, J):
        # time-encode B*D
        time_encoding = torch.zeros(att.size(0), self.output_dim).to(att.device)
        for i in range(self.output_dim // 2):
            time_encoding[:, 2 * i] = torch.sin(np.power(2, i) * t)
            time_encoding[:, 2 * i + 1] = torch.cos(np.power(2, i) * t)
        att1 = att + time_encoding

        bp = torch.einsum('bij,bj->bi', Z_pinv, att1)
        if self.p_compute_type == 'max_hoyer':
            p = (bp - (torch.sum(bp, dim=1) - 1).unsqueeze(-1) *
                torch.sum(Ap, dim=2) / torch.sum(Ap, dim=(1, 2)).unsqueeze(-1))
        elif self.p_compute_type == 'min_norm':
            p = bp
        elif self.p_compute_type == 'ada_h':
            p = bp + torch.einsum('bij,bj->bi', Ap, self.hs)
        self.temp1 = p
        p = p - p.min(dim=1, keepdim=True).values
        p = p / p.sum(dim=1, keepdim=True)  # B*N

        zp = torch.einsum('bij,bj->bi', Z_trans, p)
        datt_dz = (torch.einsum('bij,bjk->bik', p.unsqueeze(1) * Z_trans, Z) -
                   torch.einsum('bi,bj->bij', zp, zp))
        zt = torch.einsum('bi,bij->bj', p, Z_pinv)
        dz_dt = self.mlp_z(zt)
        datt_dt = torch.einsum('bi,bij->bj', dz_dt, datt_dz)

        return datt_dt

    def ode_fun(self, t, embed, Z, Z_pinv, Z_trans, Ap, I, J):
        self.timesteps += 1
        # if self.timesteps % 50 == 0:
        #     print('timesteps:', self.timesteps)

        if not self.hippo_mode:
            return self.dev_s(t, embed, 0, Z, Z_pinv, Z_trans, Ap, I, J)
        # hippo
        else:
            # r:B*Nr, c:B*(Nc), att:B*D
            r = embed[:, :self.r_dim]
            c = embed[:, self.r_dim: -self.output_dim]
            att = embed[:, -self.output_dim:]

            dr_dt = self.mlp_r(torch.cat((r, c, att), dim=1))
            dc_dt = (c @ self.A.T + self.linear(r) * self.B) / t
            datt_dt = self.dev_att(t, att, 0, Z, Z_pinv, Z_trans, Ap, I, J)

            return torch.cat([dr_dt, dc_dt, datt_dt], dim=1)

    def forward(self, times, Y, mask, casetime, eval_mode=False, bridge_info=None):
        """
        eval mode returns the ode integrations at multiple times in between observations
        """

        if self.p_compute_type == 'ada_h':
            self.hs = nn.Parameter(torch.randn(Y.shape[0], 1461)).to(Y.device)
        if not self.hippo_mode and self.p_compute_type != 'ada_h':
            init_embed = torch.rand(Y.shape[0], self.output_dim, device=Y.device)
        else:
            init_embed = torch.zeros(Y.shape[0], self.r_dim + self.Nc + self.output_dim, device=Y.device)

        # print('time:', times)
        self.timesteps = 0

        if eval_mode:
            # generates 8 equally spaced time points between each pair of consecutive original time points
            tlist = [times[0][..., None]]
            for i in range(times.shape[0] - 1):
                tlist.append(torch.linspace(times[i], times[i + 1], steps=10)[1:].to(init_embed.device))
            eval_times = torch.cat(tlist).to(init_embed.device)
        else:
            eval_times = times.to(init_embed.device)

        torch.autograd.set_detect_anomaly(True)
        # map data Y to hidden state Z
        time1 = time.time()

        # gru
        gru_embed = torch.rand(1, Y.shape[0], self.output_dim, device=Y.device, requires_grad=False)
        # ex: Y[:, :self.time_num, :]
        if self.impute_mode:
            ally = torch.zeros(Y.shape[0], len(times), Y.shape[2], device=Y.device)  # B*allT*d
            ally.scatter_(1, casetime[:, :self.time_num].long().unsqueeze(-1).expand(-1, -1, ally.shape[2]), Y[:, :self.time_num, :])
            Z, lastz = self.gru_cell(ally.transpose(0, 1), gru_embed)
        else:
            Z = self.gru_cell(Y.transpose(0, 1), gru_embed)[0]

        Z = Z.transpose(0, 1)
        U, S, V = torch.svd(Z)
        Z_pinv = U * torch.reciprocal(S).unsqueeze(1) @ (V.transpose(1, 2) * self.Wv)
        # print('Wv:', self.Wv.data)

        Z_trans = Z.permute(0, 2, 1)  # Z.T: B*D*T
        # identity matrix
        I = torch.eye(Z.shape[1]).to(Z.device)
        # all one vector
        J = torch.ones(Z.shape[1]).to(Z.device)
        Ap = I - torch.einsum('bij,bjk->bik', Z_pinv, Z_trans)

        if (Z.isnan().any()) or (torch.isinf(Z).any()):
            print('Z-nan:', torch.nonzero(Z.isnan()).flatten())
            print('Z-inf:', torch.nonzero(Z.isinf()).flatten())
            import ipdb;
            ipdb.set_trace()

        ode_func_with_args = lambda t, embed: self.ode_fun(t, embed, Z, Z_pinv, Z_trans, Ap, I, J)
        # B*t*D(Nr+Nc+D)
        embed_series = odeint(ode_func_with_args, init_embed, eval_times, method=self.method,
                              options={"step_size": self.delta_t}).permute(1, 0, 2)

        # B*t*d
        if not self.hippo_mode:
            pred = self.out_fun1(embed_series).permute(0, 2, 1)
            for i in range(2):
                pred = self.conv[i](pred) + pred
            pred = self.out_fun3(pred.permute(0, 2, 1))
            # pred = self.out_fun(embed_series)
        else:
            pred = self.hippo_out_fun(embed_series)

        # B*t*d
        y_traj = pred
        # print('ytraj:', y_traj.shape, eval_mode)
        # maxb = torch.argmax(y_traj) // (y_traj.shape[1] * y_traj.shape[2])
        # print('maxy:', maxb, y_traj[maxb])

        # B*T*d
        if eval_mode:
            y_preds = y_traj[:, ::9, :]
        else:
            y_preds = y_traj

        if (y_preds.isnan().any()) or (torch.isinf(y_preds).any()):
            import ipdb;
            ipdb.set_trace()
        # B*(t*D)
        embed_series = embed_series.reshape(embed_series.size(0), -1)

        return y_preds, y_traj, eval_times, embed_series


class ATThippo(pl.LightningModule):
    def __init__(
            self,
            # channels,
            time_num,
            hippo_mode,
            lr=0.001,
            hidden_dim=32,
            input_dim=1,
            att_dim=8,
            r_dim=16,
            hippo_dim=16,
            weight_decay=0.,
            Delta=5,
            delta_t=0.05,
            method="dopri5",
            temp_mode="s",
            direct_classif=False,
            bridge_ode=False,
            impute_mode=False,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.output_dim = att_dim
        self.input_dim = input_dim
        self.time_num = time_num
        self.node_model = ATThippomod(Nc=hippo_dim, r_dim=r_dim, input_dim=input_dim, output_dim=att_dim,
                                      hidden_dim=hidden_dim, time_num=time_num, temp_mode=temp_mode,
                                      hippo_mode=hippo_mode, Delta=Delta, delta_t=delta_t, method=method,
                                      bridge_ode=bridge_ode, impute_mode=impute_mode, **kwargs)

        self.Delta = Delta

        self.direct_classif = direct_classif
        self.bridge_ode = bridge_ode
        self.impute_mode = impute_mode

        if self.hparams["data_type"] == "pMNIST":
            self.loss_class = torch.nn.CrossEntropyLoss()
            classify_dim = 10
        elif self.hparams["data_type"] == "Character":
            self.loss_class = torch.nn.CrossEntropyLoss()
            classify_dim = 20
        else:
            self.loss_class = torch.nn.BCEWithLogitsLoss()
            classify_dim = 1
        if not hippo_mode:
            embed_dim = att_dim
        else:
            embed_dim = r_dim + hippo_dim + att_dim
        if direct_classif:
            self.classify_model = nn.Sequential(nn.Linear(time_num * embed_dim, hidden_dim), nn.ReLU(),
                                                nn.Linear(hidden_dim, classify_dim))

    def forward(self, times, Y, mask, casetime, eval_mode=False, bridge_info=None):
        return self.node_model(times, Y, mask, casetime, eval_mode=eval_mode, bridge_info=bridge_info)

    def get_embedding(self, times, Y, mask, casetime, eval_mode=False):
        _, _, _, embedding = self(times, Y, mask, casetime, eval_mode=eval_mode)
        return embedding

    def compute_loss(self, Y, preds, mask, stds=None):
        if self.impute_mode:
            loss = ((preds - Y).pow(2) * mask).sum() / mask.sum()
        elif stds is not None:
            loss = ((2 * torch.log(2 * torch.pi * stds.pow(2) + 0.00001) + (preds - Y).pow(2) /
                     (0.001 + stds.pow(2))) * (mask[..., None])).mean(-1).sum() / mask.sum()
        else:
            if len(mask.shape) == 3:
                loss = ((preds - Y).pow(2) * mask).sum() / mask.sum()
            else:
                loss = ((preds - Y).pow(2) * mask[..., None]
                        ).mean(-1).sum() / mask.sum()
        return loss

    def process_batch(self, batch, forecast_mode=False):
        if self.bridge_ode:
            times, Y, mask, label, ids, ts, ys, mask_ids = batch
            return times, Y, mask, label, (ids, ts, ys, mask_ids)
        elif forecast_mode:
            times, Y, mask, label, _, Y_past, Y_future, mask_past, mask_future = batch
            return times, Y, mask, label, None, Y_past, Y_future, mask_past, mask_future
        elif self.impute_mode:
            times, Y, mask, truth, casetime = batch
            return times, Y, mask, truth, casetime
        else:
            times, Y, mask, label, _ = batch
            return times, Y, mask, label, None

    def predict_step(self, batch, batch_idx):
        times, Y, mask, label, bridge_info, Y_past, Y_future, mask_past, mask_future = self.process_batch(
            batch, forecast_mode=True)

        # assert len(torch.unique(T)) == T.shape[1]
        # times = torch.sort(torch.unique(T))[0]
        preds, preds_traj, times_traj, _, _ = self(
            times, Y_past, mask_past, bridge_info=bridge_info)

        _, _, times_traj, cn_embedding, uncertainty_embedding = self(
            times, Y, mask, bridge_info=bridge_info)

        # B*n*d
        cn_embedding = torch.stack(torch.chunk(cn_embedding, self.output_dim, -1), -1)

        Tmax = times.max()
        Nc = cn_embedding.shape[1]  # number of coefficients
        backward_window = self.Delta
        mask_rec = (times - Tmax + backward_window) > 0
        rec_span = times[mask_rec]
        # rec_span = np.linspace(Tmax-self.Delta, Tmax)
        recs = [np.polynomial.legendre.legval(
            (2 / self.Delta) * (rec_span - Tmax).cpu().numpy() + 1,
            (cn_embedding[..., out_dim].cpu().numpy() * [(2 * n + 1) ** 0.5 for n in range(Nc)]).T) for out_dim in
            range(self.output_dim)]
        recs = torch.Tensor(np.stack(recs, -1))

        return {"Y_future": Y_future, "preds": preds, "mask_future": mask_future,
                "pred_rec": recs, "Y_rec": Y[:, mask_rec, :],
                "mask_rec": mask[:, mask_rec, ...]}

    def training_step(self, batch, batch_idx):

        if self.impute_mode:
            times, Y, mask, truth, casetime = self.process_batch(batch)
            label = bridge_info = None
        else:
            times, Y, mask, label, bridge_info = self.process_batch(batch)
            casetime = times
            truth = Y
        # assert len(torch.unique(T)) == T.shape[1]
        # times = torch.sort(torch.unique(T))[0]
        preds, preds_traj, times_traj, cn_embedding = self(times, Y, mask, casetime, bridge_info=bridge_info)
        print(times.dtype, times.shape, casetime.dtype, casetime.shape, preds.shape)
        # import ipdb;
        # ipdb.set_trace()

        preds_class = None
        if self.impute_mode:
            preds = preds[torch.arange(preds.size(0)).unsqueeze(1), casetime.long()]
        mse = self.compute_loss(truth, preds, mask)
        if (mse.isnan().any()) or torch.isinf(mse):
            import ipdb;
            ipdb.set_trace()
        self.log("train_loss", mse, on_epoch=True)
        return {"loss": mse}

    def validation_step(self, batch, batch_idx):
        if self.impute_mode:
            times, Y, mask, truth, casetime = self.process_batch(batch)
            label = bridge_info = None
        else:
            times, Y, mask, label, bridge_info = self.process_batch(batch)
            casetime = times
            truth = Y
        # assert len(torch.unique(T)) == T.shape[1]
        # times = torch.sort(torch.unique(T))[0]
        preds, preds_traj, times_traj, cn_embedding = self(times, Y, mask, casetime, bridge_info=bridge_info,
                                                           eval_mode=False)
        print(times.shape, casetime.shape, preds.shape)
        # import ipdb;
        # ipdb.set_trace()

        if self.impute_mode:
            preds = preds[torch.arange(preds.size(0)).unsqueeze(1), casetime.long()]
        preds_class = None
        mse = self.compute_loss(truth, preds, mask)
        if (mse.isnan().any()) or torch.isinf(mse):
            import ipdb;
            ipdb.set_trace()
        mse = mse
        self.log("val_mse", mse, on_epoch=True)
        self.log("val_loss", mse, on_epoch=True)
        return {"Y": Y, "preds": preds, "T": times, "preds_traj": preds_traj, "times_traj": times_traj, "mask": mask,
                "label": label, "pred_class": preds_class, "cn_embedding": cn_embedding}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    @classmethod
    def add_model_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=32)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0.)
        # parser.add_argument('--Delta', type=float, default=5, help="Memory span")
        parser.add_argument('--corr_time', type=float, default=0.5, help="Correction span")
        parser.add_argument('--delta_t', type=float, default=0.05, help="integration step size")
        parser.add_argument('--method', type=str, default="implicit_adams", help="integration method")
        parser.add_argument('--direct_classif', type=str2bool, default=False)
        parser.add_argument('--bridge_ode', type=str2bool, default=False)
        # parser.add_argument('--impute_mode', type=str2bool, default=False)
        parser.add_argument('--att_dim', type=int, default=8, help="dimension of ATT")
        parser.add_argument('--r_dim', type=int, default=16, help="dimension of r")
        parser.add_argument('--hippo_dim', type=int, default=16, help="dimension of hippo coefficients")
        parser.add_argument('--hippo_mode', type=str2bool, default=False)
        parser.add_argument('--ah_compute_type', type=str, default="equals_p", choices=["equals_p", "p", "att"])
        parser.add_argument('--p_compute_type', type=str, default="max_hoyer",
                            choices=["max_hoyer", "min_norm", "ada_h"])
        parser.add_argument('--zpinv', type=str, default="mlp", choices=["norm", "Tnorm", "mlp", "vmlp"])
        parser.add_argument('--temp_mode', type=str, default=None)

        return parser


class MultiLabelCrossEntropyLoss:
    def __init__(self):
        self.loss1 = torch.nn.CrossEntropyLoss()
        self.loss2 = torch.nn.CrossEntropyLoss()

    def __call__(self, inputs, targets):
        import ipdb;
        ipdb.set_trace()
        return 0.5 * (self.loss1(inputs[:, 0], targets[:, 0].float()) + self.loss2(inputs[:, 1], targets[:, 1].float()))


class ATThippoClassification(pl.LightningModule):
    def __init__(self, lr,
                 hidden_dim,
                 att_dim,
                 r_dim,
                 hippo_dim,
                 hippo_mode,
                 p_compute_type,
                 weight_decay,
                 init_model,
                 time_num,
                 time_gap,
                 pre_compute_ode=False,
                 num_dims=1,
                 regression_mode=False,
                 **kwargs
                 ):

        super().__init__()
        self.save_hyperparameters()
        self.embedding_model = init_model
        self.embedding_model.freeze()
        self.pre_compute_ode = pre_compute_ode

        self.regression_mode = regression_mode
        if regression_mode:
            self.loss_class = torch.nn.MSELoss()
            classify_dim = 1
        else:
            if self.hparams["data_type"] == "pMNIST":
                self.loss_class = torch.nn.CrossEntropyLoss()
                classify_dim = 10
            elif self.hparams["data_type"] == "Character":
                self.loss_class = torch.nn.CrossEntropyLoss()
                classify_dim = 20
            elif self.hparams["data_type"] == "Activity":
                self.loss_class = MultiLabelCrossEntropyLoss()
                classify_dim = 6 * 2
            else:
                self.loss_class = torch.nn.BCEWithLogitsLoss()
                classify_dim = 1
        print('MMMat:', att_dim, math.ceil(time_num / time_gap))
        self.time_num = time_num
        self.time_gap = time_gap
        if not hippo_mode and p_compute_type != 'ada_h':
            embed_dim = att_dim
        # elif p_compute_type == 'ada_h':
        #     embed_dim = att_dim + time_num
        else:
            embed_dim = r_dim + hippo_dim + att_dim
        self.classify_model = nn.Sequential(
            nn.Linear(math.ceil(time_num / time_gap) * embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, classify_dim))

    def forward(self, times, Y, mask, coeffs, eval_mode=False):

        if self.pre_compute_ode:
            embeddings = coeffs
        else:
            _, _, _, embeddings = self.embedding_model(times, Y, mask)
        embeddings = embeddings.reshape(embeddings.shape[0], self.time_num, -1)[:, ::self.time_gap, :].reshape(
            embeddings.shape[0], -1)

        preds = self.classify_model(embeddings)
        return preds

    def predict_step(self, batch, batch_idx):
        times, Y, mask, label, embeddings = batch
        preds = self(times, Y, mask, embeddings)
        if self.regression_mode:
            if len(label.shape) == 1:
                label = label[:, None]
            loss = self.loss_class(preds, label)
        else:
            if preds.shape[-1] == 1:
                preds = preds[:, 0]
                loss = self.loss_class(preds.double(), label)
            else:
                loss = self.loss_class(preds.double(), label.long())
        return {"Y": Y, "preds": preds, "T": times, "labels": label}

    def training_step(self, batch, batch_idx):
        times, Y, mask, label, embeddings = batch
        preds = self(times, Y, mask, embeddings)
        if self.regression_mode:
            if len(label.shape) == 1:
                label = label[:, None]
            loss = self.loss_class(preds, label)
        else:
            if preds.shape[-1] == 1:
                preds = preds[:, 0]
                loss = self.loss_class(preds.double(), label)
            else:
                loss = self.loss_class(preds.double(), label.long())
        self.log("train_loss", loss, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        times, Y, mask, label, embeddings = batch
        preds = self(times, Y, mask, embeddings)
        if self.regression_mode:
            if len(label.shape) == 1:
                label = label[:, None]
            loss = self.loss_class(preds, label)
        else:
            if preds.shape[-1] == 1:
                preds = preds[:, 0]
                loss = self.loss_class(preds.double(), label)
            else:
                loss = self.loss_class(preds.double(), label.long())
        self.log("val_loss", loss, on_epoch=True)
        return {"Y": Y, "preds": preds, "T": times, "labels": label}

    def validation_epoch_end(self, outputs):
        if not self.regression_mode:
            preds = torch.cat([x["preds"] for x in outputs])
            labels = torch.cat([x["labels"] for x in outputs])
            if (self.hparams["data_type"] == "pMNIST") or (self.hparams["data_type"] == "Character"):
                preds = torch.nn.functional.softmax(preds, dim=-1).argmax(-1)
                accuracy = accuracy_score(
                    labels.long().cpu().numpy(), preds.cpu().numpy())
                self.log("val_acc", accuracy, on_epoch=True)
            elif self.hparams["data_type"] == "Activity":
                import ipdb;
                ipdb.set_trace()
                auc1 = roc_auc_score(labels[:, 0].cpu().numpy(), preds[:, 0].cpu().numpy())
                auc2 = roc_auc_score(labels[:, 0].cpu().numpy(), preds[:, 1].cpu().numpy())
            else:
                auc = roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())
                self.log("val_auc", auc, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.classify_model.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

    @classmethod
    def add_model_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=32)
        parser.add_argument('--att_dim', type=int, default=8, help="dimension of ATT")
        parser.add_argument('--r_dim', type=int, default=16, help="dimension of r")
        parser.add_argument('--hippo_dim', type=int, default=16, help="dimension of hippo coefficients")
        parser.add_argument('--hippo_mode', type=str2bool, default=False)
        parser.add_argument('--p_compute_type', type=str, default="max_hoyer",
                            choices=["max_hoyer", "min_norm", "ada_h"])
        parser.add_argument('--time_gap', type=int, default=1, help="division times of embedding")
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0.)
        return parser
