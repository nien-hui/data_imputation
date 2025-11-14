# stdlib
from typing import Any, List, Tuple, Union
from pathlib import Path

# third party
import numpy as np
import math, sys, argparse
import pandas as pd
import torch
from torch import nn
from functools import partial
import time, os, json
from utils import NativeScaler, MAEDataset, adjust_learning_rate, get_dataset
import model_mae
from torch.utils.data import DataLoader, RandomSampler
import sys
import timm.optim.optim_factory as optim_factory
from utils import get_args_parser

# hyperimpute absolute
from hyperimpute.plugins.imputers import ImputerPlugin
from sklearn.datasets import load_iris
from hyperimpute.utils.benchmarks import compare_models
from hyperimpute.plugins.imputers import Imputers

eps = 1e-8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReMasker:

    def __init__(self):
        args = get_args_parser().parse_args()

        self.batch_size = args.batch_size
        self.accum_iter = args.accum_iter
        self.min_lr = args.min_lr
        self.norm_field_loss = args.norm_field_loss
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        self.blr = args.blr
        self.warmup_epochs = args.warmup_epochs
        self.model = None
        self.norm_parameters = None

        self.embed_dim = args.embed_dim
        self.depth = args.depth
        self.decoder_depth = args.decoder_depth
        self.num_heads = args.num_heads
        self.mlp_ratio = args.mlp_ratio
        self.max_epochs = args.max_epochs
        self.mask_ratio = args.mask_ratio
        self.encode_func = args.encode_func

    def fit(self, X_raw: pd.DataFrame):
        X = X_raw.clone()

        # Parameters
        no = len(X)
        dim = len(X[0, :])

        X = X.cpu()

        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

        for i in range(dim):
            min_val[i] = np.nanmin(X[:, i])
            max_val[i] = np.nanmax(X[:, i])
            X[:, i] = (X[:, i] - min_val[i]) / (max_val[i] - min_val[i] + eps)

        self.norm_parameters = {"min": min_val, "max": max_val}

        # Set missing
        M = 1 - (1 * (np.isnan(X)))
        M = M.float().to(device)

        X = torch.nan_to_num(X)
        X = X.to(device)

        self.model = model_mae.MaskedAutoencoder(
            rec_len=dim,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            decoder_embed_dim=self.embed_dim,
            decoder_depth=self.decoder_depth,
            decoder_num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=eps),
            norm_field_loss=self.norm_field_loss,
            encode_func=self.encode_func
        )

        # if self.improve and os.path.exists(self.path):
        #     self.model.load_state_dict(torch.load(self.path))
        #     self.model.to(device)
        #     return self

        self.model.to(device)

        # set optimizers
        # param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
        eff_batch_size = self.batch_size * self.accum_iter
        if self.lr is None:  # only base_lr is specified
            self.lr = self.blr * eff_batch_size / 64
        # param_groups = optim_factory.add_weight_decay(self.model, self.weight_decay)
        # optimizer = torch.optim.AdamW(param_groups, lr=self.lr, betas=(0.9, 0.95))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.95))
        loss_scaler = NativeScaler()

        dataset = MAEDataset(X, M)
        dataloader = DataLoader(
            dataset, sampler=RandomSampler(dataset),
            batch_size=self.batch_size,
        )

        # if self.resume and os.path.exists(self.path):
        #     self.model.load_state_dict(torch.load(self.path))
        #     self.lr *= 0.5

        self.model.train()

        # for epoch in range(self.max_epochs):

        #     optimizer.zero_grad()
        #     total_loss = 0

        #     iter = 0
        #     for iter, (samples, masks) in enumerate(dataloader):

        #         # we use a per iteration (instead of per epoch) lr scheduler
        #         if iter % self.accum_iter == 0:
        #             adjust_learning_rate(optimizer, iter / len(dataloader) + epoch, self.lr, self.min_lr,
        #                                  self.max_epochs, self.warmup_epochs)

        #         samples = samples.unsqueeze(dim=1)
        #         samples = samples.to(device, non_blocking=True)
        #         masks = masks.to(device, non_blocking=True)

        #         # print(samples, masks)

        #         with torch.cuda.amp.autocast():
        #             loss, _, _, _ = self.model(samples, masks, mask_ratio=self.mask_ratio)
        #             loss_value = loss.item()
        #             total_loss += loss_value

        #         if not math.isfinite(loss_value):
        #             print("Loss is {}, stopping training".format(loss_value))
        #             sys.exit(1)

        #         loss /= self.accum_iter
        #         loss_scaler(loss, optimizer, parameters=self.model.parameters(),
        #                     update_grad=(iter + 1) % self.accum_iter == 0)

        #         if (iter + 1) % self.accum_iter == 0:
        #             optimizer.zero_grad()

        #     total_loss = (total_loss / (iter + 1)) ** 0.5
        #     # if total_loss < best_loss:
        #     #     best_loss = total_loss
        #     #     torch.save(self.model.state_dict(), self.path)
        #     # if (epoch + 1) % 10 == 0 or epoch == 0:
        #     # print((epoch+1),',', total_loss)

        # # torch.save(self.model.state_dict(), self.path)
        # return self

        from tqdm import tqdm
        
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        loss_log_path = os.path.join(checkpoint_dir, "epoch_losses.csv")
        # 如果檔案不存在就寫入 header
        if not os.path.exists(loss_log_path):
            with open(loss_log_path, "w") as f:
                f.write("timestamp,epoch,epoch_loss\n")

        for epoch in range(self.max_epochs):
            optimizer.zero_grad()
            total_loss = 0

            print(f"\nEpoch [{epoch+1}/{self.max_epochs}]")
            dataloader_iter = tqdm(enumerate(dataloader), total=len(dataloader),
                                desc=f"Epoch {epoch+1}", leave=False)

            for iter, (samples, masks) in dataloader_iter:
                # 動態調整學習率
                if iter % self.accum_iter == 0:
                    adjust_learning_rate(
                        optimizer,
                        iter / len(dataloader) + epoch,
                        self.lr,
                        self.min_lr,
                        self.max_epochs,
                        self.warmup_epochs,
                    )

                samples = samples.unsqueeze(dim=1)
                samples = samples.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    loss, _, _, _ = self.model(samples, masks, mask_ratio=self.mask_ratio)
                    loss_value = loss.item()
                    total_loss += loss_value

                if not math.isfinite(loss_value):
                    print(f"Loss is {loss_value}, stopping training")
                    sys.exit(1)

                loss /= self.accum_iter
                loss_scaler(
                    loss,
                    optimizer,
                    parameters=self.model.parameters(),
                    update_grad=(iter + 1) % self.accum_iter == 0,
                )

                if (iter + 1) % self.accum_iter == 0:
                    optimizer.zero_grad()

                # 更新 tqdm 顯示平均 loss
                avg_loss = total_loss / (iter + 1)
                dataloader_iter.set_postfix(loss=f"{avg_loss:.4f}")

            # 每個 epoch 結束後計算平均 loss
            epoch_loss = (total_loss / (iter + 1)) ** 0.5
            print(f"✅ Epoch [{epoch+1}/{self.max_epochs}] finished | Loss: {epoch_loss:.4f}")

            try:
                with open(loss_log_path, "a") as f:
                    f.write(f"{epoch+1},{epoch_loss:.6f}\n")
            except Exception as e:
                print(f"Warning: 無法寫入 loss log: {e}")

            if (epoch + 1) % 3 == 0:
                ckpt = {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                # 若 loss_scaler 支援 state_dict()（例如是 GradScaler），也一起存
                try:
                    if hasattr(loss_scaler, "state_dict"):
                        ckpt["amp_scaler_state_dict"] = loss_scaler.state_dict()
                except Exception:
                    # 如果存取失敗就跳過，不要中斷訓練
                    pass

                ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
                try:
                    torch.save(ckpt, ckpt_path)
                    print(f"💾 Saved checkpoint: {ckpt_path}")
                except Exception as e:
                    print(f"Warning: 儲存 checkpoint 失敗: {e}")

        return self

    def transform(self, X_raw: torch.Tensor):

        X = X_raw.clone().to(torch.float32).cpu()
        min_val = torch.from_numpy(self.norm_parameters["min"]).to(torch.float32).unsqueeze(0)
        max_val = torch.from_numpy(self.norm_parameters["max"]).to(torch.float32).unsqueeze(0)
        scale = (max_val - min_val + eps)

        no = X.shape[0]

        # MinMaxScaler normalization
        X = (X - min_val) / scale

        # Set missing
        M = (~torch.isnan(X)).float()
        X = torch.nan_to_num(X, nan=0.0)

        self.model.eval()

        batch_size = getattr(self, "batch_size", None) or no
        batch_size = max(1, min(batch_size, no))

        preds = []
        with torch.no_grad():
            batch_indices = range(0, no, batch_size)
            progress = None
            try:
                from tqdm import tqdm
                progress = tqdm(batch_indices, total=int(np.ceil(no / batch_size)), desc="Imputing", leave=False)
            except ImportError:
                progress = batch_indices

            for start in progress:
                end = start + batch_size
                samples = X[start:end].to(device).unsqueeze(1)
                masks = M[start:end].to(device)
                _, pred, _, _ = self.model(samples, masks)
                preds.append(pred.squeeze(dim=2).cpu())

            if hasattr(progress, "close"):
                progress.close()

        imputed_data = torch.cat(preds, dim=0)

        # Renormalize
        imputed_data = imputed_data * scale + min_val

        if torch.isnan(imputed_data).all():
            err = "The imputed result contains nan. This is a bug. Please report it on the issue tracker."
            raise RuntimeError(err)

        observed = M
        original = torch.nan_to_num(X_raw.clone().to(torch.float32), nan=0.0).cpu()
        return observed * original + (1 - observed) * imputed_data

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Imputes the provided dataset using the GAIN strategy.
        Args:
            X: np.ndarray
                A dataset with missing values.
        Returns:
            Xhat: The imputed dataset.
        """
        X = torch.tensor(X.values, dtype=torch.float32)
        return self.fit(X).transform(X).detach().cpu().numpy()
    
    def laod_param_fit_transform(self, X: torch.Tensor, direct):
        path = Path(direct)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {direct}")

        if isinstance(X, pd.DataFrame):
            X_tensor = torch.tensor(X.values, dtype=torch.float32)
        elif isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X, dtype=torch.float32)
        elif isinstance(X, torch.Tensor):
            X_tensor = X.detach().clone().float()
        else:
            raise TypeError(f"Unsupported data type for X: {type(X)}")

        if self.model is None:
            feature_dim = X_tensor.shape[1]
            self.model = model_mae.MaskedAutoencoder(
                rec_len=feature_dim,
                embed_dim=self.embed_dim,
                depth=self.depth,
                num_heads=self.num_heads,
                decoder_embed_dim=self.embed_dim,
                decoder_depth=self.decoder_depth,
                decoder_num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                norm_layer=partial(nn.LayerNorm, eps=eps),
                norm_field_loss=self.norm_field_loss,
                encode_func=self.encode_func,
            )
            self.model.to(device)
    
        checkpoint_state = torch.load(path, map_location=device)
        if isinstance(checkpoint_state, dict):
            state_dict = checkpoint_state.get("model_state_dict") or checkpoint_state.get("state_dict")
        else:
            state_dict = checkpoint_state

        if state_dict is None:
            raise RuntimeError(f"Checkpoint at {direct} lacks state dict")

        self.model.load_state_dict(state_dict)
        self.model.to(device)

        X_np = X_tensor.detach().cpu().numpy()
        if X_np.ndim != 2:
            raise RuntimeError("Input tensor must be 2D to derive normalization parameters.")
        min_vals = np.zeros(X_np.shape[1], dtype=np.float32)
        max_vals = np.zeros_like(min_vals)
        for idx in range(X_np.shape[1]):
            col = X_np[:, idx]
            if np.all(np.isnan(col)):
                min_vals[idx] = 0.0
                max_vals[idx] = 1.0
            else:
                min_vals[idx] = np.nanmin(col)
                max_vals[idx] = np.nanmax(col)

        self.norm_parameters = {"min": min_vals, "max": max_vals}

        return self.transform(X_tensor)
