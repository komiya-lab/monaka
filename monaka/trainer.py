# -*- coding: utf-8 -*-

import os
import datetime

import torch

import torch.nn as nn
import torch.distributed as dist
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from typing import List, Optional, Union, Dict
from monaka.dataset import LUWJsonLDataset
from monaka.mylogging import logger, init_logger
from monaka.model import LUWParserModel, init_device, is_master
from monaka.model import DistributedDataParallel as DDP


class Trainer:

    def __init__(self,
            train_files: Union[str, List[str]],
            dev_files: Union[str, List[str]],
            test_files: Optional[Union[str, List[str]]],
            dataeset_options: Dict,
            model_name: str,
            model_config: str,
            batch_size: int=8,
            epochs: int=1,
            lr: float=2e-3,
            mu: float=.9,
            nu: float=.9,
            epsilon: float=1e-12,
            clip: float=5.0,
            decay: float=.75,
            decay_steps: float=5000,
            patience: float=100,
            verbose: bool=True,
            **kwargs):
        
        init_logger(logger, verbose=verbose)

        logger.info("loading train files")
        self.train_data = LUWJsonLDataset(train_files, **dataeset_options)

        logger.info("loading dev files")
        self.dev_data = LUWJsonLDataset(dev_files, **dataeset_options)

        logger.info("loading test files")
        self.test_data = LUWJsonLDataset(test_files, **dataeset_options) if test_files else None

        self.batch_size=batch_size // dist.get_world_size()
        self.epochs = epochs
        self.lr = lr
        self.mu = mu
        self.nu = nu
        self.epsilon = epsilon
        self.clip = clip
        self.decay = decay
        self.decay_steps = decay_steps
        self.patience = patience
        self.verbose = verbose
        self.model_name = model_name

        logger.info("loading model")
        self.model = LUWParserModel.by_name(model_name).from_config(model_config, **dataeset_options)

        if dist.is_initialized():
            self.model = DDP(self.model,
                             device_ids=[dist.get_rank()],
                             find_unused_parameters=True)

    def train(self, output_dir: str, device: int=-1, local_rank: int=-1):
        os.makedirs(output_dir, exist_ok=True)

        init_logger(logger, f"{output_dir}/train.log", verbose=self.verbose)
        init_device(device, local_rank)
        self.model.to(device)

        optimizer = Adam(self.model.parameters(),
                              self.lr,
                              (self.mu, self.nu),
                              self.epsilon)
        scheduler = ExponentialLR(optimizer, self.decay**(1/self.decay_steps))

        train_loader = DataLoader(self.train_data, self.batch_size, shuffle=True)
        dev_loader = DataLoader(self.dev_data, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False) if self.test_data else None
        metric = -1

        for epoch in range(1, self.epochs + 1):
            start = datetime.now()

            logger.info(f"Epoch {epoch} / {self.epochs}:")

            for data in progress_bar(train_loader):
                subwords = pad_sequence(data["input_ids"].to(device), batch_first=True, padding_value=self.train_data.pad_token_id)
                label_ids = pad_sequence(data["label_ids"].to(device), batch_first=True, padding_value=0)
                pos_ids = pad_sequence(data["pos_ids"].to(device), batch_first=True, padding_value=0) if "pos_ids" in data else None
                mask = label_ids.ne(0)

                out = self.model(subwords, pos_ids)
                loss = self.model.loss(out, label_ids, mask)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                optimizer.step()
                scheduler.step()

            t = datetime.now() - start
            logger.info("dev evaluation")
            dev_loss, dev_acc = self.evaluate(dev_loader, device)

            if dev_acc > metric:
                logger.info("save best model")
                self.save(os.path.join(output_dir, f"best_at_{epoch}.pt"))
                metric = dev_acc

            if test_loader:
                logger.info("test evaluation")
                self.evaluate(test_loader, device)
            logger.info(f"{t}s elapsed\n")

        self.save(os.path.join(output_dir, f"last_at_{epoch}.pt"))
        
        
    @torch.no_grad()
    def evaluate(self, dataloader, device):
        correct = 0
        length = 0
        loss = 0
        for data in progress_bar(dataloader):
                subwords = pad_sequence(data["input_ids"].to(device), batch_first=True, padding_value=self.train_data.pad_token_id)
                label_ids = pad_sequence(data["label_ids"].to(device), batch_first=True, padding_value=0)
                pos_ids = pad_sequence(data["pos_ids"].to(device), batch_first=True, padding_value=0) if "pos_ids" in data else None
                mask = label_ids.ne(0)

                out = self.model(subwords, pos_ids)
                loss += self.model.loss(out, label_ids, mask).detach().cpu().item()
                pred = torch.argmax(out, dim=-1)
                correct += ((pred == label_ids) & mask).sum().detach().cpu().item()
                length += (mask).sum().detach().cpu().item()
        logger.info(f"accuracy: {correct/length*100}, loss: {loss}")
        return loss, correct/length
    
    def save(self, path):
        model = self.model
        if hasattr(model, 'module'):
            model = self.model.module
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(state_dict, path)
