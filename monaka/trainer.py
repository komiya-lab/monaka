# -*- coding: utf-8 -*-

import os
import datetime

import torch
import tqdm

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
            lr: float=2e-5,
            mu: float=.9,
            nu: float=.9,
            epsilon: float=1e-12,
            clip: float=5.0,
            decay: float=.75,
            decay_steps: float=5000,
            patience: float=100,
            evaluate_step:int =20,
            verbose: bool=True,
            **kwargs):
        
        init_logger(logger, verbose=verbose)

        logger.info("loading train files")
        self.train_data = LUWJsonLDataset(train_files, **dataeset_options)

        logger.info("loading dev files")
        self.dev_data = LUWJsonLDataset(dev_files, **dataeset_options)

        logger.info("loading test files")
        self.test_data = LUWJsonLDataset(test_files, **dataeset_options) if test_files else None

        self.batch_size=batch_size
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
        self.evaluate_step = evaluate_step
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
        init_device(str(device), local_rank)
        if dist.is_initialized():
            self.batch_size = self.batch_size // dist.get_world_size()
        self.model.to(device)

        optimizer = Adam(self.model.parameters(),
                              self.lr,
                              (self.mu, self.nu),
                              self.epsilon)
        scheduler = ExponentialLR(optimizer, self.decay**(1/self.decay_steps))

        train_loader = DataLoader(self.train_data, self.batch_size, shuffle=True, collate_fn=LUWJsonLDataset.collate_function)
        dev_loader = DataLoader(self.dev_data, batch_size=self.batch_size, shuffle=False, collate_fn=LUWJsonLDataset.collate_function)
        test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, collate_fn=LUWJsonLDataset.collate_function) if self.test_data else None
        metric = -1

        for epoch in range(1, self.epochs + 1):
            start = datetime.datetime.now()

            logger.info(f"Epoch {epoch} / {self.epochs}:")

            for i, data in tqdm.tqdm(enumerate(train_loader)):
                subwords = pad_sequence(data["input_ids"], batch_first=True, padding_value=self.train_data.pad_token_id).to(device)
                label_ids = pad_sequence(data["label_ids"], batch_first=True, padding_value=1).to(device)
                pos_ids = pad_sequence(data["pos_ids"], batch_first=True, padding_value=1).to(device) if "pos_ids" in data else None
                mask = label_ids.ne(1)

                out = self.model(subwords, pos_ids)
                loss = self.model.loss(out, label_ids, mask)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                optimizer.step()
                scheduler.step()
                if (i+1) % self.evaluate_step == 0:
                    dev_loss, dev_acc = self.evaluate(dev_loader, device)
                    logger.info(f"dev evaluation: loss {dev_loss}, acc {dev_acc}")

            t = datetime.datetime.now() - start
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
        self.model.eval()
        for data in dataloader:
                subwords = pad_sequence(data["input_ids"], batch_first=True, padding_value=self.train_data.pad_token_id).to(device)
                label_ids = pad_sequence(data["label_ids"], batch_first=True, padding_value=1).to(device)
                pos_ids = pad_sequence(data["pos_ids"], batch_first=True, padding_value=1).to(device) if "pos_ids" in data else None
                mask = label_ids.ne(1)

                out = self.model(subwords, pos_ids)
                loss += self.model.loss(out, label_ids, mask).detach().cpu().item()
                pred = torch.argmax(out, dim=-1)
                correct += ((pred == label_ids) & mask).sum().detach().cpu().item()
                length += (mask).sum().detach().cpu().item()
        logger.info(f"accuracy: {correct/length*100}, loss: {loss}")
        self.model.train(True)
        return loss, correct/length
    
    def save(self, path):
        model = self.model
        if hasattr(model, 'module'):
            model = self.model.module
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(state_dict, path)
