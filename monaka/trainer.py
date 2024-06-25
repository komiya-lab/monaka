# -*- coding: utf-8 -*-

import os
import json
import logging
import datetime

import torch
import tqdm

import torch.nn as nn
import torch.distributed as dist
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from registrable import Registrable

from typing import List, Optional, Union, Dict
from monaka.dataset import LUWJsonLDataset
from monaka.mylogging import init_logger, get_logger
from monaka.model import LUWParserModel, LUWLemmaModel, init_device, is_master
from monaka.model import DistributedDataParallel as DDP

import random
import numpy as np

logger = None

def torch_fix_seed(seed=419):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

class Trainer(Registrable):
    
    def __init__(self, *args, **kwargs):
        Registrable.__init__(self)

    def train(self, device: int=-1, local_rank: int=-1):
        raise NotImplementedError
    
    def evaluate(self, dataloader, device):
        raise NotImplementedError

@Trainer.register("segmentation")
class SegmentationTrainer(Trainer):

    def __init__(self,
            train_files: Union[str, List[str]],
            dev_files: Union[str, List[str]],
            test_files: Optional[Union[str, List[str]]],
            dataeset_options: Dict,
            model_name: str,
            model_config: Dict,
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
            seed: int = 419,
            output_dir: str="",
            **kwargs):
        
        global logger
        os.makedirs(output_dir, exist_ok=True)

        logger = get_logger(f"monaka.trainer.{output_dir.replace('/', '.')}")
        init_logger(logger, handlers=[logging.StreamHandler(), logging.FileHandler(f"{output_dir}/train.log", 'w')], verbose=verbose)
        self.output_dir = output_dir

        logger.info("dataset options:")
        logger.info(json.dumps(dataeset_options, indent=True, ensure_ascii=False))
        options = {"logger": logger}
        options.update(dataeset_options)
        logger.info("loading train files")
        self.train_data = LUWJsonLDataset(train_files, **options)

        label_dic = self.train_data.label_dic
        with open(os.path.join(output_dir, "labels.json"), "w") as f:
            json.dump(label_dic, f, indent=True, ensure_ascii=False)

        pos_dic = getattr(self.train_data, "pos_dic", None)
        if pos_dic is not None:
            with open(os.path.join(output_dir, "pos.json"), "w") as f:
                json.dump(pos_dic, f, indent=True, ensure_ascii=False)

        logger.info("loading dev files")
        self.dev_data = LUWJsonLDataset(dev_files, **options)

        logger.info("loading test files")
        self.test_data = LUWJsonLDataset(test_files, **options) if test_files else None

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
        conf = {
            "batch_size": batch_size,
            "epochs": epochs,
            "mu": mu,
            "nu": nu,
            "epsilon": epsilon,
            "clip": clip,
            "decay": decay,
            "decay_steps": decay_steps,
            "patience": patience,
            "verbose": verbose,
            "evaluate_step": evaluate_step,
            "seed": seed
        }
        torch_fix_seed(seed)
        conf.update(kwargs)

        logger.info("loading model")
        self.model = LUWParserModel.by_name(model_name).from_config(model_config, **dataeset_options)
        logger.info(str(self.model))
        logger.info(json.dumps(model_config, indent=True, ensure_ascii=False))

        logger.info("training setup:")
        logger.info(json.dumps(conf, indent=True, ensure_ascii=False))

        if dist.is_initialized():
            logger.info("distributed mode ON")
            self.model = DDP(self.model,
                             device_ids=[dist.get_rank()],
                             find_unused_parameters=True)

    def train(self, device: int=-1, local_rank: int=-1):
        init_device(str(device), local_rank)
        if dist.is_initialized():
            self.batch_size = self.batch_size // dist.get_world_size()
        try:
            device = int(device)
        except:
            pass
        self.model.to(device)

        optimizer = Adam(self.model.parameters(),
                              self.lr,
                              (self.mu, self.nu),
                              self.epsilon)
        scheduler = ExponentialLR(optimizer, self.decay**(1/self.decay_steps))
        writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "tb"))

        train_loader = DataLoader(self.train_data, self.batch_size, shuffle=True, collate_fn=LUWJsonLDataset.collate_function)
        dev_loader = DataLoader(self.dev_data, batch_size=self.batch_size, shuffle=False, collate_fn=LUWJsonLDataset.collate_function)
        test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, collate_fn=LUWJsonLDataset.collate_function) if self.test_data else None
        metric = -1
        total_itr = 0

        for epoch in range(1, self.epochs + 1):
            start = datetime.datetime.now()

            logger.info(f"Epoch {epoch} / {self.epochs}:")

            for i, data in tqdm.tqdm(enumerate(train_loader)):
                subwords = pad_sequence(data["input_ids"], batch_first=True, padding_value=self.train_data.pad_token_id).to(device)
                word_ids = pad_sequence([torch.LongTensor(js.word_ids()) for js in data["subwords"]], batch_first=True, padding_value=-1).to(device)
                label_ids = pad_sequence(data["label_ids"], batch_first=True, padding_value=1).to(device)
                pos_ids = pad_sequence(data["pos_ids"], batch_first=True, padding_value=1).to(device) if "pos_ids" in data else None
                mask = label_ids.ne(1)

                out = self.model(subwords, word_ids, pos_ids)
                loss = self.model.loss(out, label_ids, mask)
                writer.add_scalar("Loss/train", loss, total_itr + i)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                optimizer.step()
                scheduler.step()
                if (i+1) % self.evaluate_step == 0:
                    dev_loss, dev_acc = self.evaluate(dev_loader, device)
                    writer.add_scalar("Loss/dev", dev_loss, total_itr + i)
                    writer.add_scalar("Acc/dev", dev_acc, total_itr + i)

            total_itr += i
            t = datetime.datetime.now() - start
            logger.info("dev evaluation")
            dev_loss, dev_acc = self.evaluate(dev_loader, device)
            writer.add_scalar("Loss/dev", dev_loss, total_itr)
            writer.add_scalar("Acc/dev", dev_acc, total_itr)

            if dev_acc > metric:
                logger.info("save best model")
                self.save(os.path.join(self.output_dir, f"best_at_{epoch}.pt"))
                metric = dev_acc

            if test_loader:
                logger.info("test evaluation")
                test_loss, test_acc = self.evaluate(test_loader, device)
                writer.add_scalar("Loss/test", test_loss, total_itr)
                writer.add_scalar("Acc/test", test_acc, total_itr)
            logger.info(f"{t}s elapsed\n")

        self.save(os.path.join(self.output_dir, f"last_at_{epoch}.pt"))
        
        
    @torch.no_grad()
    def evaluate(self, dataloader, device):
        correct = 0
        length = 0
        loss = 0
        self.model.eval()
        for data in dataloader:
                subwords = pad_sequence(data["input_ids"], batch_first=True, padding_value=self.train_data.pad_token_id).to(device)
                word_ids = pad_sequence([torch.LongTensor(js.word_ids()) for js in data["subwords"]], batch_first=True, padding_value=-1).to(device)
                label_ids = pad_sequence(data["label_ids"], batch_first=True, padding_value=1).to(device)
                pos_ids = pad_sequence(data["pos_ids"], batch_first=True, padding_value=1).to(device) if "pos_ids" in data else None
                mask = label_ids.ne(1)

                out = self.model(subwords, word_ids, pos_ids)
                loss += self.model.loss(out, label_ids, mask).detach().cpu().item()
                pred = torch.argmax(out, dim=-1)
                try:
                    correct += ((pred == label_ids) & mask).sum().detach().cpu().item()
                    length += (mask).sum().detach().cpu().item()
                except:
                    logger.info(f"evaluation skipped: {data['sentence']}")
        logger.info(f"accuracy: {correct/length*100}, loss: {loss}")
        self.model.train(True)
        return loss, correct/length
    
    def save(self, path):
        model = self.model
        if hasattr(model, 'module'):
            model = self.model.module
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(state_dict, path)

@Trainer.register("lemma")
class LemmaTrainer(Trainer):

    def __init__(self,
            train_files: Union[str, List[str]],
            dev_files: Union[str, List[str]],
            test_files: Optional[Union[str, List[str]]],
            dataeset_options: Dict,
            model_name: str,
            model_config: Dict,
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
            seed: int = 419,
            output_dir: str="",
            **kwargs):
        
        global logger
        os.makedirs(output_dir, exist_ok=True)

        logger = get_logger(f"monaka.trainer.lemma.{output_dir.replace('/', '.')}")
        init_logger(logger, handlers=[logging.StreamHandler(), logging.FileHandler(f"{output_dir}/train.lemma.log", 'w')], verbose=verbose)
        self.output_dir = output_dir

        logger.info("dataset options:")
        logger.info(json.dumps(dataeset_options, indent=True, ensure_ascii=False))
        options = {"logger": logger}
        options.update(dataeset_options)
        logger.info("loading train files")
        self.train_data = LUWJsonLDataset(train_files, **options)

        logger.info("loading dev files")
        self.dev_data = LUWJsonLDataset(dev_files, **options)

        logger.info("loading test files")
        self.test_data = LUWJsonLDataset(test_files, **options) if test_files else None

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
        conf = {
            "batch_size": batch_size,
            "epochs": epochs,
            "mu": mu,
            "nu": nu,
            "epsilon": epsilon,
            "clip": clip,
            "decay": decay,
            "decay_steps": decay_steps,
            "patience": patience,
            "verbose": verbose,
            "evaluate_step": evaluate_step,
            "seed": seed
        }
        torch_fix_seed(seed)
        conf.update(kwargs)

        logger.info("loading model")
        self.model = LUWLemmaModel.by_name(model_name).from_config(model_config, **dataeset_options)
        logger.info(str(self.model))
        logger.info(json.dumps(model_config, indent=True, ensure_ascii=False))

        logger.info("training setup:")
        logger.info(json.dumps(conf, indent=True, ensure_ascii=False))

        if dist.is_initialized():
            logger.info("distributed mode ON")
            self.model = DDP(self.model,
                             device_ids=[dist.get_rank()],
                             find_unused_parameters=True)
            
    @staticmethod
    def batch_lemma_target(data: List[List[int]]):
        prv = 0
        res = []
        #print(data)
        for d in data:
            l = [v + prv for v in d]
            res.append(torch.tensor(l))
            prv = max(l) + 1
        return res

    def train(self, device: int=-1, local_rank: int=-1):
        init_device(str(device), local_rank)
        if dist.is_initialized():
            self.batch_size = self.batch_size // dist.get_world_size()
        try:
            device = int(device)
            logger.info(f"device: {device}")
        except:
            logger.warn(f"device is not int: {device}")
            pass
        self.model.to(device)

        optimizer = Adam(self.model.parameters(),
                              self.lr,
                              (self.mu, self.nu),
                              self.epsilon)
        scheduler = ExponentialLR(optimizer, self.decay**(1/self.decay_steps))
        writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "tb"))

        train_loader = DataLoader(self.train_data, self.batch_size, shuffle=True, collate_fn=LUWJsonLDataset.collate_function)
        dev_loader = DataLoader(self.dev_data, batch_size=1, shuffle=False, collate_fn=LUWJsonLDataset.collate_function)
        test_loader = DataLoader(self.test_data, batch_size=1, shuffle=False, collate_fn=LUWJsonLDataset.collate_function) if self.test_data else None
        metric = -1
        total_itr = 0

        for epoch in range(1, self.epochs + 1):
            start = datetime.datetime.now()

            logger.info(f"Epoch {epoch} / {self.epochs}:")

            for i, data in tqdm.tqdm(enumerate(train_loader)):
                subwords = pad_sequence(data["input_ids"], batch_first=True, padding_value=self.train_data.pad_token_id).to(device)
                # lemma List[Tensor[lemma, subwords]]
                l = list()
                for d in data["lemma_ids"]:
                    [l.append(v) for v in d]
                label_ids = pad_sequence(l,  batch_first=True, padding_value=self.train_data.pad_token_id).to(device)
                lemma_target = pad_sequence(self.batch_lemma_target(data["lemma_target"]), batch_first=True, padding_value=self.train_data.pad_token_id).to(device)
                #label_ids = torch.flatten(label_ids, 0, 1) # batch, subwords
                #print(label_ids.size())
                mask = label_ids.ne(self.train_data.pad_token_id)

                #print(subwords.size(), lemma_target.size())
                out = self.model(subwords, lemma_target) # batch * luw, vocab size
                loss = self.model.loss(out, label_ids, mask)
                writer.add_scalar("Loss/train", loss, total_itr + i)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                optimizer.step()
                scheduler.step()
                if (i+1) % self.evaluate_step == 0:
                    dev_loss, dev_acc = self.evaluate(dev_loader, device)
                    writer.add_scalar("Loss/dev", dev_loss, total_itr + i)
                    writer.add_scalar("Acc/dev", dev_acc, total_itr + i)

            total_itr += i
            t = datetime.datetime.now() - start
            logger.info("dev evaluation")
            dev_loss, dev_acc = self.evaluate(dev_loader, device)
            writer.add_scalar("Loss/dev", dev_loss, total_itr)
            writer.add_scalar("Acc/dev", dev_acc, total_itr)

            if dev_acc > metric:
                logger.info("save best model")
                self.save(os.path.join(self.output_dir, f"best_at_{epoch}.pt"))
                metric = dev_acc

            if test_loader:
                logger.info("test evaluation")
                test_loss, test_acc = self.evaluate(test_loader, device)
                writer.add_scalar("Loss/test", test_loss, total_itr)
                writer.add_scalar("Acc/test", test_acc, total_itr)
            logger.info(f"{t}s elapsed\n")

        self.save(os.path.join(self.output_dir, f"last_at_{epoch}.pt"))
        
        
    @torch.no_grad()
    def evaluate(self, dataloader, device):
        correct = 0
        length = 0
        loss = 0
        self.model.eval()
        for data in dataloader:
                subwords = pad_sequence(data["input_ids"], batch_first=True, padding_value=self.train_data.pad_token_id).to(device)
                # lemma List[Tensor[lemma, subwords]]
                l = list()
                for d in data["lemma_ids"]:
                    [l.append(v) for v in d]
                label_ids = pad_sequence(l,  batch_first=True, padding_value=self.train_data.pad_token_id).to(device)
                lemma_target = pad_sequence(self.batch_lemma_target(data["lemma_target"]), batch_first=True, padding_value=self.train_data.pad_token_id).to(device)
                #label_ids = torch.flatten(label_ids, 0, 1) # batch, subwords
                #print(label_ids.size())
                mask = label_ids.ne(self.train_data.pad_token_id)

                out = self.model(subwords, lemma_target) # batch * luw, vocab size
                if out.size()[0] != label_ids.size()[0] or out.size()[1] < label_ids.size()[1]:
                    logger.warn(f"eval: unmatch output and label size: {data['sentence']}, {out.size()}, {label_ids.size()}")
                    continue
                loss += self.model.loss(out, label_ids, mask).detach().cpu().item()
                pred = torch.argmax(out, dim=-1)
                lsize = label_ids.size()
                #print(label_ids.size(), pred.size())
                try:
                    correct += ((pred[:lsize[0], :lsize[1]] == label_ids)).sum().detach().cpu().item()
                    length += lsize[0] * lsize[1]
                except Exception as e:
                    logger.info(f"evaluation skipped: {data['sentence']}")
                    raise e
        logger.info(f"accuracy: {correct/length*100}, loss: {loss}")
        self.model.train(True)
        return loss, correct/length
    
    def save(self, path):
        model = self.model
        if hasattr(model, 'module'):
            model = self.model.module
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(state_dict, path)
