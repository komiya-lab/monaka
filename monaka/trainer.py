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
from monaka.dataset import LUWJsonLDataset, LemmaJsonDataset, ChunkDepJsonLDataset
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


@Trainer.register("dependency")
class DependencyTrainer(Trainer):

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
        self.train_data = ChunkDepJsonLDataset(train_files, **options)

        label_dic = self.train_data.label_dic
        with open(os.path.join(output_dir, "labels.json"), "w") as f:
            json.dump(label_dic, f, indent=True, ensure_ascii=False)

        rel_dic = self.train_data.rel_dic
        with open(os.path.join(output_dir, "rels.json"), "w") as f:
            json.dump(rel_dic, f, indent=True, ensure_ascii=False)

        pos_dic = getattr(self.train_data, "pos_dic", None)
        if pos_dic is not None:
            with open(os.path.join(output_dir, "pos.json"), "w") as f:
                json.dump(pos_dic, f, indent=True, ensure_ascii=False)

        logger.info("loading dev files")
        self.dev_data = ChunkDepJsonLDataset(dev_files, **options)

        logger.info("loading test files")
        self.test_data = ChunkDepJsonLDataset(test_files, **options) if test_files else None

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

        train_loader = DataLoader(self.train_data, self.batch_size, shuffle=True, collate_fn=ChunkDepJsonLDataset.collate_function)
        dev_loader = DataLoader(self.dev_data, batch_size=self.batch_size, shuffle=False, collate_fn=ChunkDepJsonLDataset.collate_function)
        test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, collate_fn=ChunkDepJsonLDataset.collate_function) if self.test_data else None
        metric = -1
        total_itr = 0

        for epoch in range(1, self.epochs + 1):
            start = datetime.datetime.now()

            logger.info(f"Epoch {epoch} / {self.epochs}:")

            for i, data in tqdm.tqdm(enumerate(train_loader)):
                subwords = pad_sequence(data["input_ids"], batch_first=True, padding_value=self.train_data.pad_token_id).to(device)
                word_ids = pad_sequence([torch.LongTensor(js.word_ids()) for js in data["subwords"]], batch_first=True, padding_value=-1).to(device)
                chunk_ids = pad_sequence(data["chunk_ids"], batch_first=True, padding_value=-1).to(device)
                dep_ids = pad_sequence(data["dep_ids"], batch_first=True, padding_value=-1).to(device)
                word_rel_ids = pad_sequence(data["word_rel_ids"], batch_first=True, padding_value=self.train_data.pad_token_id).to(device)
                dep_rel_ids = pad_sequence(data["dep_rel_ids"], batch_first=True, padding_value=self.train_data.pad_token_id).to(device)
                pos_ids = pad_sequence(data["pos_ids"], batch_first=True, padding_value=1).to(device) if "pos_ids" in data else None
                wmask = word_rel_ids.ne(1)
                dmask = dep_ids.ne(-1)
                rmask = dep_rel_ids.ne(1)

                dep_out, deprel_out, word_out  = self.model(subwords, word_ids, chunk_ids, pos_ids)
                loss, dep_loss, rel_loss, wrd_loss = self.model.loss(dep_out, deprel_out, word_out , dep_ids, dep_rel_ids, word_rel_ids, wmask, dmask, rmask)
                writer.add_scalar("Loss/train", loss, total_itr + i)
                writer.add_scalar("DependencyLoss/train", dep_loss, total_itr + i)
                writer.add_scalar("RelationLoss/train", rel_loss, total_itr + i)
                writer.add_scalar("WordRelLoss/train", wrd_loss, total_itr + i)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                optimizer.step()
                scheduler.step()
                if (i+1) % self.evaluate_step == 0:
                    dev_loss, dev_dep_loss, dev_rel_loss, dev_wrd_loss, dev_dep_acc, dev_rel_acc, dev_wrd_acc = self.evaluate(dev_loader, device)
                    writer.add_scalar("Loss/dev", dev_loss, total_itr + i)
                    writer.add_scalar("DependencyLoss/dev", dev_dep_loss, total_itr + i)
                    writer.add_scalar("RelationLoss/dev", dev_rel_loss, total_itr + i)
                    writer.add_scalar("WordRelLoss/dev", dev_wrd_loss, total_itr + i)
                    writer.add_scalar("DependencyAcc/dev", dev_dep_acc, total_itr + i)
                    writer.add_scalar("RelationAcc/dev", dev_rel_acc, total_itr + i)
                    writer.add_scalar("WordRelAcc/dev", dev_wrd_acc, total_itr + i)

            total_itr += i
            t = datetime.datetime.now() - start
            logger.info("dev evaluation")
            dev_loss, dev_dep_loss, dev_rel_loss, dev_wrd_loss, dev_dep_acc, dev_rel_acc, dev_wrd_acc = self.evaluate(dev_loader, device)
            writer.add_scalar("Loss/dev", dev_loss, total_itr)
            writer.add_scalar("DependencyLoss/dev", dev_dep_loss, total_itr)
            writer.add_scalar("RelationLoss/dev", dev_rel_loss, total_itr)
            writer.add_scalar("WordRelLoss/dev", dev_wrd_loss, total_itr)
            writer.add_scalar("DependencyAcc/dev", dev_dep_acc, total_itr)
            writer.add_scalar("RelationAcc/dev", dev_rel_acc, total_itr)
            writer.add_scalar("WordRelAcc/dev", dev_wrd_acc, total_itr)

            if dev_dep_acc > metric:
                logger.info("save best model")
                self.save(os.path.join(self.output_dir, f"best.pt"))
                metric = dev_dep_acc

            if test_loader:
                logger.info("test evaluation")
                test_loss, test_dep_loss, test_rel_loss, test_wrd_loss, test_dep_acc, test_rel_acc, test_wrd_acc = self.evaluate(test_loader, device)
                writer.add_scalar("Loss/test", test_loss, total_itr)
                writer.add_scalar("DependencyLoss/test", test_dep_loss, total_itr)
                writer.add_scalar("RelationLoss/test", test_rel_loss, total_itr)
                writer.add_scalar("WordRelLoss/test", test_wrd_loss, total_itr)
                writer.add_scalar("DependencyAcc/test", test_dep_acc, total_itr)
                writer.add_scalar("RelationAcc/test", test_rel_acc, total_itr)
                writer.add_scalar("WordRelAcc/test", test_wrd_acc, total_itr)

            logger.info(f"{t}s elapsed\n")

        self.save(os.path.join(self.output_dir, f"last_at_{epoch}.pt"))
        
        
    @torch.no_grad()
    def evaluate(self, dataloader, device):
        dep_correct = 0
        dep_length = 0
        rel_correct = 0
        rel_length = 0
        wrd_correct = 0
        wrd_length = 0

        loss = 0
        dep_loss = 0
        rel_loss = 0
        wrd_loss = 0
        self.model.eval()
        for data in dataloader:
                subwords = pad_sequence(data["input_ids"], batch_first=True, padding_value=self.train_data.pad_token_id).to(device)
                word_ids = pad_sequence([torch.LongTensor(js.word_ids()) for js in data["subwords"]], batch_first=True, padding_value=-1).to(device)
                chunk_ids = pad_sequence(data["chunk_ids"], batch_first=True, padding_value=-1).to(device)
                dep_ids = pad_sequence(data["dep_ids"], batch_first=True, padding_value=-1).to(device)
                word_rel_ids = pad_sequence(data["word_rel_ids"], batch_first=True, padding_value=1).to(device)
                dep_rel_ids = pad_sequence(data["dep_rel_ids"], batch_first=True, padding_value=1).to(device)
                pos_ids = pad_sequence(data["pos_ids"], batch_first=True, padding_value=1).to(device) if "pos_ids" in data else None
                wmask = word_rel_ids.ne(1)
                dmask = dep_ids.ne(-1)
                rmask = dep_rel_ids.ne(1)

                dep_out, deprel_out, word_out  = self.model(subwords, word_ids, chunk_ids, pos_ids)
                l, dep_l, rel_l, wrd_l = self.model.loss(dep_out, deprel_out, word_out , dep_ids, dep_rel_ids, word_rel_ids, wmask, dmask, rmask)
                loss += l.detach().cpu().item()
                dep_loss += dep_l.detach().cpu().item()
                rel_loss += rel_l.detach().cpu().item()
                wrd_loss += wrd_l.detach().cpu().item()
                dep_pred = torch.argmax(dep_out, dim=-1)
                rel_pred = torch.argmax(deprel_out, dim=1)
                wrd_pred = torch.argmax(word_out, dim=-1)
                try:
                    dep_correct += ((dep_pred == dep_ids) & dmask).sum().detach().cpu().item()
                    dep_length += (dmask).sum().detach().cpu().item()
                    rel_size = rel_pred.size()
                    rel_correct += ((rel_pred == dep_rel_ids[:, :rel_size[1], :rel_size[2]]) & rmask[:, :rel_size[1], :rel_size[2]]).sum().detach().cpu().item()
                    rel_length += (rmask[:, :rel_size[1], :rel_size[2]]).sum().detach().cpu().item()
                    wrd_correct += ((wrd_pred == word_rel_ids) & wmask).sum().detach().cpu().item()
                    wrd_length += (wmask).sum().detach().cpu().item()
                except Exception as e:
                    raise e
                    logger.info(f"evaluation skipped: {data['text']}")
        logger.info(f"dep accuracy: {dep_correct/dep_length*100}, rel accuracy: {rel_correct/rel_length*100}, word rel accuracy: {wrd_correct/wrd_length*100}, loss: {loss}")
        self.model.train(True)
        return loss, dep_loss, rel_loss, wrd_loss, dep_correct/dep_length, rel_correct/rel_length, wrd_correct/wrd_length
    
    def save(self, path):
        model = self.model
        if hasattr(model, 'module'):
            model = self.model.module
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(state_dict, path)


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

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer as TTrainer, AutoModelForSeq2SeqLM


@Trainer.register("lemma-decoder")
class LemmaDeocderTrainer(Trainer):

    def __init__(self,
            train_files: Union[str, List[str]],
            dev_files: Union[str, List[str]],
            test_files: Optional[Union[str, List[str]]],
            dataset_options: Dict,
            model_name: str,
            model_config: Dict,
            batch_size: int=8,
            epochs: int=1,
            steps: int=1,
            lr: float=2e-5,
            decay: float=.75,
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
        self.training_config = Seq2SeqTrainingArguments(
            output_dir = output_dir,
            num_train_epochs = epochs, 
            max_steps = steps,
            evaluation_strategy="steps",
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size = 2,
            eval_accumulation_steps = 100,
            learning_rate = lr,
            weight_decay = decay,
            save_steps = evaluate_step * 4,
            eval_steps=evaluate_step,
            logging_dir=output_dir,
            seed=seed,
            do_eval = True
        )
        logger.info(train_files)
        logger.info(dev_files)
        self.dataset_options = dataset_options
        self.train_dataset = LemmaJsonDataset(train_files, **dataset_options)
        self.dev_dataset = LemmaJsonDataset(dev_files, **dataset_options)
        self.test_dataset = None
        if test_files:
            self.test_dataset = LemmaJsonDataset(test_files, **dataset_options)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.trainer = TTrainer(self.model, args=self.training_config, train_dataset=self.train_dataset, eval_dataset=self.dev_dataset, compute_metrics=self.compute_metrics)
        self.tokenizer = self.train_dataset.tokenizer

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        #preds_ = np.argmax(preds)
        preds_ = [np.argmax(prd, axis=-1) for prd in preds]
        logger.info(len(preds_))
        logger.info(preds_[0].shape)

        # decode preds and labels
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds_[0], skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        correct = [1 for p,l in zip(decoded_preds, decoded_labels) if p.strip() == l.strip()]
        return {"accuracy": len(correct) / len(decoded_preds), "preds": decoded_preds, "labels": decoded_labels}
    
    def train(self, device, local_rank):
        self.trainer.train()
        self.trainer.save_model(os.path.join(self.output_dir, "last-checkpoint"))
        if self.test_dataset:
            metrics = self.trainer.evaluate(self.test_dataset)
            logger.info(metrics)
            with open(os.path.join(self.output_dir, "predict.json"), 'w') as f:
                json.dump(metrics, f, indent=True, ensure_ascii=False)

    @torch.no_grad()
    def evaluate(self):
        pass