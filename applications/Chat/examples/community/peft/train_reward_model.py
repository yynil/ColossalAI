import argparse
from random import randint

import loralib as lora
import torch
from coati.dataset import HhRlhfDataset, RmStaticDataset
from coati.models import LogExpLoss, LogSigLoss
from coati.models.base import RewardModel
from coati.models.bloom import BLOOMRM
from coati.models.deberta import DebertaRM
from coati.models.gpt import GPTRM
from coati.models.llama import LlamaRM
from coati.models.opt import OPTRM
from coati.models.roberta import RoBERTaRM
from coati.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy
from coati.utils import prepare_llama_tokenizer_and_embedding
from datasets import load_dataset
from torch.optim import Adam
from transformers import AutoTokenizer, BloomTokenizerFast, DebertaV2Tokenizer, LlamaTokenizer, RobertaTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

from colossalai.nn.optimizer import HybridAdam
from coati.models.glm import ChatGLMRM
from easy_dataset import EasyRewardDataset
import os
import datasets

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
from coati.trainer.strategies.base import Strategy
from coati.trainer.utils import is_rank_0
from coati.models.utils import compute_reward, normalize
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from torch.optim import Optimizer, lr_scheduler
import torch.distributed as dist
import wandb
import time
import pandas as pd

class RewardModelTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        loss_fn (callable): the loss function to use for training
        train_dataset (Dataset): the dataset to use for training
        valid_dataset (Dataset): the dataset to use for validation
        eval_dataset (Dataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
    """

    def __init__(
        self,
        model,
        strategy: Strategy,
        optim: Optimizer,
        loss_fn,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        eval_dataset: Dataset,
        batch_size: int = 1,
        max_epochs: int = 1,
        save_every_step: int = 1000,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        train_sampler = None

        if dist.is_initialized() and dist.get_world_size() > 1:
            train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=42, drop_last=True)
        self.train_dataloader = DataLoader(train_dataset,
                                           shuffle=(train_sampler is None),
                                           sampler=train_sampler,
                                           batch_size=batch_size,collate_fn=EasyRewardDataset.collate_fn)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,collate_fn=EasyRewardDataset.collate_fn)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True,collate_fn=EasyRewardDataset.collate_fn)

        self.model = strategy.setup_model(model)
        self.loss_fn = loss_fn
        self.optimizer = strategy.setup_optimizer(optim, self.model)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, self.train_dataloader.__len__() // 100)
        self.save_every_step = save_every_step

    def eval_acc(self, dataloader):
        dist = 0
        on = 0
        cnt = 0
        self.model.eval()
        with torch.no_grad():
            for chosen_ids, c_mask, reject_ids, r_mask in dataloader:
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                chosen_reward = self.model(chosen_ids, attention_mask=c_mask)
                reject_reward = self.model(reject_ids, attention_mask=r_mask)
                for i in range(len(chosen_reward)):
                    cnt += 1
                    if chosen_reward[i] > reject_reward[i]:
                        on += 1
                dist += (chosen_reward - reject_reward).mean().item()
            dist_mean = dist / len(dataloader)
            acc = on / cnt
        self.model.train()
        return dist_mean, acc

    def fit(self):
        #init a string with YYYY_MM_DD_HH_MM_SS
        time_str = time.strftime("%Y_%m_%d_%H_%M_%S")
        epoch_bar = tqdm(range(self.epochs), desc='Train epoch', disable=not is_rank_0())
        for epoch in range(self.epochs):
            step_bar = tqdm(range(self.train_dataloader.__len__()),
                            desc='Train step of epoch %d' % epoch,
                            disable=not is_rank_0())
            # train
            self.model.train()
            cnt = 0
            acc = 0
            dist = 0
            for chosen_ids, c_mask, reject_ids, r_mask in self.train_dataloader:
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                # print(f'0.cuda memory usage {torch.cuda.memory_allocated()/1024/1024}  MB')
                chosen_reward = self.model(chosen_ids, attention_mask=c_mask)
                # print(f'1.cuda memory usage {torch.cuda.memory_allocated()/1024/1024}  MB')
                reject_reward = self.model(reject_ids, attention_mask=r_mask)
                # print(f'1.cuda memory usage {torch.cuda.memory_allocated()/1024/1024}  MB')
                loss = self.loss_fn(chosen_reward, reject_reward)
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer)
                self.optimizer.zero_grad()
                cnt += 1
                if cnt == 100:
                    self.scheduler.step()
                    dist, acc = self.eval_acc(self.valid_dataloader)
                    cnt = 0
                    if is_rank_0():
                        log = pd.DataFrame([[step_bar.n, loss.item(), dist, acc]],
                                           columns=['step', 'loss', 'dist', 'acc'])
                        log_file = f'log_{time_str}.csv'
                        if not os.path.exists(log_file):
                            mode = 'w'
                        else:
                            mode = 'a'
                        log.to_csv(log_file, mode=mode, header=False, index=False)
                step_bar.update()
                step_bar.set_postfix({'dist': dist, 'acc': acc})

            # eval
            dist, acc = self.eval_acc(self.eval_dataloader)
            if is_rank_0():
                log = pd.DataFrame([[step_bar.n, loss.item(), dist, acc]], columns=['step', 'loss', 'dist', 'acc'])
                log_file = 'log.csv'
                if not os.path.exists(log_file):
                    mode = 'w'
                else:
                    mode = 'a'
                log.to_csv(log_file, mode=mode, header=False, index=False)
            epoch_bar.update()
            step_bar.set_postfix({'dist': dist, 'acc': acc})
            step_bar.close()

    def save_model(self,
                   path: str,
                   only_rank0: bool = False,
                   tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
        self.strategy.save_model(model=self.model, path=path, only_rank0=only_rank0, tokenizer=tokenizer)



def train(args):
    # configure strategy
    if args.strategy == 'naive':
        strategy = NaiveStrategy()
    elif args.strategy == 'ddp':
        strategy = DDPStrategy()
    elif args.strategy == 'colossalai_gemini':
        strategy = ColossalAIStrategy(stage=3, placement_policy='cuda')
    elif args.strategy == 'colossalai_zero2':
        strategy = ColossalAIStrategy(stage=2, placement_policy='cuda',precision='fp16')
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')

    # configure model
    with strategy.model_init_context():
        if args.model == 'bloom':
            model = BLOOMRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        elif args.model == 'opt':
            model = OPTRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        elif args.model == 'gpt2':
            model = GPTRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        elif args.model == 'deberta':
            model = DebertaRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        elif args.model == 'llama':
            model = LlamaRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        elif args.model == 'roberta':
            model = RoBERTaRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        elif args.model == 'chatglm':
            model = ChatGLMRM(pretrained=args.pretrain, lora_rank=args.lora_rank,lora_path=args.model_path).to(torch.cuda.current_device())
        else:
            raise ValueError(f'Unsupported model "{args.model}"')

        # if args.model_path is not None and os.path.exists(args.model_path):
        #     print(model.state_dict().keys())
        #     print(f'loading model from {args.model_path}')
        #     state_dict = torch.load(args.model_path,map_location=torch.device('cpu'))
        #     print("-----------")
        #     print(state_dict.keys())
        #     model.to("cpu")
        #     model.load_state_dict(state_dict, strict=True)
        #     del state_dict
        #     model.to(torch.cuda.current_device())
    print(model)
    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )
    print_trainable_parameters(model)
    print(f'current cuda memory is {torch.cuda.memory_allocated()/1024/1024} MB')
    # configure tokenizer
    if args.model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    elif args.model == 'bloom':
        tokenizer = BloomTokenizerFast.from_pretrained('bigscience/bloom-560m')
    elif args.model == 'opt':
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    elif args.model == 'deberta':
        tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-large')
    elif args.model == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained(args.pretrain)
    elif args.model == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    elif args.model == 'chatglm':
        tokenizer = AutoTokenizer.from_pretrained(args.pretrain, trust_remote_code=True)
    else:
        raise ValueError(f'Unsupported model "{args.model}"')
    max_len = args.max_len

    if args.model == 'llama':
        tokenizer = prepare_llama_tokenizer_and_embedding(tokenizer, model)
    else:
        tokenizer.pad_token = tokenizer.eos_token

    # configure optimizer
    if args.strategy.startswith('colossalai'):
        optim = HybridAdam(model.parameters(), lr=5e-6)
    else:
        optim = Adam(model.parameters(), lr=5e-6)

    # configure loss function
    if args.loss_fn == 'log_sig':
        loss_fn = LogSigLoss()
    elif args.loss_fn == 'log_exp':
        loss_fn = LogExpLoss()
    else:
        raise ValueError(f'Unsupported loss function "{args.loss_fn}"')

    # configure dataset
    if args.train_file is not None:
        #if the train_file is a file , we use the EasyRewardDataset
        if os.path.isfile(args.train_file):
            train_dataset = EasyRewardDataset(args.train_file, tokenizer,max_length= max_len)
        #if the train_file is a directory, we use datasets.load_from_disk
        elif os.path.isdir(args.train_file):
            train_dataset = datasets.load_from_disk(args.train_file)
        print(f'train dataset {train_dataset}')
    if args.valid_file is not None:
        if os.path.isfile(args.valid_file):
            valid_dataset = EasyRewardDataset(args.valid_file, tokenizer,max_length= max_len)
        elif os.path.isdir(args.valid_file):
            valid_dataset = datasets.load_from_disk(args.valid_file)
        print(f'valid dataset {valid_dataset}')
    else:
        valid_dataset = None
    if args.eval_file is not None:
        if os.path.isfile(args.eval_file):
            eval_dataset = EasyRewardDataset(args.eval_file, tokenizer,max_length= max_len)
        elif os.path.isdir(args.eval_file): 
            eval_dataset = datasets.load_from_disk(args.eval_file)
        print(f'eval dataset {eval_dataset}')
    else:
        eval_dataset = None
    
    print(f'current cuda memory is {torch.cuda.memory_allocated()/1024/1024} MB')
    trainer = RewardModelTrainer(model=model,
                                 strategy=strategy,
                                 optim=optim,
                                 loss_fn=loss_fn,
                                 train_dataset=train_dataset,
                                 valid_dataset=valid_dataset,
                                 eval_dataset=eval_dataset,
                                 batch_size=args.batch_size,
                                 max_epochs=args.max_epochs)

    trainer.fit()
    # save model checkpoint after fitting on only rank0
    trainer.save_model(path=args.save_path, only_rank0=True, tokenizer=tokenizer)
    # save optimizer checkpoint on all ranks
    if args.need_optim_ckpt:
        strategy.save_optimizer(trainer.optimizer,
                                'rm_optim_checkpoint_%d.pt' % (torch.cuda.current_device()),
                                only_rank0=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy',
                        choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2'],
                        default='naive')
    parser.add_argument('--model', choices=['gpt2', 'bloom', 'opt', 'deberta', 'llama', 'roberta','chatglm'], default='chatglm')
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--need_optim_ckpt', type=bool, default=False)
    parser.add_argument('--train_file',type=str, default=None)
    parser.add_argument('--valid_file',type=str, default=None)
    parser.add_argument('--eval_file',type=str, default=None)
    parser.add_argument('--subset', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='rm_ckpt')
    parser.add_argument('--max_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--lora_rank', type=int, default=32, help="low-rank adaptation matrices rank")
    parser.add_argument('--loss_fn', type=str, default='log_sig', choices=['log_sig', 'log_exp'])
    args = parser.parse_args()
    train(args)
