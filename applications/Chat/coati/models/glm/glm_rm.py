import torch
import torch.nn as nn
from torch.nn import Module 
from transformers import AutoModel
from peft import PeftModel,LoraConfig,TaskType,get_peft_model
from peft.tuners.lora import LoraLayer
from coati.models.utils import masked_mean
import os
from typing import Optional

class ChatGLMRM(Module):
    """
    ChatGLMRM Reward model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (BloomConfig): Model config.
        checkpoint (bool): Enable gradient checkpointing.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self,
                 pretrained: str = None,
                 lora_path :str = None,
                 lora_rank :int = 0,
                 pad_id :int = 3) -> None:
        super().__init__()
        self.pad_id = pad_id
        if pretrained is not None:
            model = AutoModel.from_pretrained(
                pretrained,
                trust_remote_code=True,
            ).half().cpu() # load model to cpu and half 
            if lora_path is not None and os.path.exists(lora_path+'/adapter_config.json') \
            and os.path.exists(lora_path+'/adapter_model.bin'):
                print('load lora from ',lora_path)
                model = PeftModel.from_pretrained(model, lora_path).half().cpu()
                self.model = model
            else:
                #we'll use peft lora library to do the lora
                lora_rank = lora_rank if lora_rank > 0 else 32
                #config lora with rank of lora_rank
                lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                        inference_mode=False,
                                        r=lora_rank,
                                        lora_alpha=32,
                                        lora_dropout=0.1)
                model = get_peft_model(model, lora_config)
                self.model = model
        else:
            raise ValueError("No pretrained model provided!")
        value_head = nn.Linear(model.config.hidden_size, 1)
        if lora_path is not None and os.path.exists(os.path.join(lora_path,'value_head.bin')):
            print('load value_head from ',os.path.exists(os.path.join(lora_path,'value_head.bin')))
            value_head.load_state_dict(torch.load(os.path.join(lora_path,'value_head.bin')))
            print('enable value_head grad')
        else:
            value_head.weight.data.normal_(mean=0.0, std=1 / (model.config.hidden_size + 1))
        value_head = value_head.half().cpu()
        self.value_head = value_head

    def train(self, mode: bool = True):
        self.model.train(mode=mode)
        self.value_head.train(mode=mode)
    
    def mark_only_lora_trainable(self, bias: str = "none"):
        #since loading lora through peft will not mark sub-modules trainable, if you need to train from last checkpoint, you need to call this function
        self.model.requires_grad_(True)
        self.value_head.requires_grad_(True)
        from peft.tuners.lora import mark_only_lora_as_trainable
        mark_only_lora_as_trainable(self.model, bias)
    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

    def forward(self, sequences: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        #ChatGLM attention_mask is a [seq,seq] torch.bool matrix, it's not compatiable with other models' attention_mask
        #Since the model can generate the attention_mask by itself with special ending token, we can just pass the attention_mask as None
        bs = sequences.shape[0]
        seq = sequences.shape[1]
        prompt_mask = (sequences != self.pad_id).to(torch.float)
        if attention_mask is not None and (attention_mask.dtype != torch.bool or len(attention_mask.shape) != 4 or attention_mask.shape != (bs,1,seq,seq)):
            attention_mask = None
        outputs = self.model(sequences, attention_mask=attention_mask,return_dict=True, output_hidden_states=True)
        last_hidden_states = outputs['hidden_states'][-1].transpose(0,1)#change from (seq,b,dim) to (b,seq,dim)
        values = self.value_head(last_hidden_states).squeeze(2)[:,:-1]  # remove last token and squeeze the last dimension
        prompt_mask = prompt_mask[:, :-1]
        #calculate the mean of the values with the indexes only when the sequence[batch_id][index] != pad_id
        value = masked_mean(values, prompt_mask, dim=1)
        return value
    
    def get_base_model(self):
        return self.model
    
    def save_pretrained(self,save_directory):
        self.model.save_pretrained(save_directory)
        torch.save(self.value_head.state_dict(),os.path.join(save_directory,'value_head.bin'))
