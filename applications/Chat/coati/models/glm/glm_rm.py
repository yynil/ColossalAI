import torch
import torch.nn as nn
from torch.nn import Module 
from transformers import AutoModel
from peft import PeftModel,LoraConfig,TaskType,get_peft_model
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
                 lora_rank :int = 0) -> None:
        super().__init__()
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
        else:
            value_head.weight.data.normal_(mean=0.0, std=1 / (model.config.hidden_size + 1))
        self.value_head = value_head

    def train(self, mode: bool = True):
        self.model.train(mode=mode)
        self.value_head.train(mode=mode)
    

    def print_trainable_params(self):
        self.model.print_trainable_parameters()

    def forward(self, sequences: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        #ChatGLM attention_mask is a [seq,seq] torch.bool matrix, it's not compatiable with other models' attention_mask
        #Since the model can generate the attention_mask by itself with special ending token, we can just pass the attention_mask as None
        outputs = self.model(sequences, attention_mask=None,return_dict=True, output_hidden_states=True)
        last_hidden_states = outputs['hidden_states'][-1]
        # print(last_hidden_states.shape)
        values = self.value_head(last_hidden_states)[:-1, :]
        # print(values.shape)
        values = values.transpose(0,1)#change from (seq,B) to (B,seq)
        value = values.mean(dim=1).squeeze(1)    # ensure shape is (B)
        # print(value.shape)
        return value
    
    def get_base_model(self):
        return self.model
    
    def save_pretrained(self,save_directory):
        self.model.save_pretrained(save_directory)
        torch.save(self.value_head.state_dict(),os.path.join(save_directory,'value_head.bin'))
