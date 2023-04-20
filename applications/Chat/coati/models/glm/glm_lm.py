import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module 
from typing import Optional, Tuple, Union
from transformers import AutoModel
import os
from peft import PeftModel,LoraConfig,TaskType,get_peft_model
from coati.models.generation import generate
from coati.models.utils import log_probs_from_logits
import numpy as np

class ChatGLMActor(Module):

    def __init__(self, pretrained: str = None,
                 lora_path :str = None,
                 lora_rank :int = 0) -> None:
        super().__init__()
        if pretrained is not None:
            model = AutoModel.from_pretrained(
                pretrained,
                trust_remote_code=True,
            ).half().cpu() # load model to cpu and half 
        else:
            raise ValueError("No pretrained model provided!")
        if lora_path is not None and os.path.exists(lora_path+'/adapter_config.json') \
            and os.path.exists(lora_path+'/adapter_model.bin'):
            print('load lora from ',lora_path)
            model = PeftModel.from_pretrained(model, lora_path).half().cpu()
        else:
            print('init lora from scratch')
            lora_rank = lora_rank if lora_rank > 0 else 32
            #config lora with rank of lora_rank
            lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                        inference_mode=False,
                                        r=lora_rank,
                                        lora_alpha=32,
                                        lora_dropout=0.1)
            model = get_peft_model(model, lora_config)
        self.model = model

    def mark_only_lora_trainable(self, bias: str = "none"):
        #since loading lora through peft will not mark sub-modules trainable, if you need to train from last checkpoint, you need to call this function
        self.model.requires_grad_(True)
        from peft.tuners.lora import mark_only_lora_as_trainable
        mark_only_lora_as_trainable(self.model, bias)
    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

    @staticmethod
    def generate_glm_attention_mask(sequences : torch.Tensor, pad_token_id : int = 3, eos_token_id : int = 130005,bos_token_id :int = 130004) -> torch.Tensor:
        bs = sequences.size(0)
        seq_length = sequences.size(1)
        attention_masks = []
        for i in range(bs):
            inputs_ids = sequences[i]
            if bos_token_id in inputs_ids:
                context_length = inputs_ids.tolist().index(bos_token_id)
            else:
                context_length = seq_length
            attention_mask = np.ones((1, seq_length, seq_length))
            attention_mask = np.tril(attention_mask)
            attention_mask[:, :, :context_length] = 1
            attention_mask = np.bool_(attention_mask < 0.5)
            attention_masks.append(attention_mask)
        return torch.tensor(attention_masks,dtype=torch.bool,device=sequences.device)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_action_mask: bool = True,
        **kwargs
    ) -> Union[Tuple[torch.LongTensor, torch.LongTensor], Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]]:
        sequences = generate(self.model, input_ids,attention_mask=attention_mask, **kwargs)
        attention_mask = None
        pad_token_id = kwargs.get('pad_token_id', None)
        eos_token_id = kwargs.get('eos_token_id', None)
        attention_mask = self.generate_glm_attention_mask(sequences, pad_token_id, eos_token_id)
        if not return_action_mask:
            return sequences, attention_mask, None
        input_len = input_ids.size(1)

        action_mask = torch.where((sequences==pad_token_id) | (sequences==eos_token_id) , 0, 1).to(dtype=torch.long, device=sequences.device)
        action_mask[:,:input_len]=0
        
        #set action_mask to (bs,seq_len), the prompts, pads and eos are masked to zero only generated actions are masked to 1
        return sequences, attention_mask, action_mask

    def forward(self,
                sequences: torch.LongTensor = None,
                action_mask: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns action log probs
        """
        output = self.model(sequences, attention_mask=attention_mask)
        logits = output['logits']
        log_probs = log_probs_from_logits(logits[:, :-1, :], sequences[:, 1:])
        return log_probs
        # context_length = action_mask[0].tolist().index(1)
        # return log_probs[:, context_length:]#shape is (B,num_actions)

    def get_base_model(self):
        return self.model
    
    def save_pretrained(self,save_directory):
        self.model.save_pretrained(save_directory)


class LM(ChatGLMActor):

    def forward(self, sequences: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns output log probs
        """
        output = self.model(sequences, attention_mask=attention_mask)
        logits = output['logits']
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
    

class ChatGLMLM(LM):
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels, **kwargs)