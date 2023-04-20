from .glm_rm import ChatGLMRM
import torch
from typing import Optional
from coati.models.utils import masked_mean
class ChatGLMCritic(ChatGLMRM):
    def __init__(self, pretrained: str = None, lora_path: str = None, lora_rank: int = 0,use_action_mask: bool = True,pad_id :int = 3) -> None:
        super().__init__(pretrained, lora_path, lora_rank)
        self.use_action_mask = use_action_mask
        self.pad_id = pad_id

    def forward(self,
                sequences: torch.LongTensor,
                attention_mask: Optional[torch.Tensor] = None,
                action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bs = sequences.shape[0]
        seq = sequences.shape[1]
        if attention_mask is not None and (attention_mask.dtype != torch.bool or len(attention_mask.shape) != 4 or attention_mask.shape != (bs,1,seq,seq)):
            print('attention_mask is not compatiable with ChatGLM, set it to None',attention_mask.shape,attention_mask.dtype)
            attention_mask = None
        outputs = self.model(sequences, attention_mask=attention_mask,return_dict=True, output_hidden_states=True)
        last_hidden_states = outputs['hidden_states'][-1].transpose(0,1)#change from (seq,b,dim) to (b,seq,dim)
        values = self.value_head(last_hidden_states).squeeze(2)[:,:-1]  # remove last token and squeeze the last dimension

        #init prompt_mask with 0 if sequence[batch_id][index] == pad_id else 1
        prompt_mask = (sequences != self.pad_id).to(torch.float)
        if action_mask is not None and self.use_action_mask:
            prompt_mask = prompt_mask * (1 - action_mask)
            prompt_mask = prompt_mask[:, :-1]
            value = masked_mean(values, prompt_mask, dim=1)
            return value

        prompt_mask = prompt_mask[:, :-1]
        #calculate the mean of the values with the indexes only when the sequence[batch_id][index] != pad_id
        value = masked_mean(values, prompt_mask, dim=1)
        return value