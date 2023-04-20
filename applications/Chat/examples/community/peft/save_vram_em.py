from typing import Dict, Optional, Union
import torch
import torch.nn as nn
from coati.models.utils import normalize,compute_reward
from coati.experience_maker.base import Experience
from coati.models.glm import ChatGLMActor,ChatGLMCritic,ChatGLMRM
from transformers import AutoTokenizer
import torch.nn.functional as F


class SaveVramExperienceMaker:
    
    def __init__(self,
                 actor: ChatGLMActor, 
                 critic: ChatGLMCritic, 
                 reward_model: ChatGLMRM, 
                 initial_model: ChatGLMActor, 
                 kl_coef: float = 0.1) -> None:
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.initial_model = initial_model
        self.kl_coef = kl_coef

    """
    Naive experience maker.
    """

    @torch.no_grad()
    def make_experience(self, inputs:  Union[torch.Tensor, Dict[str, torch.Tensor]], **generate_kwargs) -> Experience:
        #first we move the initial_model,reward_model and critic to cpu,only actor in gpu
        self.initial_model.to("cpu")
        self.reward_model.to("cpu")
        self.critic.to("cpu")

        self.actor.half().to(torch.cuda.current_device())
        self.actor.eval()
        print(f"cuda memeor usage actor generate: {torch.cuda.memory_allocated(0)/1024/1024} MB")

        if isinstance(inputs, torch.Tensor):
            input_ids = inputs
            attention_mask = None
        elif isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
        sequences, attention_mask, action_mask = self.actor.generate(input_ids.to(torch.cuda.current_device()),
                                                                     attention_mask=attention_mask,
                                                                     return_action_mask=True,
                                                                     **generate_kwargs)
        # num_actions = action_mask.size(1)
        print(f"cuda memeor usage actor log probs: {torch.cuda.memory_allocated(0)/1024/1024} MB")

        action_log_probs = self.actor(sequences, action_mask, attention_mask)
        self.actor.to("cpu")
        self.initial_model.half().to(torch.cuda.current_device())
        self.initial_model.eval()
        print(f"cuda memeor usage initial_model log_probs: {torch.cuda.memory_allocated(0)/1024/1024} MB")
        base_action_log_probs = self.initial_model(sequences, action_mask, attention_mask)
        self.initial_model.to("cpu")

        self.critic.to(torch.cuda.current_device())
        self.critic.eval()
        print(f"cuda memeor usage critic : {torch.cuda.memory_allocated(0)/1024/1024} MB")
        value = self.critic(sequences, attention_mask,action_mask)
        self.critic.to("cpu")

        self.reward_model.to(torch.cuda.current_device())
        self.reward_model.eval()
        print(f"cuda memeor reward critic : {torch.cuda.memory_allocated(0)/1024/1024} MB")
        r = self.reward_model(sequences, attention_mask)
        self.reward_model.to("cpu")
        print(f"cuda memeor reward after reward : {torch.cuda.memory_allocated(0)/1024/1024} MB")
        reward = compute_reward(r, self.kl_coef, action_log_probs, base_action_log_probs, action_mask=action_mask[:, :-1])

        advantage = reward - value
        # TODO(ver217): maybe normalize adv
        if advantage.ndim == 1:
            advantage = advantage.unsqueeze(-1)

        return Experience(sequences, action_log_probs, value, reward, advantage, attention_mask, action_mask)


if __name__ == '__main__':
    #tmp test !
    model_path = "/home/yueyulin/pretrained_models/chatglm-6b"
    lora_path = "/home/yueyulin/models/sft_law_chatglm6b_ask_law_prompts"
    from transformers import AutoTokenizer
    actor = ChatGLMActor(model_path, lora_path=lora_path).half().cpu()
    print(f'cuda vram usage after loading actor: {torch.cuda.memory_allocated(0)/1024/1024} MB')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(tokenizer)
    initial_model = ChatGLMActor(model_path, lora_path=lora_path).half().cpu()
    print(f'cuda vram usage after loading initial_model: {torch.cuda.memory_allocated(0)/1024/1024} MB')

    reward_model_lora = "/home/yueyulin/models/chatglmrm"
    reward_model = ChatGLMRM(model_path,reward_model_lora).half().cpu()
    print(f'cuda vram usage after loading reward_model: {torch.cuda.memory_allocated(0)/1024/1024} MB')
    critic = ChatGLMCritic(model_path,reward_model_lora).half().cpu()
    print(f'cuda vram usage after loading critic: {torch.cuda.memory_allocated(0)/1024/1024} MB')

    input_prompts = ["提问：正在按揭还贷款的房子在不花钱的情况下如何改到未成年孩子的名下 回答：",
                     "提问：长期不运动会怎么样？回答："]
    input_ids = tokenizer.batch_encode_plus(input_prompts, return_tensors="pt", padding=True, truncation=True)
    print(input_ids)
    attention_mask = input_ids['attention_mask']
    input_ids = input_ids["input_ids"]
    inputs = dict(input_ids=input_ids, attention_mask=attention_mask)

    generate_kwargs = {"max_length": 32,
                       "do_sample": True,
                       "temperature": 1.0,
                       "top_k":50,
                       "pad_token_id":tokenizer.pad_token_id,
                       "eos_token_id":tokenizer.eos_token_id}
    
    exp_maker = SaveVramExperienceMaker(actor,critic,reward_model, initial_model,  kl_coef=0.1)
    exp = exp_maker.make_experience(inputs, **generate_kwargs)
    print(exp)
        
        
        
        
        
        
