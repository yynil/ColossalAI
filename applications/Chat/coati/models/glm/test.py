from transformers import AutoTokenizer
from coati.models.glm import ChatGLMRM,ChatGLMLM,ChatGLMActor,ChatGLMCritic
import torch
def test_rm(chat_glm_model_path, rm_model_path, input_ids,test_device :str = 'cuda'):
    glm_rm = ChatGLMRM(pretrained=chat_glm_model_path,lora_path=rm_model_path)
    glm_rm.mark_only_lora_trainable()
    glm_rm.print_trainable_parameters()
    glm_rm.eval()
    glm_rm.print_trainable_parameters()
    glm_rm.to(test_device)
    rm = glm_rm(input_ids['input_ids'].to(test_device),input_ids['attention_mask'].to(test_device))
    print(rm)
    rm = glm_rm(input_ids['input_ids'].to(test_device),torch.ones_like(input_ids['input_ids']).to(test_device))
    print(rm)
    rm = glm_rm(input_ids['input_ids'].to(test_device),None)
    print(rm)
    del glm_rm
def test_critic(chat_glm_model_path, rm_model_path, input_ids,test_device :str = 'cuda'):
    glm_critic = ChatGLMCritic(pretrained=chat_glm_model_path,lora_path=rm_model_path,use_action_mask=True)
    glm_critic.mark_only_lora_trainable()
    glm_critic.print_trainable_parameters()
    glm_critic.eval()
    seq_len = input_ids['input_ids'].shape[1]
    fake_action_num = 10
    action_mask = torch.cat([torch.zeros((input_ids['input_ids'].shape[0],seq_len-fake_action_num),dtype=torch.long),torch.ones((input_ids['input_ids'].shape[0],fake_action_num),dtype=torch.long)],dim=1)
    glm_critic.to(test_device)
    critic = glm_critic(input_ids['input_ids'].to(test_device),input_ids['attention_mask'].to(test_device),action_mask.to(test_device))
    print(critic)
    glm_critic.use_action_mask = False
    critic = glm_critic(input_ids['input_ids'].to(test_device),input_ids['attention_mask'].to(test_device),action_mask.to(test_device))
    print(critic)
    del critic
def test_lm(chat_glm_model_path, rm_model_path, input_ids,tokeninzer, **kwargs):
    glm_lm = ChatGLMLM(pretrained=chat_glm_model_path,lora_path=rm_model_path)
    glm_lm.to('cuda')
    glm_lm.mark_only_lora_trainable()
    glm_lm.print_trainable_parameters()
    glm_lm.eval()
    attention_mask = input_ids['attention_mask']
    input_ids = input_ids['input_ids']
    bos_token_id = tokeninzer.bos_token_id
    print(bos_token_id)
    sequences,attention_mask,action_mask = glm_lm.generate(input_ids.to('cuda'),attention_mask = attention_mask.to('cuda'),**kwargs)
    print(action_mask)
    action_numbers = action_mask.sum(dim=1)
    print(action_numbers)
    output = glm_lm(sequences.to('cuda'),attention_mask = attention_mask.to('cuda'),labels=sequences.to('cuda'))
    print(output.loss)
    
    logprobs = ChatGLMActor.forward(glm_lm,sequences.to('cuda'),action_mask=action_mask.to('cuda'),attention_mask = attention_mask.to('cuda'))
    print(logprobs)
    print(logprobs.shape)
    del glm_lm
if __name__ == '__main__':
    chat_glm_model_path = '/home/yueyulin/pretrained_models/chatglm-6b/'
    rm_model_path = '/home/yueyulin/models/chatglmrm/'

    tokenizer = AutoTokenizer.from_pretrained(chat_glm_model_path,trust_remote_code=True)
    print(tokenizer.eos_token)
    print(tokenizer.eos_token_id)

    input_str = '请根据法律知识回答。提问：办理营业执照需要多久才能下来\t回答：\t法律分析：办理营业执照多久能下来：营业执照一般审核通过后当天即可领证。 法律规定，申请人提交的申请材料齐全、符合法定形式的，登记机关应当当场予以登记，并发给申请人准予登记通知书。 根据法定条件和程序，需要对申请材料的实质性内容进行核实的，登记机关应当指派两名以上工作人员进行核查，并填写申请材料核查情况报告书。登记机关应当自受理登记申请之日起15日内作出是否准予登记的决定。法律依据：《个体工商户登记管理办法》第二十条     申请人提交的申请材料齐全、符合法定形式的，登记机关应当当场予以登记，并发给申请人准予登记通知书。  根据法定条件和程序，需要对申请材料的实质性内容进行核实的，登记机关应当指派两名以上工作人员进行核查，并填写申请材料核查情况报告书。登记机关应当自受理登记申请之日起15日内作出是否准予登记的决定。'
    max_len = 256
    input_ids = tokenizer(input_str,
                                     max_length=max_len,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")
    print(input_ids)
    print('test reward model')
    test_rm(chat_glm_model_path, rm_model_path, input_ids)
    print('test critic')
    test_critic(chat_glm_model_path, rm_model_path, input_ids)

    input_str = '请根据法律知识回答。提问：办理营业执照需要多久才能下来\t回答：\t'
    max_len = 96
    input_ids = tokenizer(input_str,
                                     max_length=max_len,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")
    
    print('input_ids',input_ids['input_ids'])
    kwargs = {
        "max_length":256,
        "do_sample":True,
        "temperature":1.0,
        "top_k":50,
        "pad_token_id":tokenizer.pad_token_id,
        "eos_token_id":tokenizer.eos_token_id,
    }
    lora_path = '/home/yueyulin/models/sft_law_chatglm6b_ask_law_prompts'
    test_lm(chat_glm_model_path, lora_path, input_ids,tokenizer, **kwargs)

