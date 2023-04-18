from transformers import AutoTokenizer
from glm_rm import ChatGLMRM
if __name__ == '__main__':
    chat_glm_model_path = '/Volumes/Samsung_X5/models/chatglm-6b'
    rm_model_path = '/Volumes/Samsung_X5/models/chats/chatglmrm'

    tokenizer = AutoTokenizer.from_pretrained(chat_glm_model_path,trust_remote_code=True)

    input_str = '请根据法律知识回答。提问：办理营业执照需要多久才能下来\t回答：\t法律分析：办理营业执照多久能下来：营业执照一般审核通过后当天即可领证。 法律规定，申请人提交的申请材料齐全、符合法定形式的，登记机关应当当场予以登记，并发给申请人准予登记通知书。 根据法定条件和程序，需要对申请材料的实质性内容进行核实的，登记机关应当指派两名以上工作人员进行核查，并填写申请材料核查情况报告书。登记机关应当自受理登记申请之日起15日内作出是否准予登记的决定。法律依据：《个体工商户登记管理办法》第二十条     申请人提交的申请材料齐全、符合法定形式的，登记机关应当当场予以登记，并发给申请人准予登记通知书。  根据法定条件和程序，需要对申请材料的实质性内容进行核实的，登记机关应当指派两名以上工作人员进行核查，并填写申请材料核查情况报告书。登记机关应当自受理登记申请之日起15日内作出是否准予登记的决定。'
    max_len = 256
    input_ids = tokenizer.encode_plus(input_str, return_tensors='pt', add_special_tokens=False, max_length=max_len,truncation=True,padding='max_length')
    print(input_ids)
    glm_rm = ChatGLMRM(pretrained=chat_glm_model_path,lora_path=rm_model_path)
    print(glm_rm)
    glm_rm.print_trainable_params()