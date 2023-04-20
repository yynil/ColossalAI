PROMPT_PATH='/home/yueyulin/ask_law_data_dir/ask_law_sft_prompt/train'
PRETRAIN_PATH='/home/yueyulin/ask_law_data_dir/ask_law_sft_test_1000/train'
PRETRAINED_MODEL='/home/yueyulin/pretrained_models/chatglm-6b'
SFT_LORA='/home/yueyulin/models/sft_law_chatglm6b_ask_law_prompts'
RM_LORA='/home/yueyulin/models/chatglmrm/'
SAVE_PATH='lora_ppo'
#change to THUDM/chatglm-6b if you don't have the pretrained model
torchrun --standalone --nproc_per_nod=1 \
        train_peft_prompts.py \
        --prompt_path $PROMPT_PATH \
        --pretrain_dataset $PRETRAIN_PATH \
        --model chatglm \
        --pretrain $PRETRAINED_MODEL \
        --sft_lora_path $SFT_LORA \
        --rm_lora_path $RM_LORA \
        --save_path $SAVE_PATH \
        --strategy colossalai_zero2 \
        --num_episodes 1 \
        --max_timesteps 5000 \
        --update_timesteps 200 \
        --train_batch_size 1 \
        --experience_batch_size 1