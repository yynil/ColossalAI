# DS='/home/yueyulin/ask_law_data_dir/laws/train/'
# SAVE='/home/yueyulin/models/sft_law_chatglm6b_laws'
# LOAD='ss'
DS='/home/yueyulin/ask_law_data_dir/ask_law_sft_train/train/'
SAVE='/home/yueyulin/models/sft_law_chatglm6b_ask_law_prompts'
LOAD='/home/yueyulin/models/sft_law_chatglm6b_laws'
torchrun --standalone --nproc_per_node=1 \
        train_peft_sft.py \
        --dataset $DS\
        --model chatglm \
        --model_path $LOAD \
        --pretrain /home/yueyulin/pretrained_models/chatglm-6b \
        --strategy colossalai_zero2 \
        --batch_size 2 \
        --max_epochs 2 \
        --save_path $SAVE 