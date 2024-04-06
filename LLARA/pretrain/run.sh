export http_proxy=http://127.0.0.1:15777 https_proxy=http://127.0.0.1:15777

torchrun --nproc_per_node 8 \
run.py \
--output_dir ./output \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--train_data ../../data/pretrain/toy_pretrain_data.jsonl \
--learning_rate 1e-5 \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--dataloader_drop_last True \
--cutoff_len 128 \
--logging_steps 1 \
--save_steps 500 \
--save_total_limit 20 \
--gradient_checkpointing \
--ddp_find_unused_parameters False \
--use_flash_attn True \
--deepspeed ../../stage1.json \
--warmup_ratio 0.1 \
--remove_stop_words True \
--use_lora False \
--bf16 \
--cache_dir /share/LMs \
--token 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