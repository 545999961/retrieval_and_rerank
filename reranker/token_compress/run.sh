export https_proxy=http://127.0.0.1:15777 http_proxy=http://127.0.0.1:15777

torchrun --nproc_per_node 8 \
run.py \
--output_dir ./output \
--model_name_or_path google/gemma-2b \
--train_data ../../data/finetune/toy_finetune_data.jsonl \
--learning_rate 2e-4 \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16 \
--dataloader_drop_last True \
--query_max_len 32 \
--passage_max_len 192 \
--train_group_size 16 \
--logging_steps 1 \
--save_steps 2000 \
--save_total_limit 50 \
--ddp_find_unused_parameters False \
--gradient_checkpointing \
--deepspeed ../../stage1.json \
--warmup_ratio 0.1 \
--bf16 \
--use_lora True \
--lora_rank 32 \
--lora_alpha 64 \
--use_flash_attn False \
--target_modules q_proj k_proj v_proj o_proj \
--token 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 \
--cache_dir /share/LMs \
--cache_path /share/cf/pycharm/reranker_finetune_lightweight/data_cache \
--compress_method weighted_mean_direct

# sample last weighted_drop weighted_mean_direct mean