export http_proxy=http://127.0.0.1:15777 https_proxy=http://127.0.0.1:15777

torchrun --nproc_per_node 8 \
run.py \
--output_dir ./output \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--train_data ../../data/finetune/toy_finetune_data.jsonl \
--learning_rate 3e-4 \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.01 \
--query_max_len 64 \
--passage_max_len 160 \
--train_group_size 16 \
--logging_steps 10 \
--save_steps 500 \
--save_total_limit 3 \
--ddp_find_unused_parameters False \
--negatives_cross_device \
--gradient_checkpointing \
--deepspeed ../../stage1.json \
--warmup_ratio 0.1 \
--fp16 \
--cache_dir /share/LMs \
--token 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