#!/bin/bash
# All training, eval and test scripts should be generated using either of the commands specified in preprocess.sh

# CPT 
python continued_pretraining.py \
--model_name google/mt5-small \
--lang mun \
--cpt_file ./train_mun_mun.json \
--per_device_train_batch 32 \
--logging_strategy steps \
--logging_steps 50 \
--learning_rate 1e-3 \
--num_train_epochs 10 \
--gradient_accumulation 2 \
--save_strategy steps \
--learning_rate 1e-3 \
--num_train_epochs 20 \
--output_dir pretrained_model \
--do_train 

# MT5 Training with CPT 

python train.py \
--source_prefix "Translate Hindi to Mundari:" \
--model_name google/mt5-small \
--do_eval \
--src_lang hi \
--tgt_lang mun \
--train_file train_hi_mun.json \
--validation_file val_hi_mun.json \
--output_dir pretraining_data/finetuned_model \
--per_device_train_batch 32 \
--per_device_eval_batch 32 \
--evaluation_strategy steps \
--eval_steps 50 \
--logging_strategy steps \
--logging_steps 50 \
--learning_rate 1e-3 \
--num_train_epochs 20 \
--gradient_accumulation 2 \
--load_best_model_at_end True \
--save_strategy steps \
--save_steps 800 \
--save_total_limit 2 \
--overwrite_output_dir \
--do_predict True \
--eval_accumulation_steps 2 \
--test_file test_hi_mun.json \
--predict_with_generate True \
--do_train \
--earlier_checkpoint ./pretraining_data/pretrained_model/

# MT5 Training without CPTed model: Finetuning the MT5

python train.py \ 
--source_prefix "Translate Hindi to Mundari:" \
--model_name google/mt5-small \
--do_eval \
--src_lang hi \
--tgt_lang mun \
--train_file train_hi_mun.json \
--validation_file val_hi_mun.json \
--output_dir finetuned_model \
--per_device_train_batch 16 \
--per_device_eval_batch 16 \
--evaluation_strategy steps \
--eval_steps 50 \
--logging_strategy steps \
--logging_steps 50 \
--learning_rate 1e-3 \
--num_train_epochs 20 \
--gradient_accumulation 2 \
--load_best_model_at_end True \
--save_strategy steps \
--save_steps 800 \
--save_total_limit 2 \
--overwrite_output_dir \
--do_predict True \
-eval_accumulation_steps 2 \
--test_file test_hi_mun.json \
--predict_with_generate True \
--do_train


# Training the Distilled Variant of the Model 
python train.py \
--vocab_path vocab_bzd.json \
--do_train \
--do_eval \
--src_lang esp \
--tgt_lang bzd \
--train_file train_distillation_esp_bzd.json \
--validation_file val_distillation_esp_bzd.json \
--output_dir distilled_esp_bzd/ \
--per_device_train_batch 256 \
--per_device_eval_batch 256 \
--evaluation_strategy steps \
--eval_steps 400 \
--logging_strategy steps \
--logging_steps 400 \
--learning_rate 5e-5 \
--num_train_epochs 30 \
--gradient_accumulation 2  \
--load_best_model_at_end True \
--metric_for_best_model bleu \
--save_strategy steps \
--save_steps 800 \
--save_total_limit 4 \
--overwrite_output_dir \
--warmup_steps 500 \
--do_predict True \
--eval_accumulation_steps 2 \
--test_file _distillation_esp_bzd.json --predict_with_generate True --max_train_samples 1000000

# Training for different student architectures and data samples (--distill_config and --max_train_samples have to be varied)
python train.py \
--vocab_path vocab_gondi.json \
--do_train \
--do_eval \
--src_lang hi \
--tgt_lang gondi \
--train_file train_distillation_hi_gondi.json \
--validation_file val_distillation_hi_gondi.json \
--output_dir distilled_hi_gondi_8_6/ \
--per_device_train_batch 256 \
--per_device_eval_batch 256 \
--evaluation_strategy steps \
--eval_steps 200 \
--logging_strategy steps \
--logging_steps 200 \
--learning_rate 5e-4 \
--num_train_epochs 20 \
--load_best_model_at_end True \
--save_strategy steps \
--save_steps 400 \
--save_total_limit 2 \
--overwrite_output_dir \
--warmup_steps 500 \
--do_predict True \
--eval_accumulation_steps 4 \
--test_file val_distillation_hi_gondi.json \
--predict_with_generate True \
--max_train_samples 500000 \
--distil_config 8_6
