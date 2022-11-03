#!/bin/bash
# Sample inference scripts: All model path's specified here are either Hm (Best performing models) or Dm (Distilled Models) - refer to terminology at https://arxiv.org/abs/2210.15184

# Batch inferencing of the online models: MT5 

python inference.py \
--src_lang hi \
--tgt_lang mun \
--benchmark_path mt5_base_hi_mun.txt \
--model_arch google/mt5-base \
--model_path Hm/mt5_base_hi_mun/ \
--src_file test_hi.txt \
--task_prefix "Translate Hindi to Mundari:"


# Batch inferencing of the online models: TFB 

python inference.py \
--src_lang hi \
--tgt_lang mun \
--benchmark_path tfb_hi_mun.txt \
--model_path Hm/tfb_hi_mun \
--vocab_path vocab_mun.json
--src_file test_hi.txt \

# Generate inference for the quantized (Batch processing is WIP)
python inference.py \
--src_lang en \
--tgt_lang as \
--benchmark_path mt5_small_quantized.txt \
--model_arch google/mt5-small
--mode offline \
--src_file eng_Latn.devtest \
--task_prefix "Translate English to Assamesse:" \
--encoder_interpreter_path mt5_small_en_as_encoder.tflite \
--decoder_interpreter_path mt5_small_en_as_decoder.tflite \

# Generating inferences for the teacher model on an unlabelled dataset for distillation
python student_labels.py \
--src_lang esp \
--tgt_lang bzd \
--root ./distilled_labels \
--model_path Hm/mt5_small_esp_bzd/  \
--src_file distillation_scaling_es.txt \
--batch_size 500 \
--model_arch google/mt5-small \
--task_prefix "Translate Spanish to Bribi:" \  
--max_infer_samples 2000000
