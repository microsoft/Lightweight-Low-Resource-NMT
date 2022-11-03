#!/bin/bash
python confidence_estimation.py \
--src_lang hi \
--tgt_lang mun \
--benchmark_path ./confidence/hi_mun_ood_scores.txt \
--model_arch mt5 \
--model_path Hm/mt5_small_hi_mun/ \
--task_prefix "Translate Hindi to Mundari:" \
--callibration_file monolingual_data/final_hi.txt
