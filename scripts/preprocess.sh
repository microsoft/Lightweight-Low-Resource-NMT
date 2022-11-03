#!/bin/bash
# All input files are of the form *.txt

# If continued_pretraining - then preprocess data with: 
python preprocess.py \
--root ./pretraining_data \
--continued_pretraining True \
--src_lang mun \
--tgt_lang mun \
--train_src mun.output \
--train_tgt mun.output \
--split True

# If preprocessing the data for training the distilled model: 
python preprocess.py \
--root ./distillation_data \
--src_lang esp \
--tgt_lang bzd \
--train_src Distillation_esp_bzd_cleaned.inputs \ 
--train_tgt Distillation_esp_bzd_cleaned.labels \
--test_src test.es \
--test_tgt test.bzd \
--distill True 
