import wandb
import random
import numpy as np
import datasets 
from datasets import load_dataset, load_metric
from dataclasses import dataclass, field
import os 
import transformers
from typing import Optional
from transformers import (
    EarlyStoppingCallback,
    HfArgumentParser, 
    PreTrainedTokenizerFast,  
    AutoConfig,
    MarianTokenizer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    MarianConfig,
    MarianMTModel,
    Seq2SeqTrainingArguments,
    set_seed
)

@dataclass 
class ModelArguments:
    lang: str = field(default=None)
    model_name: str = field(default = 'vanilla')
    cache_dir: str = field(default='./cached_models')
    source_prefix: str = field(default='')
    cpt_file: str = field(default=None)
    max_source_length: str = field(default=1024)
    max_target_length: str = field(default=1024)
    max_train_samples: int = field(default=None) #During Hyperparameter Tuning - This value is usually set between 75K-100K 
    earlier_checkpoint: str = field(default=None)
    marian_tokenizer: bool = field(default=False)
    source_spm: str = field(default=None)
    target_spm: str = field(default=None)
    mask_percentage: str = field(default=0.15)


def main():
    parser = HfArgumentParser((ModelArguments, Seq2SeqTrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_name,cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, config=config, cache_dir=args.cache_dir)
    
    prefix = args.source_prefix if args.source_prefix is not None else ""
    
    data_files = {}
    if args.cpt_file is not None: 
        data_files["train"] = args.cpt_file
    raw_datasets = load_dataset('json', data_files=data_files, cache_dir=args.cache_dir)

    lang = args.lang
    tokenizer.lang = args.lang

    train_dataset = raw_datasets["train"]


    def convert_to_sentinel(samples):
            sentinel_samples = []
            for sample in samples: 
                current_sentinel_id = 0
                padding_list = []
                sentinel_sample = []        
                for idx in range(len(sample)): 
                    if sample[idx] == -1:
                        current_sentinel_token = f"<extra_id_{current_sentinel_id}>"
                        sample[idx] = tokenizer(current_sentinel_token)['input_ids'][0]
                        current_sentinel_id += 1
                        if sample[idx + 1] == -1: # Case of a Consecutive span   # When there is a consecutive span - the tensor will have to be padded to uniformize the lost length due to the consecutive spans being masked as one. 
                            padding_list.append(tokenizer.pad_token_id)
                            continue
                        else: 
                            sentinel_sample.append(sample[idx])
                    else: 
                        sentinel_sample.append(sample[idx])
                        
                sentinel_sample.extend(padding_list)
                sentinel_samples.append(sentinel_sample)
        
            return sentinel_samples

        
    def preprocess_function(examples):         
        
        inputs = [ex[lang] for ex in examples["pretraining"]]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=True, truncation=True) #return ['input_ids'] and ['attention_mask'] for the passed sequence
        labels = tokenizer(inputs, max_length=args.max_target_length, padding=True, truncation=True)
        unmasked_inputs = model_inputs['input_ids'] # Assuming that you want to perform continued pretraining with the samples for the target language (which is the low-resource language)
        model_inputs['labels'] = labels['input_ids']

        """ mT5's training objective involves corrupting randomly chosen spans of tokenized text - replaced by a set of unique sentinel tokens. """
        
        masked_inputs, masked_labels = [], []
        for unmasked_input in unmasked_inputs:
            masking_indices = random.sample(range(0, len(unmasked_input)), int(len(unmasked_input)*args.mask_percentage)) #Corrupting only 15% of the the tokens
            masking_indices.sort()
            masked_input, masked_label = unmasked_input, [1]*len(unmasked_input)

            for idx in range(len(masked_input) - 1): 
                if idx in masking_indices:
                    masked_label[idx] = masked_input[idx]
                    masked_input[idx] = -1 
                else: 
                    masked_label[idx] = -1 

            masked_inputs.append(masked_input)
            masked_labels.append(masked_label)
        sentinel_inputs = convert_to_sentinel(masked_inputs)
        sentinel_labels = convert_to_sentinel(masked_labels)

        model_inputs['input_ids'] = sentinel_inputs
        model_inputs['labels'] = sentinel_labels

        return model_inputs


    if training_args.do_train:
        train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on train dataset",
        )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id = tokenizer.pad_token_id)
    metric = load_metric("sacrebleu")
    
    set_seed(training_args.seed)
    trainer = Seq2SeqTrainer(
        model=model, 
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if training_args.do_train: 
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload
        
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

if __name__ == "__main__": 
    main()