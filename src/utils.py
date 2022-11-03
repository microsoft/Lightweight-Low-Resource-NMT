import io
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
import argparse 
import matplotlib.pyplot as plt 
import ast
import pandas as pd
import csv
import seaborn as sns
sns.set(style="whitegrid")

def convert_src_tgt(src_lang, tgt_lang, src_file):
    """Script converts between unseen language and a language that is seen by the target tokenizer. 
    arguments: 
    src_lang - source language (follow language codes specified at https://github.com/anoopkunchukuttan/indic_nlp_library)
    tgt_lang - target language or the language to script convert to (follow language codes specified at https://github.com/anoopkunchukuttan/indic_nlp_library)
    src_file - source language samples
    """
    with open(src_file, 'r', encoding = 'UTF-8') as file, open(f'{src_file}_script_converted.txt', 'w', encoding = 'UTF-8') as out:
        samples = file.read().split('\n')
        for sample in samples: 
            out.write(UnicodeIndicTransliterator.transliterate(sample,src_lang,tgt_lang) + '\n')

def visualize_confidence_stats(stats_file):
    """Use to visualize the stats generated using confidence_estimation.py. Any of the 4-5 metrics logged in that script can be visualized 
    and edit index accordingly. 
    """
    with open(stats_file, 'r') as f: 
        confidence_stats = f.read().split('\n')
        avg_softmax_scores = ast.literal_eval(confidence_stats[4])
        y = [i for i in range(0, len(avg_softmax_scores))]
        plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')
        plt.scatter(y, avg_softmax_scores, color = 'blue', label = 'In Domain')
        plt.xticks(fontsize=12); plt.yticks(fontsize=12)
        plt.savefig('./confidence_stats_scatter')

        df = pd.DataFrame((avg_softmax_scores), columns = [''])
        f, ax = plt.subplots(figsize=(8, 8))
        sns.violinplot(data=df, inner="box", palette="Set3", cut=2, linewidth=3)
        sns.despine(left=True) 


def convert_tgt_src(src_lang, tgt_lang, src_file):
    def convert_src_tgt(src_lang, tgt_lang, src_file):
        with open(src_file, 'r', encoding = 'UTF-8') as file,  open(f'{src_file}_script_restored.txt', 'w', encoding = 'UTF-8') as out:
            samples = file.read().split('\n')
            for sample in samples: 
                out.write(UnicodeIndicTransliterator.transliterate(sample,src_lang,tgt_lang) + '\n')
  
def check_dedup(src_file, tgt_file, check_src_file, check_tgt_file):
    """ Function to check for duplicates between train and test files.
        arguments: 
        src_file - source side sentences of train/dev set (.txt). 
        tgt_file - target side sentences of train/dev set (.txt). 
        check_src_file - source side sentences of test set (.txt).
        check_tgt_file - target side sentences of test set (.txt).
    """
    # src_file = "/home/t-hdiddee/INMT-Lite/data/hi-mun/train_hi.txt_cleaned"
    # tgt_file = "/home/t-hdiddee/INMT-Lite/data/hi-mun/train_mun.txt_cleaned"

    # check_src_file = "/home/t-hdiddee/INMT-Lite/data/hi-mun/test_hi.txt"
    # check_tgt_file = "/home/t-hdiddee/INMT-Lite/data/hi-mun/test_mun.txt"

    src_samples = io.open(src_file, encoding='UTF-8').read().rstrip(' ').split('\n')
    tgt_samples = io.open(tgt_file, encoding='UTF-8').read().rstrip(' ').split('\n')
    check_src_samples = io.open(check_src_file, encoding='UTF-8').read().rstrip(' ').split('\n')
    check_tgt_samples = io.open(check_tgt_file, encoding='UTF-8').read().rstrip(' ').split('\n')

    print(len(src_samples))
    source, target = set(), set()
    for ele in zip(src_samples, tgt_samples):
        source.add((ele[0].lstrip(' '), ele[1].lstrip(' ')))
    for ele in zip(check_src_samples, check_tgt_samples):
        target.add((ele[0].lstrip(' '), ele[1].lstrip(' ')))

    print(len(source.intersection(target)))
    print(source.intersection(target))

    deduped = source.difference(target)
    print(f'Cleaned Train Set has {len(deduped)} instances. ')

    
def dedup(src_file, tgt_file, check_src_file, check_tgt_file)
    """ Function to clean any duplicates between train and test files using criteria: if either source or target of test set exist in 
        in the source and target train set respectively, remove them and rewrite the train file.
        arguments: 
        src_file - source side sentences of train/dev set (.txt). 
        tgt_file - target side sentences of train/dev set (.txt). 
        check_src_file - source side sentences of test set (.txt).
        check_tgt_file - target side sentences of test set (.txt).
    """
    with open(f'{src_file}_cleaned', 'w') as src, open(f'{tgt_file}_cleaned', 'w') as tgt: 
        for ele in deduped: 
            src.write(ele[0] + '\n')
        for ele in deduped: 
            tgt.write(ele[1] + '\n')
                    
    source_side_duplicates, target_side_duplicates = [], []
    for ele in src_samples: 
        if ele in check_src_samples: 
            source_side_duplicates.append(ele)
    print(len(source_side_duplicates))

    for ele in tgt_samples: 
        if ele in check_tgt_samples: 
            target_side_duplicates.append(ele)
    print(len(target_side_duplicates))


    deduped = set()
    duplicate_ctr = 0
    if len(target_side_duplicates) or len(source_side_duplicates): 
        print('Train-Test Sets have Leak. Rectifying...')
        for train_src, train_tgt in zip(src_samples, tgt_samples):
            for test_src, test_tgt in zip(check_src_samples, check_tgt_samples): 
                if train_tgt == test_tgt or train_src == test_src : 
                    duplicate_ctr += 1  
                else: 
                    deduped.add((train_src, train_tgt))
    print(f'{duplicate_ctr} number of duplucates existed.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lang", type=str)
    parser.add_argument("--tgt_lang", type=str)
    parser.add_argument("--src_file", type=str)
    parser.add_argument("--tgt_file", type=str)
    parser.add_argument("--check_src_file", type=str)
    parser.add_argument("--check_tgt_file", type=str)
    parser.add_argument("--stats_file", type=str)
    args = parser.parse_args()

    convert_src_tgt(args.src_lang, args.tgt_lang, args.src_file)
    convert_tgt_src(args.src_lang, args.tgt_lang, args.src_file)




