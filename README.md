## Low-Resource MT Compression

#### Official Code for "Too Brittle To Touch: Comparing the Stability of Quantization and Distillation Towards Developing Lightweight Low-Resource MT Models" (To Appear in WMT 2022)

Leveraging shared learning through Massively Multilingual Models, state-of-the-art machine translation models are often able to adapt to the paucity of data for low-resource languages. However, this performance comes at the cost of significantly bloated models which are not practically deployable. In this work, we evaluate knowledge distillation's use to compress MT models focusing on languages with extremely limited training data. Through our analysis across 8 languages, we find that the variance in the performance of the distilled models due to their dependence on multiple priors makes distillation a brittle compression mechanism. We further explore the use of post-training quantization for the compression of these models. Here, we find that quantization provides more consistent performance trends (than distillation) for the entire range of languages, especially the lowest-resource languages in our target set.

#### Languages Covered and Data Sources 

We cover 8 languages of diverse linguistic origins, varying data between 7K samples to 3M samples for our study. The train-test splits for Gondi and Mundari will be released soon and testsets for all other languages are publicly available (listed in the paper). 

| **Language** | **Train Data (Sentence Pairs)** | **Links**                                                |
|--------------|---------------------------------|-----------                                               |
| Bribri       | ~7000                           | [Here](https://github.com/AmericasNLP/americasnlp2021)   |
| Wixarica     | ~8000                           | [Here](https://github.com/AmericasNLP/americasnlp2021)   |
| Mundari      | ~11000                          | Public Link Available Soon                               |
| Gondi        | ~25000                          | [Here](http://cgnetswara.org/hindi-gondi-corpus.html)    |   
| Assammesse   | ~135000                         | [Here](https://ai4bharat.iitm.ac.in/samanantar)          |
| Odia         | ~1M                             | [Here](https://ai4bharat.iitm.ac.in/samanantar)          |
| Punjabi      | ~2.4M                           | [Here](https://ai4bharat.iitm.ac.in/samanantar)          |
| Gujarati     | ~3M                             | [Here](https://ai4bharat.iitm.ac.in/samanantar)          |

#### Model Benchmarks - Compressed Variants 
Each of the quantized variants is at least 3x smaller than it's best performing model and the distilled variants are at least 6x smaller.  Models and their compressed variants (for plug-and-play usage) coming soon! 

| **Language** | **Best Uncompressed Variant** | **Best Distilled Variant** |           | **Best Quantized Variant** |           |
|--------------|-------------------------------|----------------------------| --------- |----------------------------|-----------|
|              |           **spBLEU**          |         **spBLEU**         | **chrF2** |         **spBLEU**         | **chrF2** |
| _Bribri_     |              6.4              |             6.8            |    13.2   |             7.4            |    19.4   |
| _Wixarica_   |              6.2              |             4.1            |    17.3   |             7.2            |    26.8   |
| _Mundari_    |              15.9             |            18.2            |    32.7   |            15.7            |    29.3   |
| _Gondi_      |              14.3             |            14.2            |    32.8   |            13.8            |    31.1   |
| _Assamesse_  |              10.7             |             9.6            |    27.4   |             6.2            |    25.7   |
| _Odia_       |              27.4             |            20.2            |    40.7   |            21.0            |    41.3   |
| _Punjabi_    |              38.4             |            32.8            |    46.6   |            27.0            |    48.0   |
| _Gujarati_   |              35.9             |            29.8            |    48.6   |            28.4            |    51.4   |
----------------------------------------------------------------------------------------------------------------------------------

#### Environment Information 
The environment can be setup using the provided requirements file (Requires pip > pip 22.0.2)
```
pip install -r requirements.txt 
```

#### Directory Structure
```
├── readme.md
├── requirements.txt
├── scripts                            # Scripts with all the variants of the commands + default hyperparameter values
│   ├── confidence_estimation.sh       # logging the confidence statistics
│   ├── inference.sh                   # inference for both architectures - online and offline graphs 
│   ├── preprocess.sh                  # preprocessing data for training and evaluation
│   ├── sweep.yaml                     # sweep yaml for hyperparameter trials 
│   └── train.sh                       # variants of training and continued pretraining
└── src                                # src files for all the experiments 
    ├── confidence_estimation.py       # logging confidence stats: average softmax entropy, standard deviation of log probabilities
    ├── continued_pretraining.py       # continued pretraining of mt5
    ├── inference.py                   # online and graph inference
    ├── preprocess.py                  # preprocessing bilingual and monolingual data + vocab and tokenizer creation 
    ├── split_saving.py                # generating the offline graphs for both model architectures 
    ├── student_labels.py              # generating the student labels for the best model architecture for the models   
    ├── train.py                       # training script for vanilla, distilled and pretrained model configuration
    └── utils.py                       # utils like script conversion, checking for deduplication
```

#### Training Procedure 
```
1. Run **preprocess.py** to convert training data to HF format and generating the Tokenizer Files for the Vanilla tranformer. 
2. Run **train.py** for training and saving the best model. (monitored metric is BLEU with mt13eval tokenizer)
3. Run **split_saving_{model_architecture_type}.py** to quantize the encoder and decoder separately. 
4. Run **inference.py** (with offline = True) for offline inference on the quantized graphs.  

Sample commands with default hyperparameter values are specified in scripts/
```


#### Evaluation Signature: BLEU and chrF
```
{
 "nrefs:1|case:mixed|eff:no|tok:spm-flores|smooth:exp|version:2.2.0",
 "verbose_score":,
 "nrefs": "1",
 "case": "mixed",
 "eff": "no",
 "tok": "spm-flores",
 "smooth": "exp",
 "version": "2.2.0"
}
{
 "name": "chrF2",
 "signature": "nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.2.0",
 "nrefs": "1",
 "case": "mixed",
 "eff": "yes",
 "nc": "6",
 "nw": "0",
 "space": "no",
 "version": "2.2.0"
}
```

#### Datasets Used 
 - [Wixarika]
    Mager, M., Carrillo, D., & Meza, I. (2018). Probabilistic finite-state morphological sgmenter for wixarika (huichol) language. Journal of Intelligent & Fuzzy Systems, 34(5), 3081-3087.
 - [Bribri]
    Feldman, I., & Coto-Solano, R. (2020, December). [Neural Machine Translation Models with Back-Translation for the Extremely Low-Resource Indigenous Language Bribri](https://www.aclweb.org/anthology/2020.coling-main.351.pdf). In Proceedings of the 28th International Conference on Computational Linguistics (pp. 3965-3976).
 - [Mundari]: Data to be released soon. 
 - [Odia] [Punjabi] [Gujarati] and [Assamesse]: @article{10.1162/tacl_a_00452,
    author = {Ramesh, Gowtham and Doddapaneni, Sumanth and Bheemaraj, Aravinth and Jobanputra, Mayank and AK, Raghavan and Sharma, Ajitesh and Sahoo, Sujit and Diddee, Harshita and J, Mahalakshmi and Kakwani, Divyanshu and Kumar, Navneet and Pradeep, Aswin and Nagaraj, Srihari and Deepak, Kumar and Raghavan, Vivek and Kunchukuttan, Anoop and Kumar, Pratyush and Khapra, Mitesh Shantadevi},
    title = "{Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {10},
    pages = {145-162},
    year = {2022},
    month = {02},
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00452},
    url = {https://doi.org/10.1162/tacl\_a\_00452},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00452/1987010/tacl\_a\_00452.pdf},
}


#### Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

#### Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
