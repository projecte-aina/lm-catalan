# Models de Llengua i Datasets<br/> per al Català 
# Catalan Language Models & Datasets


A repository for the AINA project.<br/>
_Repositori del projecte AINA._

## Corpora 📃
The training corpus consists of several corpora gathered from web crawling and public corpora.<br/>
_El corpus d'entrenament és la suma de diversos corpus obtinguts a partir de corpus publics i crawlings del web._

| Dataset               | Original n. of tokens | Final n. of tokens    | Size | Sort of documents                |
|---------              |----------------------:|----------------------:|-----:|----------------------------------|
|1 [DOGC](http://opus.nlpl.eu/DOGC-v2.php)                 |            126.65     |  126.65               | -    | documents from the Official Gazette of the Catalan Government |
|2 [Cat. Open Subtitles](http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/mono/OpenSubtitles.raw.ca.gz)  | 3.52                  | 3.52                  | -    | translated movie subtitles       |
|3 [Cat. OSCAR](https://traces1.inria.fr/oscar/)          | 1,355.53              | 695.37                | -    | monolingual corpora, filtered from [Common Crawl](https://commoncrawl.org/about/) |
|4 [CaWac](http://nlp.ffzg.hr/resources/corpora/cawac/)                | 1,535.10              | 650.98                | -    | a web corpus built from the .cat top-level-domain |
|5 [Cat. Wikipedia](https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/cawiki/20200801/)            | 198.36                | 167.47                | -    | Catalan Wikipedia articles       |
|6 Cat. Gen. Crawling   | 1,092.98              | 434.82                | -    | crawling of the 500 most popular .cat and .ad domains |
|7 Cat. Gov. Crawling   | 303.10                | 39.12                 | -    | crawling the [.gencat](gencat.cat) domain and subdomains |
|8 [Cat. News Agency](https://www.acn.cat/)                  | 81.28                 | 75.61                 | -    | newswire text  |
|**Total**              | **4,696.52**          | **2,193.54**          | -    | |
|**Deduplicated (CaText)** |                    | **1,770.32**          |**10,9GB**    | https://doi.org/10.5281/zenodo.4519348 |

To obtain a high-quality training corpus, each corpus have preprocessed with a pipeline of operations, including among the others, sentence splitting, language detection, filtering of bad-formed sentences and deduplication of repetitive contents. During the process, we keep document boundaries are kept. Finally, the corpora are concatenated and further global deduplication among the corpora is applied.<br/>
_A fi d'obtenir un corpus d'entrenament d'alta qualitat, cada corpus ha estat processat amb una pipeline d'operacions, incloent separació de frases, detecció d'idioma, filtratge de frases mal formades i deduplicació de continguts repetitius, entre d'altres. Durant el procés, hem mantingut els límits dels documents. Finalment, hem concatenat els corpus i hem aplicat una nova dedupliació._

The Catalan Textual Corpus can be found here: https://doi.org/10.5281/zenodo.4519348<br/>
_Aquí podeu trobar el Catalan Textual Corpus: https://doi.org/10.5281/zenodo.4519348_

## Models 🤖
### BERTa: RoBERTa-based Catalan language model 
BERTa is a transformer-based masked language model for the Catalan language. It is based on the RoBERTA base model and has been trained on a medium-size corpus collected from publicly available corpora and crawlers.<br/>
_BERTa és un model de llenguatge basat en transformers per a la llengua catalana. Es basa en el model RoBERTa-base i ha estat entrenat en un corpus de mida mitjana, a partir de corpus diponibles públicament i crawlers._

Available here: https://huggingface.co/PlanTL-GOB-ES/roberta-base-ca

#### Tokenization and pretraining 

The training corpus has been tokenized using a byte version of [Byte-Pair Encoding (BPE)](https://github.com/openai/gpt-2) used in the original [RoBERTA](https://github.com/pytorch/fairseq/tree/master/examples/roberta) model with a vocabulary size of 52,000 tokens. 

The BERTa pretraining consists of a masked language model training that follows the approach employed for the RoBERTa base model with the same hyperparameters as in the original work. The training lasted a total of 48 hours with 16 NVIDIA V100 GPUs of 16GB DDRAM.

_El corpus d'entrenament ha estat tokenitzat fent servir ...  a byte version of [Byte-Pair Encoding (BPE)](https://github.com/openai/gpt-2) utilitzat en el model [RoBERTA](https://github.com/pytorch/fairseq/tree/master/examples/roberta) original, amb una vocabulari de 52.000 tokens._

_El pre entrenament de BERTa consisteix en un entrenament de model de llenguatge per masking (?), seguint l'enfoc que es va utilitzar per al model RoBERTa-base, amb els mateixos hiperparàmetres que en treball original. L'entrenament va durar 48 hores amb 16 GPUs NVIDIA V100 de 16GB DDRAM._

#### Usage example ⚗️
For the RoBERTa-base
```python
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer, FillMaskPipeline
from pprint import pprint
tokenizer_hf = AutoTokenizer.from_pretrained('BSC-TeMU/roberta-base-ca')
model = AutoModelForMaskedLM.from_pretrained('BSC-TeMU/roberta-base-ca')
model.eval()
pipeline = FillMaskPipeline(model, tokenizer_hf)
text = f"¡Hola <mask>!"
res_hf = pipeline(text)
pprint([r['token_str'] for r in res_hf])
```

## Fine-tuned models 🧗🏼‍♀️🏇🏼🤽🏼‍♀️🏌🏼‍♂️🏄🏼‍♀️

- roberta-base-ca-cased-ner for NER: https://huggingface.co/projecte-aina/roberta-base-ca-cased-ner
- roberta-base-ca-cased-pos for POS: https://huggingface.co/projecte-aina/roberta-base-ca-cased-pos
- roberta-base-ca-cased-tc for text classification: https://huggingface.co/projecte-aina/roberta-base-ca-cased-tc
- roberta-base-ca-cased-te for textual entailment: https://huggingface.co/projecte-aina/roberta-base-ca-cased-te
- roberta-base-ca-cased-sts for semantic textual similarity (STS): https://huggingface.co/projecte-aina/roberta-base-ca-cased-sts
- roberta-base-ca-cased-sts for extractive question answering (QA): https://huggingface.co/projecte-aina/roberta-base-ca-cased-qa

### Fine-tuning
The fine-tuning scripts for the downstream taks are available here: https://github.com/projecte-aina/berta.<br/>
They are based on the HuggingFace [**Transformers**](https://github.com/huggingface/transformers) library.

## Word embeddings 🔤

https://doi.org/10.5281/zenodo.4522040

## Datasets 🗂️

|name           | task                                    | link                                                        |
|---------------|-----------------------------------------|-------------------------------------------------------------|
| ancora-ca-pos | Part of Speech tagging                  | https://huggingface.co/datasets/universal_dependencies      |
| ancora-ca-ner | Named Entity Recognition                | https://huggingface.co/datasets/projecte-aina/ancora-ca-ner |
| STS-ca        | Semantic Textual Similarity             | https://huggingface.co/datasets/projecte-aina/sts-ca        |
| TECa          | Textual Entailment                      | https://huggingface.co/datasets/projecte-aina/teca          |
| TeCla         | Text Classification                     | https://huggingface.co/datasets/projecte-aina/tecla         |
| xquad-ca      | Extractive Question Answering           | https://huggingface.co/datasets/projecte-aina/xquad-ca      |
| VilaQuAD      | Extractive Question Answering           | https://huggingface.co/datasets/projecte-aina/vilaquad      |
| ViquiQuAD     | Extractive Question Answering           | https://huggingface.co/datasets/projecte-aina/viquiquad     |

## CLUB: Catalan Language Understanding Benchmark
The CLUB benchmark consists of 5 tasks, that are Part-of-Speech Tagging (POS), Named Entity Recognition (NER), Text Classification (TC), Semantic Textual Similarity (STS) and Question Answering (QA). 

### Results ✅

| Model        | NER (F1)      | POS (F1)   | STS (Pearson)   | TC (accuracy) | QA (ViquiQuAD) (F1/EM)  | QA (XQuAD) (F1/EM) | TE (TECA) (accuracy) | 
| ------------|:-------------:| -----:|:------|:-------|:------|:----|:----|
| BERTa       | **89.63** | **98.93** | **81.20** | **74.04** | **86.99/73.25** | **67.81/49.43** | 79.12 |
| mBERT       | 86.38 | 98.82 | 76.34 | 70.56 | 86.97/72.22 | 67.15/46.51 | x |
| XLM-RoBERTa | 87.66 | 98.89 | 75.40 | 71.68 | 85.50/70.47 | 67.10/46.42 | x |
| WikiBERT-ca | 77.66 | 97.60 | 77.18 | 73.22 | 85.45/70.75 | 65.21/36.60 | x |

For more information, refer [here](https://github.com/projecte-aina/berta).


## Other Catalan Language Models 👩‍👧‍👦
...
## Cite 📣
```
@inproceedings{armengol-estape-etal-2021-multilingual,
    title = "Are Multilingual Models the Best Choice for Moderately Under-resourced Languages? {A} Comprehensive Assessment for {C}atalan",
    author = "Armengol-Estap{\'e}, Jordi  and
      Carrino, Casimiro Pio  and
      Rodriguez-Penagos, Carlos  and
      de Gibert Bonet, Ona  and
      Armentano-Oller, Carme  and
      Gonzalez-Agirre, Aitor  and
      Melero, Maite  and
      Villegas, Marta",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.437",
    doi = "10.18653/v1/2021.findings-acl.437",
    pages = "4933--4946",
}
```

## Contact 📧
📋 We are interested in (1) extending our corpora to make larger models (2) train/evaluate the model in other tasks.

For questions regarding this work, contact us at aina@bsc.es
