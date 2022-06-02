# Models de Llengua i Datasets<br/> per al Català 💬
# Catalan Language Models & Datasets 💬


A repository for the [AINA project](https://politiquesdigitals.gencat.cat/en/tic/aina-el-projecte-per-garantir-el-catala-en-lera-digital/index.html).<br/>
_Repositori del [projecte AINA](https://politiquesdigitals.gencat.cat/ca/tic/aina-el-projecte-per-garantir-el-catala-en-lera-digital/)._


## Models 🤖
* ✨ <b>new</b> ✨ roberta-base-ca-v2: https://huggingface.co/projecte-aina/roberta-base-ca-v2
* BERTa: https://huggingface.co/PlanTL-GOB-ES/roberta-base-ca

RoBERTa-base-ca-v2 and BERTa are transformer-based masked language models for the Catalan language. 
They are based on the [RoBERTA](https://github.com/pytorch/fairseq/tree/master/examples/roberta) base model 
and have been trained on a medium-size corpus collected from publicly available corpora and crawlers.

_RoBERTa-ca-v2 i BERTa són models de llenguatge basat en transformers per a la llengua catalana. Es basen en el model [RoBERTa-base](https://github.com/pytorch/fairseq/tree/master/examples/roberta) i ha estat entrenat en un corpus de mida mitjana, a partir de corpus diponibles públicament i crawlers._

### Tokenization and pretraining 

The training corpus has been tokenized using a byte version of [Byte-Pair Encoding (BPE)](https://github.com/openai/gpt-2)
used in the original [RoBERTA](https://github.com/pytorch/fairseq/tree/master/examples/roberta) model with a vocabulary size of 52,000 tokens. 
The RoBERTa-ca-v2 pretraining consists of a masked language model training that follows the approach employed for the RoBERTa base model with the same hyperparameters as in the original work.
With 16 NVIDIA V100 GPUs of 16GB DDRAM the training lasted a total of 48 hours for BERTa and a total or 96 hours for RoBERTa-ca-v2.

_El corpus d'entrenament ha estat tokenitzat fent servir un [(BPE) a nivell de bytes](https://github.com/openai/gpt-2) utilitzat en el model [RoBERTA](https://github.com/pytorch/fairseq/tree/master/examples/roberta) original, amb una vocabulari de 52.000 tokens._

_El pre-entrenament de BERTa consisteix en un entrenament de model de llenguatge per masking, seguint l'enfoc que es va utilitzar per al model RoBERTa-base, amb els mateixos hiperparàmetres que en treball original. _
_L'entrenament es va fer amb 16 GPUs NVIDIA V100 de 16GB DDRAM i va durar 48 hores per al model BERTa i 96 per al RoBERTa-ca-v2._

## Word embeddings (FastText) 🔤

https://doi.org/10.5281/zenodo.4522040

## Training corpora
The training corpora consists of several corpora gathered from web crawling and public corpora.
_Els corpus d'entrenament són la suma de diversos corpus obtinguts a partir de corpus publics i crawlings del web._

### roberta-base-ca-v2

| Corpus                                                            | Size in GB |
|-------------------------                                          |------------|
| BNE-ca                                                            | 13.00      |
| [Wikipedia](https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/cawiki/20200801/) | 1.10       |
| [DOGC](http://opus.nlpl.eu/DOGC-v2.php)                           | 0.78       |
| [Catalan Open Subtitles](http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/mono/OpenSubtitles.raw.ca.gz)   | 0.02       |
| [Catalan Oscar](https://traces1.inria.fr/oscar/)                  | 4.00       |
| [CaWaC](http://nlp.ffzg.hr/resources/corpora/cawac/)              | 3.60       |
| [Cat. General Crawling](https://doi.org/10.5281/zenodo.4636227)   | 2.50       |
| [Cat. Goverment Crawling](https://doi.org/10.5281/zenodo.4636485) | 0.24       |
| [Cat. News Agency](https://www.acn.cat/)                          | 0.42       |
| Padicat                                                           | 0.63       |
| [RacoCatalá](https://www.racocatala.cat/)                         | 8.10       |
| [Nació Digital](https://www.naciodigital.cat/)                    | 0.42       |
| [Vilaweb](https://www.vilaweb.cat/)                               | 0.06       |
| Tweets                                                            | 0.02       |

### BERTa and Word embeddings

| Corpus                                                            | Size in GB|
|---------                                                          |-----------|
|[DOGC](http://opus.nlpl.eu/DOGC-v2.php)                            | 0.801     |
|[Cat. Open Subtitles](http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/mono/OpenSubtitles.raw.ca.gz)  |0.019   |
|[Cat. OSCAR](https://traces1.inria.fr/oscar/)                      | 4         |
|[CaWac](http://nlp.ffzg.hr/resources/corpora/cawac/)               | 3.6       |
|[Cat. Wikipedia](https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/cawiki/20200801/)| 0.98    |
|[Cat. General Crawling](https://doi.org/10.5281/zenodo.4636227)    | 2.6       |
|[Cat. Goverment Crawling](https://doi.org/10.5281/zenodo.4636485)  | 0.247     |
|[Cat. News Agency](https://www.acn.cat/)                           | 0.447     |

To obtain a high-quality training corpus, each corpus has been preprocessed with a pipeline of different operations, including, among the others, sentence splitting, language detection, filtering of badly-formed sentences and deduplication of repetitive contents. During the process, we kept document boundaries. Finally, the corpora are concatenated and further global deduplication among them is applied.

The Catalan Textual Corpus can be found here: https://doi.org/10.5281/zenodo.4519348.

_A fi d'obtenir un corpus d'entrenament d'alta qualitat, cada corpus ha estat processat amb una pipeline d'operacions, incloent separació de frases, detecció d'idioma, filtratge de frases mal formades i deduplicació de continguts repetitius, entre d'altres. Durant el procés, hem mantingut els límits dels documents. Finalment, hem concatenat els corpus i hem aplicat una nova dedupliació._

_Aquí podeu trobar el Catalan Textual Corpus: https://doi.org/10.5281/zenodo.4519348._

### Tokenization and pretraining 🧩

The training corpora have been tokenized using a byte version of [Byte-Pair Encoding (BPE)](https://github.com/openai/gpt-2) used in the original [RoBERTA](https://github.com/pytorch/fairseq/tree/master/examples/roberta) model with a vocabulary size of 52,000 tokens. 

The BERTa pretraining consists of a masked language model training that follows the approach employed for the RoBERTa base model with the same hyperparameters as in the original work. The training lasted a total of 48 hours with 16 NVIDIA V100 GPUs of 16GB DDRAM.

_Els corpus d'entrenament han estat tokenitzat fent servir un [(BPE) a nivell de bytes](https://github.com/openai/gpt-2) utilitzat en el model [RoBERTA](https://github.com/pytorch/fairseq/tree/master/examples/roberta) original, amb una vocabulari de 52.000 tokens._

_El pre entrenament de BERTa consisteix en un entrenament de model de llenguatge per masking, seguint l'enfoc que es va utilitzar per al model RoBERTa-base, amb els mateixos hiperparàmetres que en treball original. L'entrenament va durar 48 hores amb 16 GPUs NVIDIA V100 de 16GB DDRAM._

## Usage example ⚗️
For the RoBERTa-base
```python
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer, FillMaskPipeline
from pprint import pprint
tokenizer_hf = AutoTokenizer.from_pretrained('BSC-TeMU/roberta-base-ca-v2')
model = AutoModelForMaskedLM.from_pretrained('BSC-TeMU/roberta-base-ca-v2')
model.eval()
pipeline = FillMaskPipeline(model, tokenizer_hf)
text = f"¡Hola <mask>!"
res_hf = pipeline(text)
pprint([r['token_str'] for r in res_hf])
```

## Fine-tuned models 🧗🏼‍♀️🏇🏼🤽🏼‍♀️🏌🏼‍♂️🏄🏼‍♀️

Fine-tuned from BERTa model:

- roberta-base-ca-cased-ner for NER: https://huggingface.co/projecte-aina/roberta-base-ca-cased-ner
- roberta-base-ca-cased-pos for POS: https://huggingface.co/projecte-aina/roberta-base-ca-cased-pos
- roberta-base-ca-cased-tc for text classification: https://huggingface.co/projecte-aina/roberta-base-ca-cased-tc
- roberta-base-ca-cased-te for textual entailment: https://huggingface.co/projecte-aina/roberta-base-ca-cased-te
- roberta-base-ca-cased-sts for semantic textual similarity (STS): https://huggingface.co/projecte-aina/roberta-base-ca-cased-sts
- roberta-base-ca-cased-sts for extractive question answering (QA): https://huggingface.co/projecte-aina/roberta-base-ca-cased-qa

### Fine-tuning
The fine-tuning scripts for the downstream tasks are available here: https://github.com/projecte-aina/club.<br/>
They are based on the HuggingFace [**Transformers**](https://github.com/huggingface/transformers) library.

_Els scripts de fine-tuning per aquestes tasques es poden trobar aquí: https://github.com/projecte-aina/club.<br/>
Es basen en la llibreria [**Transformers**](https://github.com/huggingface/transformers) de HuggingFace._

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

## CLUB: Catalan Language Understanding Benchmark 🏆
The CLUB benchmark consists of 5 tasks: Part-of-Speech Tagging (POS), Named Entity Recognition (NER), Text Classification (TC), Semantic Textual Similarity (STS) and Question Answering (QA). 

_El benchmark CLUB consisteix en 5 tasques: etiquetat de categoria gramatical (POS), reconeixement d'entitats (NER), classificació textual (TC), similitut textual semàntica (STS) i resposta de preguntes (QA)._

### Results ✅


| Task        | NER (F1)      | POS (F1)   | STS (Pearson)   | TC (accuracy) | QA (ViquiQuAD) (F1/EM)  | QA (XQuAD) (F1/EM) | 
| ------------|:-------------:| -----:|:------|:-------|:------|:----|
| RoBERTa-base-ca-v2       | **89.84** | **99.07** | **79.98** | **83.41** | **88.04/74.65** | **71.50/53.41** |
| BERTa       | 88.13 | 98.97 | 79.73 | 74.16 | 86.97/72.29 | 68.89/48.87 |
| mBERT       | 86.38 | 98.82 | 76.34 | 70.56 | 86.97/72.22 | 67.15/46.51 |
| XLM-RoBERTa | 87.66 | 98.89 | 75.40 | 71.68 | 85.50/70.47 | 67.10/46.42 |
| WikiBERT-ca | 77.66 | 97.60 | 77.18 | 73.22 | 85.45/70.75 | 65.21/36.60 |

For more information, refer [here](https://github.com/projecte-aina/club).

_[Aquí](https://github.com/projecte-aina/club) trobareu més informació._

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
📋 We are interested in (1) extending our corpora to make larger models (2) train/evaluate the model in other tasks.<br/>
For questions regarding this work, contact us at aina@bsc.es

📋 _Necessitem (1) ampliar el nostre corpus per poder fer models més grans i (2) entrenar i avaluar el model en més tasques._<br/>
_Per qualsevol cosa relacionada amb aquesta feina, podeu contactar-nos a aina@bsc.es_

