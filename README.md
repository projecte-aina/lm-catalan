# Models de Llengua i Datasets<br/> per al Català 💬
# Catalan Language Models & Datasets 💬


A repository for the [AINA project](https://politiquesdigitals.gencat.cat/ca/economia/catalonia-ai/aina/).<br/>
_Repositori del [projecte AINA](https://politiquesdigitals.gencat.cat/ca/economia/catalonia-ai/aina/)._


## Models 🤖
* ✨ <b>new</b> ✨ Ǎguila-7B: https://huggingface.co/projecte-aina/aguila-7b
* roberta-base-ca-v2: https://huggingface.co/projecte-aina/roberta-base-ca-v2
* roberta-large-ca-v2: https://huggingface.co/projecte-aina/roberta-large-ca-v2
* longformer-base-4096-ca-v2: https://huggingface.co/projecte-aina/longformer-base-4096-ca-v2
* BERTa: https://huggingface.co/PlanTL-GOB-ES/roberta-base-ca


Ǎguila-7B Ǎguila is a 7B parameters LLM that has been trained on a mixture of Spanish, Catalan and English data, adding up to a total of 26B tokens. It uses the [Falcon-7b](https://huggingface.co/tiiuae/falcon-7b) model as a starting point, a state-of-the-art English language model that was openly released just a few months ago by the Technology Innovation Institute.

_Ǎguila-7B Ǎguila és un LLM de 7B paràmetres que s'ha entrenat amb dades en castellà, català i anglès, sumant un total de 26B tokens. Utilitza com a punt de partida el model [Falcon-7b](https://huggingface.co/tiiuae/falcon-7b), un model d'última generació en llengua anglesa que l'Technology Innovation Institute va publicar obertament fa només uns mesos._

RoBERTa-base-ca-v2 and BERTa are transformer-based masked language models for the Catalan language. 
They are based on the [RoBERTA](https://github.com/pytorch/fairseq/tree/master/examples/roberta) base model 
and have been trained on a medium-size corpus collected from publicly available corpora and crawlers.

_RoBERTa-base-ca-v2 i BERTa són models de llenguatge basat en transformers per a la llengua catalana. Es basen en el model [RoBERTa-base](https://github.com/pytorch/fairseq/tree/master/examples/roberta) i han estat entrenat en un corpus de mida mitjana, a partir de corpus diponibles públicament i crawlers._

### Tokenization and pretraining 🧩

The training corpus has been tokenized using a byte version of [Byte-Pair Encoding (BPE)](https://github.com/openai/gpt-2) used in the original [RoBERTA](https://github.com/pytorch/fairseq/tree/master/examples/roberta) model with a vocabulary size of 52,000 tokens. 

The RoBERTa-ca-v2 pretraining consists of a masked language model training that follows the approach employed for the RoBERTa base model with the same hyperparameters as in the original work. With 16 NVIDIA V100 GPUs of 16GB DDRAM the training lasted a total of 48 hours for BERTa and a total of 96 hours for RoBERTa-ca-v2.

_El corpus d'entrenament ha estat tokenitzat fent servir un [(BPE) a nivell de bytes](https://github.com/openai/gpt-2) utilitzat en el model [RoBERTA](https://github.com/pytorch/fairseq/tree/master/examples/roberta) original, amb una vocabulari de 52.000 tokens._

_El pre-entrenament de RoBERTa-ca-v2 consisteix en un entrenament de model de llenguatge per masking, seguint l'enfoc que es va utilitzar per al model RoBERTa-base, amb els mateixos hiperparàmetres que en treball original. L'entrenament es va fer amb 16 GPUs NVIDIA V100 de 16GB DDRAM i va durar 48 hores per al model BERTa i 96 per al RoBERTa-ca-v2._

## Word embeddings (FastText) 🔤

* Catalan Word Embeddings in FastText: https://doi.org/10.5281/zenodo.4522040

Generated from a curated corpus of over 10GB of high-quality text.
_Generat a partir d'un corpus seleccionat de més de 10 GB de text d'alta qualitat._

## Word Embeddings (more efficient Floret version)

* Catalan CBOW Word Embeddings in Floret: https://zenodo.org/record/7330331

Trained using an expansive Catalan textual corpus, comprising over 34GB of data, through the floret method.
_Entrenat utilitzant un corpus textual català de 34 GB de dades mitjançant el mètode floret._

## Training corpora
The training corpora consists of several corpora gathered from web crawling and public corpora.
_Els corpus d'entrenament són la suma de diversos corpus obtinguts a partir de corpus publics i crawlings del web._

### roberta-base-ca-v2

| Corpus                                                            | Size in GB |
|-------------------------                                          |------------|
| Catalan Crawling                                                  | 13.00      |
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

The Catalan Textual Corpus can be found in the following link: https://doi.org/10.5281/zenodo.4519348.

_A fi d'obtenir un corpus d'entrenament d'alta qualitat, cada corpus ha estat processat amb una pipeline d'operacions, incloent separació de frases, detecció d'idioma, filtratge de frases mal formades i deduplicació de continguts repetitius, entre d'altres. Durant el procés, hem mantingut els límits dels documents. Finalment, hem concatenat els corpus i hem aplicat una nova dedupliació._

_En el següent enllaç podeu trobar el Catalan Textual Corpus: https://doi.org/10.5281/zenodo.4519348._

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

For a complete list, refer to. _Per obtenir una llista completa, consulteu el següent enllaç_: https://huggingface.co/projecte-aina/

### Fine-tuning
The fine-tuning scripts for the downstream tasks are available in the following link: https://github.com/projecte-aina/club.<br/>
They are based on the HuggingFace [**Transformers**](https://github.com/huggingface/transformers) library.

_Els scripts de fine-tuning per aquestes tasques es poden trobar en el següent enllaç: https://github.com/projecte-aina/club.<br/>
Es basen en la llibreria [**Transformers**](https://github.com/huggingface/transformers) de HuggingFace._

## spaCy models

* ca_bsc_core_trf: https://huggingface.co/projecte-aina/ca_bsc_core_trf.
  
  Spacy 3.5 version with enhanced dictionaries for better coverage, using projecte-aina/roberta-large-ca-v2 model with multitask training.
  _Versió Spacy 3.5 amb diccionaris millorats per a una millor cobertura, utilitzant el model project-aina/roberta-large-ca-v2 amb entrenament multitasca._

* ca_bsc_demo_trf: https://huggingface.co/projecte-aina/ca_bsc_demo_trf.

  Catalan transformer (projecte-aina/roberta-large-ca-v2) pipeline by BSC. Components: transformer, morphologizer, parser, ner, attribute_ruler, lemmatizer, text classification.

Available trained pipelines for Catalan in spaCy. _Pipelines per al català disponibles a spaCy_: https://spacy.io/models/ca

## Datasets 🗂️

|name           | task                                    | link                                                        |
|---------------|-----------------------------------------|-------------------------------------------------------------|
| ancora-ca-ner | Named Entity Recognition                | https://huggingface.co/datasets/projecte-aina/ancora-ca-ner |
| ancora-ca-pos | Part of Speech tagging                  | https://huggingface.co/datasets/universal_dependencies      |
| STS-ca        | Semantic Textual Similarity             | https://huggingface.co/datasets/projecte-aina/sts-ca        |
| TeCla         | Text Classification                     | https://huggingface.co/datasets/projecte-aina/tecla         |
| TECa          | Textual Entailment                      | https://huggingface.co/datasets/projecte-aina/teca          |
| VilaQuAD      | Extractive Question Answering           | https://huggingface.co/datasets/projecte-aina/vilaquad      |
| ViquiQuAD     | Extractive Question Answering           | https://huggingface.co/datasets/projecte-aina/viquiquad     |
| CatalanQA     | Extractive Question Answering           | https://huggingface.co/datasets/projecte-aina/catalanqa     |
| xquad-ca      | Extractive Question Answering           | https://huggingface.co/datasets/projecte-aina/xquad-ca      |

For a complete list, refer to. _Per obtenir una llista completa, consulteu_: https://huggingface.co/projecte-aina/

## CLUB: Catalan Language Understanding Benchmark 🏆
The CLUB benchmark consists of 6 tasks: Named Entity Recognition (NER), Part-of-Speech Tagging (POS), Semantic Textual Similarity (STS), Text Classification (TC), Textual Entailment (TE), and Question Answering (QA). 

_El benchmark CLUB consisteix en 6 tasques: reconeixement d'entitats (NER), etiquetat de categoria gramatical (POS), similitut textual semàntica (STS), classificació textual (TC), implicació textual (TE) i resposta de preguntes (QA)._

### Results ✅

| Task        | NER (F1)      | POS (F1)   | STS (Combined)   | TC (Accuracy) | TE (Accuracy) | QA (Vilaquad) (F1/EM)| QA (ViquiQuAD) (F1/EM) | QA (CatalanQA) (F1/EM) | QA (XQuAD-Ca)* (F1/EM) | 
| ------------|:-------------:| -----:|:------|:------|:-------|:------|:----|:----|:----|
| RoBERTa-base-ca-v2      | **89.45** | 99.09 | 79.07 | **74.26** | **83.14** | **87.74/72.58** | **88.72/75.91** | **89.50**/76.63 | **73.64/55.42** |
| BERTa      | 88.94 | **99.10** | **80.19** | 73.65 | 79.26 | 85.93/70.58 | 87.12/73.11 | 89.17/**77.14** | 69.20/51.47 |
| mBERT      | 87.36 | 98.98 | 74.26 | 69.90 | 74.63 | 82.78/67.33 | 86.89/73.53 | 86.90/74.19 | 68.79/50.80 |
| XLM-RoBERTa      | 88.07 | 99.03 | 61.61 | 70.14 | 33.30 | 86.29/71.83 | 86.88/73.11 | 88.17/75.93 | 72.55/54.16 |

*: Trained on CatalanQA, tested on XQuAD-Ca.

For more information, refer to. _Per a més informació, consulteu el següent enllaç_ https://club.aina.bsc.es/ 

## Demos

* Bot: Demo of the integration of voice functionalities in Catalan. _Demostració d'incorporació de funcionalitats de veu en català._
  - Link to demo. _Enllaç a la demo_: https://bot.aina.bsc.es/#/
  - Link to code. _Enllaç al codi_: https://github.com/projecte-aina/minibot
    
* spaCy: Demo of the capabilities of natural language processing chains and spaCy models implemented within the AINA project. _Demostrador de les capacitats de les cadenes de processament de llenguatge natural i models spaCy implementats dins del projecte AINA._
  - Link to demo. _Enllaç a la demo_: https://aina.bsc.es/apps/spacy
  - Link to code. _Enllaç al codi_: https://github.com/projecte-aina/spacy_ca_demo

* ViquiQA: Demon of the Question and Answer model trained with the CatalanQA dataset using the Catalan Wikipedia. _Demostrador del model de Pregunta i Resposta entrenat amb el dataset CatalanQA fent servir la Viquipèdia en català._
  - Link to demo. _Enllaç a la demo_: https://aina.bsc.es/apps/viquiqa

* Traductor: Automatic translators between Catalan and Spanish (general and specialized administrative-legal text) and between Catalan and English (general text). _Traductors automàtics entre català i castellà (text general i d'especialitat administratiu-legal) i entre català i anglés (text general)._
  - Link to demo. _Enllaç a la demo_: https://aina.bsc.es/apps/traductor

* oTranscribe+: Free and private speech recognition web app for transcribing recorded interviews. _Aplicació web de reconeixement de veu gratuïta i privada per a la transcripció d'entrevistes gravades._
  - Link to demo. _Enllaç a la demo_: https://otranscribe.bsc.es/

* CLUB: Platform for comparative evaluation of language models for Catalan. _Plataforma d'avaluació comparativa de models de llengua per al català._
  - Link to demo. _Enllaç a la demo_: https://club.aina.bsc.es/

* TTS: Multi-speaker speech synthesis engine demo. _Demostrador del motor de síntesi de parla multi parlant._
  - Link to demo. _Enllaç a la demo_: https://aina.bsc.es/apps/tts
  
  
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

