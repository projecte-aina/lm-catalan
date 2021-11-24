# Models de Llenguatge i Datasets<br/> per al Català 
# Catalan Language Models & Datasets


A repository for the AINA project.<br/>
_Repositori del projecte AINA._

## Corpora 📃

| Dataset               | Original n. of tokens | Final n. of tokens    | 
|---------              |----------------------:|----------------------:|
|1 DOGC                 |            126.65     |  126.65               | 
|2 Cat. Open Subtitles  | 3.52                  | 3.52                  | 
|3 Cat. OSCAR           | 1,355.53              | 695.37                | 
|4 CaWaC                | 1,535.10              | 650.98                | 
|5 Wikipedia            | 198.36                | 167.47                | 
|6 Cat. Gen. Crawling   | 1,092.98              | 434.82                | 
|7 Cat. Gov. Crawling   | 303.10                | 39.12                 | 
|8 ACN                  | 81.28                 | 75.61                 | 
|**Total**              | **4,696.52**          | **2,193.54**          | 
|**Deduplicated (CaText)** |                    | **1,770.32**          |

The Catalan Textual Corpus can be found here: https://doi.org/10.5281/zenodo.4519348<br/>
_Aquí podeu trobar el Catalan Textual Corpus: https://doi.org/10.5281/zenodo.4519348_

## Models 🤖
...

## Fine-tunned models 🧗🏼‍♀️🏇🏼🤽🏼‍♀️🏌🏼‍♂️🏄🏼‍♀️

- roberta-base-ca-cased-ner for NER: https://huggingface.co/projecte-aina/roberta-base-ca-cased-ner
- roberta-base-ca-cased-pos for POS: https://huggingface.co/projecte-aina/roberta-base-ca-cased-pos


## Word embeddings 🔤

???

## Datasets 🗂️

...

## Evaluation ✅
 | model	   	           | NERC    		 | POS		     | STS		     | TC		     | QA(ViquiQuAD		     | QA(XQuAD)	       	 | 
 | ---------------------   | :-----------:   | :-----------: | :-----------: | :-----------: | :-------------------: | :-----------------:   | 
 | BERTa		           | 88.13(2)		 | 98.97(10)	 | 79.73(5)		 | 74.16(9)		 | 86.97/72.29(9)		 | 68.89/48.87(9)		 | 
 | BERTa+decontaminate     | 89.10(6)		 | 98.94(6)		 | 81.13(8)		 | 73.84(10		 | 86.50/70.82(6)		 | 68.61/47.26(6)		 | 
 | mBERT		           | 86.38(9)		 | 98.82(9)		 | 76.34(9)		 | 70.56(10)	 | 86.97/72.22(8)		 | 67.15/46.51(8)		 | 
 | WikiBERT-ca		       | 77.66(9)		 | 97.60(6)		 | 77.18(10)	 | 73.22(10)	 | 85.45/70.75(10)	     | 65.21/36.60(10)	     | 
 | XLM-RoBERTa		       | 87.66(8)		 | 98.89(10)	 | 75.40(10)	 | 71.68(10)	 | 85.50/70.47(5)		 | 67.10/46.42(5)		 | 


## Usage example ⚗️
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
