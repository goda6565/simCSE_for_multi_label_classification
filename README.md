# simCSE for multi label classification
## Datasets
eurlex in [lex_glue](https://huggingface.co/datasets/coastalcph/lex_glue)

Use only series lengths between 200 and 512.

|  | data | 200~512 | 200~512 % |
| --- | --- | --- | --- |
| Train | 55000 | 21956 | 39.92% |
| Dev | 5000 | 2187 | 43.74% |
| Test | 5000 | 1738 | 34.76% |
| Total | 65000 | 25881 | 39.82% |

## Model 
[princeton-nlp/unsup-simcse-bert-base-uncased](https://huggingface.co/princeton-nlp/unsup-simcse-bert-base-uncased)

## Methods
### unsup simCSE (Unique Label Sampling)
Unique Label Sampling is a batch construction method that uses label information.

This method considers the diversity within a batch, ensuring that all labels within the same batch are different.
### sup simCSE (SCL)
SCL is a contrastive learning method using the MLTC dataset.

Sentences with matching labels in the same batch are considered positive pairs, and those without matching labels are considered negative pairs.

SCL only treats samples with exact label matches as positive pairs

## Evaluation
### Training Detail
At each training step, a performance score is computed, with the highest-scoring checkpoint selected for evaluation. 

To facilitate straightforward kNN classification, the Macro-F1 score is assessed on single-label samples within the validation set. 

However, given the limited presence of single-label data in the current dataset, we transform the validation set into a single-label format to ensure reliable evaluation.