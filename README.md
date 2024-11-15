# SimCSE for multi label classification
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
### Unsup SimCSE 
**Unique Label Sampling**\
Unique Label Sampling is a batch construction method that uses label information.
This method considers the diversity within a batch, ensuring that all labels within the same batch are different.

### Sup SimCSE

**SCL** is a contrastive learning method using the MLTC dataset. Sentences with matching labels in the same batch are considered positive pairs, and those without matching labels are considered negative pairs.
SCL only treats samples with exact label matches as positive pairs


**JSCL** is a contrast learning method that computes the similarity between labels using the Jaccard coefficients and applies the similarity as the loss coefficient.


**DSCL** is a contrast learning method that computes the similarity between labels using the Dice coefficients and applies the similarity as the loss coefficient.


**SSCL** is a contrast learning method that computes the similarity between labels using the Simpson (Overlap) coefficients and applies the similarity as the loss coefficient.

**Positive Ensured Sampling**\
The method randomly selects two different samples (xi , Yi ) and (x + i , Y + i ) with matching labels from the article dataset to construct positive examples (xi , x+ i , Yi , Y + i ). Then, construct a batch {(xi , x+ i , Yi , Y + i )} B i=1 consisting of B positive examples. The SCL learning loss is modified as follows. 

## Evaluation
### Training Detail
At each training step, a performance score is computed, with the highest-scoring checkpoint selected for evaluation. 
To facilitate straightforward kNN classification, the Macro-F1 score is assessed on single-label samples within the validation set. 
However, given the limited presence of single-label data in the current dataset, we transform the validation set into a single-label format to ensure reliable evaluation.

### Pruned Problem Transformation (PPT)
A Pruned Problem Transformation (PPT) method is an approach in multi-label classification where the goal is to simplify the classification problem by transforming it into a series of single-label classification problems. 
This is done by pruning the less relevant or infrequent labels from the data set and validation In this case, based on the frequency of labels in the validation data set, the most frequent label in each data set was taken as the label for that data.

## Results

| Model | batch size | Sampling Method | dev kNN | Test Precision | Test Recall | Test mF1 | Test µF1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| princeton-nlp/unsup-simcse-bert-base-uncased |  | - | - | 0.4115 | 0.1345 | 0.1720 | 0.4588 |
| + UnsupSimCSE | 64 | Random | 0.2159 | 0.2408 | 0.0850 | 0.1108 | 0.4339 |
|  | 64 | Unique Label | 0.2918 | 0.2848 | 0.0968 | 0.1259 | 0.4733 |
| + SCL | 64 | Positive Ensured | 0.3590 | 0.5300 | 0.2592 | 0.3067 | 0.7025 |
| + JSCL | 64 | Positive Ensured | 0.3560 | 0.5133 | 0.2627 | 0.3037 | 0.7001 |
| + DSCL | 64 | Positive Ensured | 0.3367 | 0.5214 | 0.2480 | 0.2940 | 0.6927 |
| + SSCL | 64 | Positive Ensured | 0.3560 | 0.4894 | 0.2630 | 0.3042 | 0.7054 |