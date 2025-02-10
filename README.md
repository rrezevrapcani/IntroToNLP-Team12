<a>
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/Universit%C3%A4t_Bonn.svg/1280px-Universit%C3%A4t_Bonn.svg.png" alt="Uni Bonn Logo" title="UFV" align="right" height="55" />
</a>
<br>
<h1 align = "center"> Final Project - Introduction to NLP  </h1>


<h2 align = "center"> Entity Framing </h2>

This repository contains our implementation to [SemEval-2025 Task 10 Entity Framing](https://propaganda.math.unipd.it/semeval2025task10/). The task was  given a news article and a list of mentions of named entities (NEs) in the article, assign for each such mention one or more roles using a predefined taxonomy of fine-grained roles covering three main type of roles: protagonists, antagonists, and innocent. To address this multi-label multi-class text-span classification task, we fine-tuned BERT for English and BERTimbau for Portuguese.

## Getting Started
you can install the project dependencies using pip, we provide an requirements.txt file with the frozen dependencies:
```bash
pip install -r requirements.txt
```

## Training
Our model is defined in ```model.py``` and our dataset is defined in ```dataset.py```. To train using your data you have to run:
```bash
python src/main.py --data-path <path_to_data> --annotation-file <path_to_annotation> --model-name <bert_or_bertimbau> --batch-size <batch_size> --num-epochs <num_epochs>
```
### Command Line Arguments
- `--data-path` *(required)*: Path to the dataset directory.
- `--annotation-file` *(required)*: Path to the annotation file containing labels and metadata.
- `--model-name` *(required)*: Choose one of the following:
  - `bert-base-uncased` for training on **English** data.
  - `neuralmind/bert-base-portuguese-cased` for training on **Portuguese** data.
- `--batch-size` *(optional, default: `8`)*: Number of samples per batch.
- `--num-epochs` *(optional, default: `25`)*: Number of training epochs.

## Evaluating
To evaluate the trained model, you can run:

```bash
python src/evaluate.py --data-path <path_to_data> --annotation-file <path_to_annotation> --model-checkpoint <path_to_checkpoint> --model-name <bert_or_bertimbau> --output-file <path_to_output>
```

### Command Line Arguments
- `--data-path` *(required)*: Path to the dataset test directory.
- `--annotation-file` *(required)*: Path to the entity mention file containing entity positions in each article.
- `--model-name` *(required)*: Choose one of the following:
  - `bert-base-uncased` for training on **English** data.
  - `neuralmind/bert-base-portuguese-cased` for training on **Portuguese** data.
- `--model-checkpoint` *(required)*: Path to the trained model checkpoint.
- `--output-file` *(optional, default: `predictions.txt`)*: File to store the output predictions in SemEval submission format.

This will generate a `.txt` file containing predictions in a tab-separated format with the following structure:

```
article_id     entity_mention     start_offset     end_offset     main_role(*)     fine-grained_roles(*)
```

To compute the model scores, you can run the following command (this script was provided by SemEval):

```bash
python src/score.py -g <path_to_gold_labels> -p <path_to_model_predictions>
```

## Contact
### Members
---


| [Luísa Ferreira](https://github.com/ferreiraluisa)  |[Elif Yıldırır]() | [Rreze Vrapçani]() | [Ellen Steffes]()
 :------------------------------------------------:  |:------------------------------------------------: | :------------------------------------------------: | :------------------------------------------------: |
|                 Msc. Student¹                  |      Msc. Student¹                | Msc. Student¹                | Msc. Student¹                |
|          <s26ldeso@uni-bonn.de>   |      <s93eyild@uni-bonn.de>   |<s93rvrap@uni-bonn.de>               | <s77estef@uni-bonn.de>

¹University of Bonn \
Bonn, North Rhine-Westphalia, Germany


