# Report

## Dataset
This dataset was pulled from [Huggingface.com](https://huggingface.co/datasets/sms_spam) and is composed of ~5,500 rows of text messages. These text messages are labelled as being either "spam" or "ham".
* **Spam:** Irrelevant/inappropriate messages sent on the internet to a large number of recipients.
* **Ham:** The opposite of spam -- relevant/personal messages sent from another individual.

![Dataset](sms_spam_dataset.png)

The dataset was eventually split into a training and testing set with a roughly 80-20 division:
* 4,000 training parameters
* ~1,500 testing parameters

Here, **F1 Score** for spam messages is important. We are not really interested in the real messages, but just being able to determine which messages are actually spam or incorrectly classified as real from the spam messages. 

## Fine-Tuned Models
The pre-trained transformer models selected included: 
* [**BERT-small**](https://huggingface.co/prajjwal1/bert-small)
  * A smaller pre-trained BERT variant
  * ~111M parameters
  * 12 encoders with 12 bidirectional self-attention heads
    
* [**ELECTRA-base-emotion**](https://huggingface.co/bhadresh-savani/electra-base-emotion)
  * A specific model of electra used with an emotion dataset
  * ~128M parameters
  * BERT is underlying model

My dataset needed to be tokenized according to the models already set up tokenization, split into training and testing, and applied to an AutoSequence Classifier. It was then applied to a trainer and eventually ran on the testing dataset to compute the desired F1 Scores.

## Zero-Shot Classification
For zero-shot classification I utilized:

* [**BART**](https://huggingface.co/docs/transformers/en/model_doc/bart)
  * BART Large trained on the MNLI dataset
  * 407M parameters
* [**DeBERTa-v3**]([https://huggingface.co/Recognai/zeroshot_selectra_medium](https://huggingface.co/sileod/deberta-v3-base-tasksource-nli)https://huggingface.co/sileod/deberta-v3-base-tasksource-nli)
  * BERT model fine-tuned with 600+ tasks
  * 184M parameters
