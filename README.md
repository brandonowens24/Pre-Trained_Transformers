# Pre-Trained_Transformers

## About
* Jupyter notebook completed in **Google Colab**
* Imported **"sms_spam"** dataset from Huggingface.com
* **Fine-tuned** pre-trained transformer models
    * **BERT**
    * **ELECTRA**
* **Prompt-engineered** two **zero-shot** classfication models to see if they could predict "spam" or "ham"
    * **Bart**
    * **SELECTRA**
* Created **baseline** models to compare fine-tuned pre-trained transformers to
    * **Random-Guess Model**
    * **Target-Guess Model**
    * **BOW Logistic Regression**

## Files
* `Pre_Trained_Transformers.ipynb`:
> Includes all relevant code for the assignment (datasets, pre-trained and fine-tuned models, prompt engineering for zero shot models, and baselines)

* `zero_shot_results.csv`:
> Results from the selected zero shot models with an engineered prompt

* `images`
> Contains images for report

* `report.md`
> Contains all reflections on assignment