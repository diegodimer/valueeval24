import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import ast
from sklearn.metrics import f1_score
BASE = "/home/users/ddimer/subfolder/files"
BERT_BASE_PATH = "bert-base-uncased/"
BERT_BASE_TOKENIZER = BERT_BASE_PATH + "tokenizer"
BERT_BASE_MODEL =  "bert-base-uncased-fine-tuned"

BERT_LARGE_PATH = "bert-large-uncased/"
BERT_LARGE_TOKENIZER = BERT_LARGE_PATH + "tokenizer"
BERT_LARGE_MODEL =  "bert-large-uncased-fine-tuned"

ROBERTA_BASE_PATH = "roberta-base/"
ROBERTA_BASE_TOKENIZER = ROBERTA_BASE_PATH + "tokenizer"
ROBERTA_BASE_MODEL =  "roberta-base-fine-tuned"

ROBERTA_LARGE_PATH = "roberta-large/"
ROBERTA_LARGE_TOKENIZER = ROBERTA_LARGE_PATH + "tokenizer"
ROBERTA_LARGE_MODEL =  "roberta-large-fine-tuned"

DEBERTA_LARGE_PATH = "deberta-large/"
DEBERTA_LARGE_TOKENIZER = DEBERTA_LARGE_PATH + "tokenizer"
DEBERTA_LARGE_MODEL =  "deberta-large-fine-tuned"

DEBERTA_PATH = "deberta-base/"
DEBERTA_TOKENIZER = DEBERTA_PATH + "tokenizer"
DEBERTA_MODEL =  "deberta-base-fine-tuned"

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


TARGET_LIST = ['Self-direction: thought', 'Self-direction: action', 'Stimulation', 'Hedonism', 'Achievement', 'Power: dominance', 'Power: resources', 'Face', 'Security: personal', 'Security: societal', 'Tradition', 'Conformity: rules', 'Conformity: interpersonal', 'Humility', 'Benevolence: caring', 'Benevolence: dependability', 'Universalism: concern', 'Universalism: nature', 'Universalism: tolerance']

bert_model = AutoModelForSequenceClassification.from_pretrained(BASE + BERT_BASE_MODEL, num_labels = 19)
bert_model.to(device)
bert_tokenizer = AutoTokenizer.from_pretrained(BASE + BERT_BASE_TOKENIZER)

roberta_model = AutoModelForSequenceClassification.from_pretrained(BASE + ROBERTA_BASE_MODEL, return_dict=False, num_labels = 19)
roberta_model.to(device)
roberta_tokenizer = AutoTokenizer.from_pretrained(BASE + ROBERTA_BASE_TOKENIZER)

bert_large_model = AutoModelForSequenceClassification.from_pretrained(BASE + BERT_LARGE_MODEL, num_labels = 19)
bert_large_model.to(device)
bert_large_tokenizer = AutoTokenizer.from_pretrained(BASE + BERT_LARGE_TOKENIZER)

roberta_large_model = AutoModelForSequenceClassification.from_pretrained(BASE + ROBERTA_LARGE_MODEL, return_dict=False, num_labels = 19)
roberta_large_model.to(device)
roberta_large_tokenizer = AutoTokenizer.from_pretrained(BASE + ROBERTA_LARGE_TOKENIZER)

deberta_model = AutoModelForSequenceClassification.from_pretrained(BASE + DEBERTA_MODEL, return_dict=False, num_labels = 19)
deberta_model.to(device)
deberta_tokenizer = AutoTokenizer.from_pretrained(BASE + DEBERTA_TOKENIZER)

deberta_large_model = AutoModelForSequenceClassification.from_pretrained(BASE + DEBERTA_LARGE_MODEL, return_dict=False, num_labels = 19)
deberta_large_model.to(device)
deberta_large_tokenizer = AutoTokenizer.from_pretrained(BASE + DEBERTA_LARGE_TOKENIZER)

val_dataset = pd.read_csv('/home/users/ddimer/subfolder/files/df_test.csv')
val_dataset['label']

val_dataset['label'] = val_dataset['label'].apply(lambda x: ast.literal_eval(x))

labels = [label for label in TARGET_LIST if label not in ["Text-ID", "Sentence-ID"]]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

# predict for each row of val_dataset
def evaluate(model, tokenizer, name, pred_dict, pred_list): 
    acertos = 0
    erros = 0
    with torch.no_grad():
        for _ , row in val_dataset.iterrows():
            text = row.text
            true_labels = [id2label[idx] for idx, label in enumerate(row.label) if label == 1.0]
            encoding = tokenizer(text, return_tensors="pt")
            encoding = {k: v.to(model.device) for k,v in encoding.items()}

            outputs = model(**encoding,return_dict=False)
            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(outputs[0])
            pred_dict[text] = probs.cpu().numpy().tolist()[0]
            # val_dataset.loc[idx, name] = probs.cpu().numpy()

            predictions = np.zeros(probs.shape)
            predictions[np.where(probs.cpu() >= 0.5)] = 1
            pred_list[text] = (predictions[0].tolist())
            #pint(predictions)

            # turn predicted id's into actual label names
            predicted_labels = [id2label[idx] for idx, label in enumerate(predictions[0]) if label == 1.0]
            if predicted_labels == true_labels:
                acertos += 1
            else:
                erros +=1
        print(f"accuracy for {name}: {acertos/(acertos+erros)}")
            
        
bert_base_dict = {}
bert_base_preds = {}
evaluate(bert_model, bert_tokenizer, 'bert-base-uncased', bert_base_dict, bert_base_preds)

bert_large_dict = {}
bert_large_preds = {}
evaluate(bert_large_model, bert_large_tokenizer, 'bert-large-uncased', bert_large_dict, bert_large_preds)

roberta_base_dict = {}
roberta_base_preds = {}
evaluate(roberta_model, roberta_tokenizer, 'roberta-base', roberta_base_dict, roberta_base_preds)

roberta_large_base_dict = {}
roberta_large_preds = {}
evaluate(roberta_large_model, roberta_large_tokenizer, 'roberta-large', roberta_large_base_dict, roberta_large_preds)

deberta_dict = {}
deberta_preds = {}
evaluate(deberta_model, deberta_tokenizer, 'deberta-base', deberta_dict, deberta_preds)

deberta_large_dict = {}
deberta_large_preds = {}
evaluate(deberta_large_model, deberta_large_tokenizer, 'deberta-large', deberta_large_dict, deberta_large_preds)

val_dataset['bert-base-uncased-probs'] = val_dataset['text'].map(lambda x: list(bert_base_dict[x]))
val_dataset['bert-large-probs'] = val_dataset['text'].map(lambda x: list(bert_large_dict[x]))
val_dataset['roberta-base-probs'] = val_dataset['text'].map(lambda x: list(roberta_base_dict[x]))
val_dataset['roberta-large-probs'] = val_dataset['text'].map(lambda x: list(roberta_large_base_dict[x]))
val_dataset['deberta-base-probs'] = val_dataset['text'].map(lambda x: list(deberta_dict[x]))
val_dataset['deberta-large-probs'] = val_dataset['text'].map(lambda x: list(deberta_large_dict[x]))

val_dataset['bert-base-uncased-preds'] = val_dataset['text'].map(lambda x: bert_base_preds[x])
val_dataset['bert-large-preds'] = val_dataset['text'].map(lambda x: bert_large_preds[x])
val_dataset['roberta-base-preds'] = val_dataset['text'].map(lambda x: roberta_base_preds[x])
val_dataset['roberta-large-preds'] = val_dataset['text'].map(lambda x: roberta_large_preds[x])
val_dataset['deberta-base-preds'] = val_dataset['text'].map(lambda x: deberta_preds[x])
val_dataset['deberta-large-preds'] = val_dataset['text'].map(lambda x: deberta_large_preds[x])

val_dataset.to_csv('final_preds.csv')

print('F1 scores')
f1_scores_weighted = {}
f1_scores_macro = {}
f1_scores_micro = {}
f1_scores_none = {}

f1_scores_weighted['bert-base-uncased'] = f1_score(val_dataset.label.tolist(), val_dataset['bert-base-uncased-preds'].tolist(), average="weighted", zero_division=0)
f1_scores_weighted['bert-large-uncased'] = f1_score(val_dataset.label.tolist(), val_dataset['bert-large-preds'].tolist(), average="weighted", zero_division=0)
f1_scores_weighted['roberta-base'] = f1_score(val_dataset.label.tolist(), val_dataset['roberta-base-preds'].tolist(), average="weighted", zero_division=0)
f1_scores_weighted['roberta-large'] = f1_score(val_dataset.label.tolist(), val_dataset['roberta-large-preds'].tolist(), average="weighted", zero_division=0)
f1_scores_weighted['deberta-base'] = f1_score(val_dataset.label.tolist(), val_dataset['deberta-base-preds'].tolist(), average="weighted", zero_division=0)
f1_scores_weighted['deberta-large'] = f1_score(val_dataset.label.tolist(), val_dataset['deberta-large-preds'].tolist(), average="weighted", zero_division=0)
print("F1 scores weighted")
print(f1_scores_weighted)

f1_scores_macro['bert-base-uncased'] = f1_score(val_dataset.label.tolist(), val_dataset['bert-base-uncased-preds'].tolist(), average="macro", zero_division=0)
f1_scores_macro['bert-large-uncased'] = f1_score(val_dataset.label.tolist(), val_dataset['bert-large-preds'].tolist(), average="macro", zero_division=0)
f1_scores_macro['roberta-base'] = f1_score(val_dataset.label.tolist(), val_dataset['roberta-base-preds'].tolist(), average="macro", zero_division=0)
f1_scores_macro['roberta-large'] = f1_score(val_dataset.label.tolist(), val_dataset['roberta-large-preds'].tolist(), average="macro", zero_division=0)
f1_scores_macro['deberta-base'] = f1_score(val_dataset.label.tolist(), val_dataset['deberta-base-preds'].tolist(), average="macro", zero_division=0)
f1_scores_macro['deberta-large'] = f1_score(val_dataset.label.tolist(), val_dataset['deberta-large-preds'].tolist(), average="macro", zero_division=0)
print("F1 scores macro")
print(f1_scores_macro)

f1_scores_micro['bert-base-uncased'] = f1_score(val_dataset.label.tolist(), val_dataset['bert-base-uncased-preds'].tolist(), average="micro", zero_division=0)
f1_scores_micro['bert-large-uncased'] = f1_score(val_dataset.label.tolist(), val_dataset['bert-large-preds'].tolist(), average="micro", zero_division=0)
f1_scores_micro['roberta-base'] = f1_score(val_dataset.label.tolist(), val_dataset['roberta-base-preds'].tolist(), average="micro", zero_division=0)
f1_scores_micro['roberta-large'] = f1_score(val_dataset.label.tolist(), val_dataset['roberta-large-preds'].tolist(), average="micro", zero_division=0)
f1_scores_micro['deberta-base'] = f1_score(val_dataset.label.tolist(), val_dataset['deberta-base-preds'].tolist(), average="micro", zero_division=0)
f1_scores_micro['deberta-large'] = f1_score(val_dataset.label.tolist(), val_dataset['deberta-large-preds'].tolist(), average="micro", zero_division=0)
print("F1 scores micro")
print(f1_scores_micro)

f1_scores_none['bert-base-uncased'] = f1_score(val_dataset.label.tolist(), val_dataset['bert-base-uncased-preds'].tolist(), average=None, zero_division=0)
f1_scores_none['bert-large-uncased'] = f1_score(val_dataset.label.tolist(), val_dataset['bert-large-preds'].tolist(), average=None, zero_division=0)
f1_scores_none['roberta-base'] = f1_score(val_dataset.label.tolist(), val_dataset['roberta-base-preds'].tolist(), average=None, zero_division=0)
f1_scores_none['roberta-large'] = f1_score(val_dataset.label.tolist(), val_dataset['roberta-large-preds'].tolist(), average=None, zero_division=0)
f1_scores_none['deberta-base'] = f1_score(val_dataset.label.tolist(), val_dataset['deberta-base-preds'].tolist(), average=None, zero_division=0)
f1_scores_none['deberta-large'] = f1_score(val_dataset.label.tolist(), val_dataset['deberta-large-preds'].tolist(), average=None, zero_division=0)
print("F1 scores none")
for key, value in f1_scores_none.items():
    print(key)
    print(value)
    print('-------')


