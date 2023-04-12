#from github import Github
import requests
import pandas as pd
import random as rd
import string
import openpyxl

import transformers

import torch
import numpy as np
from pathlib import Path
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          PreTrainedModel, DistilBertModel, DistilBertForSequenceClassification,
                          TrainingArguments, Trainer)
from transformers.modeling_outputs import SequenceClassifierOutput
import huggingface_hub

##INPUT


"""Necessary arguments"""
model_num = 1 #number of the model you want to train (1-8)
model_specs = "model1_size_10000_portion_0.5"
errors_p = 0.05 #proportion of inputs with spelling mistakes
random_p = 0 #proportion of random inputs

"""Extra arguments"""
model_name = "distilbert-base-uncased" #if we want to further finetune a model
push_to_hub = True

##NOTHING NEEDS TO BE MODIFIED BELOW THIS LINE

#huggingface_hub.login()

##FUNCTIONS

def get_data(model_num,model_specs):
    # Github username
    username = "jimmily98"
    # pygithub object
    #g = Github("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")
    # get that user by username
    #user = g.get_user(username)
    # get github repo
    #repo = user.get_repo("generate_sentences")
    #get file list
    #files = repo.get_contents(f"model_{model_num}")
    #beginning of raw url to each file
    base_url = f"https://raw.githubusercontent.com/jimmily98/generate_sentences/main/model_{model_num}/"

    #list of urls
    #urls = [x.path for x in files]

    #get list of labels from labels.txt
    labels_url = base_url + "labels.txt"

    labels = requests.get(labels_url).text.split("\n")[:-1]

    #urls = urls[1:]

    #organize datasets into train-test pairs
    """data_pairs = []
    for x in urls:
        if x[8:15] == "testing":
            data_pairs.append((x[:8]+"training"+x[15:],x))"""
            
    train_url = base_url + "training_" + model_specs + ".xlsx"
    test_url = base_url + "testing_" + model_specs + ".xlsx"
    print(train_url)
    print(test_url)
    
    data_pairs = [(base_url + "training_" + model_specs + ".xlsx" , base_url + "testing_" + model_specs + ".xlsx")]

    #get actual datasets
    trainings_dfs = [pd.read_excel(requests.get(d[0]).content ,engine="openpyxl") for d in data_pairs]
    testing_dfs = [pd.read_excel(requests.get(d[1]).content ,engine="openpyxl") for d in data_pairs]

    #get model names
    names = [d[1][84:-5] for d in data_pairs]

    #process data
    for df in trainings_dfs+testing_dfs:
        df["labels"] = [[1*(df["Coding"].values.tolist()[j]==i) for i in range(len(labels))] for j in range(df.shape[0])]

    print(f"Model {model_num} : labels = {labels}")
    print(f"Model name : {names[0]}")

    return trainings_dfs,testing_dfs,names,labels

def add_random_data(df,p):
    #TO DO
    return df

def convert_sentence(sentence):
    lower_bound = min(5, len(sentence)//5)
    number_of_errors = rd.randint(1, lower_bound)

    for i in range(number_of_errors):
        type_of_error = rd.randint(1, 3)
        #type 1: adding a new character
        if (type_of_error == 1):
            add_character = rd.choice(string.ascii_letters)
            position_of_error = rd.randint(0, len(sentence))
            if (position_of_error == 0):
                sentence = add_character + sentence
            else:
                sentence = sentence[:position_of_error-1] + add_character + sentence[position_of_error-1:]
        #type 2: remove a character
        if (type_of_error == 2):
            position_of_error = rd.randint(0, len(sentence)-1)
            sentence = sentence[:position_of_error] + sentence[position_of_error+1:]
        #type 3: change a character to another character
        if (type_of_error == 3):
            replace_character = rd.choice(string.ascii_letters)
            position_of_error = rd.randint(0, len(sentence)-1)
            sentence = sentence[:position_of_error] + replace_character + sentence[position_of_error+1:]
    #return an error sentence
    return sentence

def with_errors(df,p): #return new df with a proportion p of entries with spelling errors
    new_sentences = df["Sentences"].values.tolist()
    cond = np.random.random(df.shape[0]) < p
    for i in range(len(cond)):
        if cond[i]:
            new_sentences[i] = convert_sentence(new_sentences[i])
    edf = pd.DataFrame({"Sentences":new_sentences})
    edf["Coding"] = df["Coding"].values.tolist()
    edf["labels"] = df["labels"].values.tolist()
    return edf

class GoEmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class DistilBertForMultilabelSequenceClassification(DistilBertForSequenceClassification):
    def __init__(self, config):
      super().__init__(config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.distilbert(input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct1 = torch.nn.MSELoss(reduction = 'sum')
            loss_fct2 = torch.nn.Sigmoid()
            loss = loss_fct1(loss_fct2(logits.view(-1, self.num_labels)),
                            labels.float().view(-1, self.num_labels))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions)

def accuracy_thresh(y_pred, y_true, thresh=0.5, sigmoid=True):
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)
    if sigmoid:
      y_pred = y_pred.sigmoid()
    return ((y_pred>thresh)==y_true.bool()).float().mean().item()


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {'accuracy_thresh': accuracy_thresh(predictions, labels)}

def train(df_train,df_test,name,labels,model_name = "distilbert-base-uncased",push_to_hub = False): #train model on datasets - default : start with model without any finetuning, save to devicer

    #set up model and tokenizer
    id2label = {i : labels[i] for i in range(len(labels))}
    label2id = {labels[i] : i for i in range(len(labels))}
    model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    num_labels=len(labels)

    model = DistilBertForMultilabelSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.config.id2label = id2label
    model.config.label2id = label2id


    #encode and format data
    train_encodings = tokenizer(df_train["Sentences"].values.tolist(), truncation=True)
    test_encodings = tokenizer(df_test["Sentences"].values.tolist(), truncation=True)


    train_labels = df_train["labels"].values.tolist()
    test_labels = df_test["labels"].values.tolist()

    train_dataset = GoEmotionDataset(train_encodings, train_labels)
    test_dataset = GoEmotionDataset(test_encodings, test_labels)

    #set up trainer
    batch_size = 32
    logging_steps = len(train_dataset) // batch_size

    args = TrainingArguments(
        output_dir=name,
        overwrite_output_dir=True,
        push_to_hub=push_to_hub,
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=logging_steps
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer)

    #train and save model, either to HuggingFace hub, or on device (default)
    trainer.train()
    """if push_to_hub:
        trainer.push_to_hub(name,overwrite=True,use_auth_token="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")
    else:
        model.save_pretrained(name)"""
    model.save_pretrained(name)



##EXECUTE CODE - TRAINS MODEL ASSOCIATED WITH NUMBER AND SPECIFICATIONS

#get data
data = get_data(model_num,model_specs)
names = data[2]
labels = data[3]

for i in range(len(names)):
    #add random data and errors
    df_train = with_errors(add_random_data(data[0][i],random_p),errors_p)
    df_test = with_errors(add_random_data(data[1][i],random_p),errors_p)
    #train
    train(df_train,df_test,names[i],labels,model_name,push_to_hub)


