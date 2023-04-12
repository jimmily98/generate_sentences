import transformers

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          PreTrainedModel, DistilBertModel, DistilBertForSequenceClassification,
                          TrainingArguments, Trainer)
from transformers.modeling_outputs import SequenceClassifierOutput
import matplotlib.pyplot as plt

## Parameters

model_path = "distilbert-base-uncased"
test_dataset = pd.read_excel(r"C:\Users\malos\OneDrive\Documents\2A\PSC\Modèles\Assistant_virtuel\Classification\BdD1.xlsx")

##

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

def evaluate(model_path,test_dataset_path):

    df = test_dataset

    labels = df.columns[2:].tolist()

    id2label = {i : labels[i] for i in range(len(labels))}
    label2id = {labels[i] : i for i in range(len(labels))}

    df["labels"] = df[labels].values.tolist()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    true_labels = df["labels"].values.tolist()

    model = DistilBertForMultilabelSequenceClassification.from_pretrained(model_path,num_labels = len(labels))

    model.config.id2label = id2label
    model.config.label2id = label2id

    encodings = tokenizer(df["text"].values.tolist(), truncation=True)

    inputs = [ torch.tensor([x]) for x in encodings["input_ids"] ]
    logits = [ model(x)[:2] for x in inputs ]
    outputs = [ torch.nn.Sigmoid()(x[0])[0].tolist() for x in logits]
    predicted_labels_idx = [ np.argmax(x) for x in outputs]

    loss = [ loss_l2(true_labels[i],outputs[i]) for i in range(len(true_labels))]

    mean_loss = np.mean(loss)

    mean_correct = 1-np.mean([loss_01(true_labels[i],outputs[i]) for i in range(len(true_labels))])

    return mean_loss,mean_correct

def loss_l2(l1,l2): #L2-loss : L2 norm of the difference between predicted and true
    assert (len(l1)==len(l2)) , "lists are of different lengths"
    diff = np.array(l1) - np.array(l2)
    return np.sum(diff*diff)

def loss_01(l1,l2): #01-loss : bool value of predicted!=true
    result = select(l2)
    return (loss_l2(l1,result) > 0.5)

def select(l): #Choice function : what is our final answer based on the values returned by the model
    idx = np.argmax(l)
    return [1*(i==idx) for i in range(len(l))]