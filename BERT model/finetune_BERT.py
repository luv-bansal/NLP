import torch
import torch.nn as nn
import transformers
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm

class BertDataset(Dataset):
    def __init__(self,root_dir, train_csv,tokenizer,max_length):
        super(BertDataset, self).__init__()
        self.root_dir=root_dir
        self.train_csv=pd.read_csv(root_dir + "\\" + train_csv)
        self.tokenizer=tokenizer
        self.target=self.train_csv['target']
        self.max_length=max_length
        
    def __len__(self):
        return len(self.train_csv)
    
    def __getitem__(self, index):
        
        text1 = self.train_csv.loc[index,'text']
        
        inputs = self.tokenizer.encode_plus(
            text1 ,
            None,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
        )
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        padding_length = self.max_length - len(ids)
        print(padding_length)
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'target': torch.tensor(self.train_csv.loc[index, 'target'], dtype=torch.long)
            }

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.drop_out= nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)
        
    def forward(self,ids,mask,token_type_ids):
        _,o2= self.bert_model(ids,attention_mask=mask,token_type_ids=token_type_ids)
        
        drop=self.drop_out(o2)
        out= self.out(drop)
        
        return out


loss_fn = nn.BCEWithLogitsLoss()



def finetune(epochs,dataloader,model,loss_fn,optimizer):
    model.train()
    for  epoch in range(epochs):
        print(epoch)
        
        loop=tqdm(enumerate(dataloader),leave=False,total=len(dataloader))
        for batch, dl in loop:
            print(batch)
            ids=dl['ids']
            token_type_ids=dl['token_type_ids']
            mask= dl['mask']
            label=dl['target']
            label = label.unsqueeze(1)
            output=model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids)
            label = label.type_as(output)

            loss=loss_fn(output,label)
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            
            # Show progress while training
            loop.set_description(f'Epoch={epoch}/{epochs}')
            loop.set_postfix(loss=loss.item(),acc=torch.rand(1).item())

    return model


tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
dataset= BertDataset('.','train.csv',tokenizer,
                     max_length=100)

dataloader=DataLoader(dataset=dataset)
model=BERT()
#Initialize Optimizer
optimizer= optim.Adam(model.parameters())
model=finetune(1, dataloader, model, loss_fn, optimizer)



def check_accuracy(loader, model):

    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        
        loop=tqdm(enumerate(loader),leave=False,total=len(loader))
        for x, dl in loop:
            
            ids=dl['ids']
            token_type_ids=dl['token_type_ids']
            mask= dl['mask']
            label=dl['target']

        
            output=model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids)
            _, predictions = output.max(1)
            num_correct += (predictions == label).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}') 
    
    model.train()
