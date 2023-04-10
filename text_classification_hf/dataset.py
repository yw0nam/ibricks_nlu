import torch
import re
from torch.utils.data import Dataset
from utils import label2int
from transformers.data.data_collator import DataCollatorForTokenClassification
class MyDataset(Dataset):

    def __init__(self, csv, tokenizer):
        csv['prompt'] = csv['prompt'].map(lambda x: re.sub('\n\n###\n\n', '', x))
        csv['completion'] = csv['completion'].map(lambda x: label2int(x))
        
        self.encodings = tokenizer(csv['prompt'].to_list(), truncation=True, padding=True)
        self.labels = csv['completion'].to_list()
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
class MyDataset_inference(Dataset):
    def __init__(self, csv):
        self.csv = csv
        self.csv['prompt'] = csv['prompt'].map(lambda x: re.sub('\n\n###\n\n', '', x))
    def __len__(self):                                                              
        return len(self.csv)

    def __getitem__(self, i):                                                       
        return self.csv['prompt'].iloc[i]