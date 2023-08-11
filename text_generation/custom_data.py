import torch
from torch.utils.data import Dataset
import pandas as pd

class MyDataset(Dataset):

    def __init__(self, tsv_path, tokenizer):
        df = pd.read_csv(tsv_path, sep='\t', names=['input_seq', 'output_seq'])
        self.encodings = tokenizer((df['input_seq']+ tokenizer.sep_token+df['output_seq']).to_list(), truncation=True, padding=True)
        
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids']
        return item
    
class textDataset(Dataset):

    def __init__(self, tsv_path):
        self.df = pd.read_csv(tsv_path, sep='\t', names=['input_seq', 'output_seq'])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        input_seq = str(self.df['input_seq'].iloc[idx])
        output_seq = str(self.df['output_seq'].iloc[idx])

        return input_seq + '\t' + output_seq

class textDataset_inference(Dataset):

    def __init__(self, tsv_path, eos_token='</s>'):
        self.df = pd.read_csv(tsv_path, sep='\t', names=['input_seq', 'output_seq'])
        self.eos_token = eos_token
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        input_seq = str(self.df['input_seq'].iloc[idx])

        return input_seq + self.eos_token
    
class dataCollator():

    def __init__(self, tokenizer, max_length, with_text=True, model_type='causal_lm'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_text = with_text
        self.model_type = model_type
        
    def __call__(self, samples):
        input_seq = [s.split('\t')[0] for s in samples]
        output_seq = [s.split('\t')[1] for s in samples]
        
        if self.model_type == 'seq2seq':
            input_encoding = self.tokenizer(
                input_seq,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length
            )
            
            output_encoding = self.tokenizer(
                output_seq,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length
            )
            return_value = {
                'input_ids': input_encoding['input_ids'],
                'attention_mask': input_encoding['attention_mask'],
                'labels': output_encoding['input_ids'],
            }
        elif self.model_type == 'causal_lm':
            seq = [input_seq+self.tokenizer.sep_token+output_seq for input_seq, output_seq in zip(input_seq,output_seq)]
            encoding = self.tokenizer(
                seq,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length
            )
            return_value = {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'labels': encoding['input_ids']
            }
        if self.with_text:
            return_value['input_seq'] = input_seq
            return_value['output_seq'] = output_seq
            
        return return_value