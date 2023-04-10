import pandas as pd
import argparse
from dataset import *
from transformers import Trainer
from transformers import TrainingArguments
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
import os, re
def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--json_path', type=str, default='./datas/news_train.jsonl')
    p.add_argument('--gradient_accumulation_steps', type=int, default=1)
    p.add_argument('--valid_ratio', type=float, default=.1)
    p.add_argument('--batch_size_per_device', type=int, default=512)
    p.add_argument('--n_epochs', type=int, default=5)
    p.add_argument('--warmup_ratio', type=float, default=.25)
    p.add_argument('--max_length', type=int, default=512)
    p.add_argument('--model_save_path', type=str, default='./models_zoo/klue_bert_base/')
    p.add_argument('--model_address', type=str, default='klue/bert-base')
    p.add_argument('--label_num', type=int, default=5)

    config = p.parse_args()

    return config
    
def main(config):
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_address)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_address, num_labels=config.label_num)

    data = pd.read_json(config.json_path, lines=True)
    train, val = train_test_split(data, test_size=0.1, random_state=1004, stratify=data['completion'])
    train_dataset = MyDataset(train, tokenizer)
    val_dataset = MyDataset(val, tokenizer)
    
    print(
        '|train| =', len(train_dataset),
        '|valid| =', len(val_dataset),
    )

    total_batch_size = config.batch_size_per_device * torch.cuda.device_count()
    n_total_iterations = int(len(train_dataset) / total_batch_size * config.n_epochs)
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    
    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )
    
    training_args = TrainingArguments(
        output_dir=os.path.join(config.model_save_path, 'checkpoints'),
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.batch_size_per_device,
        per_device_eval_batch_size=config.batch_size_per_device,
        warmup_steps=n_warmup_steps,
        fp16=True,
        evaluation_strategy='epoch',
        # logging_steps=n_total_iterations // 100,
        logging_steps=10,
        save_strategy ='epoch',
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        load_best_model_at_end=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    
    trainer.model.save_pretrained(os.path.join(config.model_save_path, 'model_weights'))
    tokenizer.save_pretrained(os.path.join(config.model_save_path, 'tokenizer'))

if __name__ == '__main__':
    config = define_argparser()
    main(config)