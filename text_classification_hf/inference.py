from sklearn import metrics
from glob import glob
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
from dataset import MyDataset_inference
from tqdm import tqdm
import argparse
import torch
from utils import label2int

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--json_path', type=str, default='./datas/news_test.jsonl')
    p.add_argument("--output_csv", type=str, default='./datas/out_metrics.csv')
    p.add_argument('--pretrained_model_pathes', type=str, default='./models_zoo/')
    args = p.parse_args()

    return args

def make_metrcis(model_path: str, dataset: torch.utils.data.Dataset):
    
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, 'tokenizer'))
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(model_path, 'model_weights'), num_labels=5)
    pipe = pipeline("text-classification",
                model=model,
                tokenizer=tokenizer,
                device=0)
    
    out_ls = []
    for out in tqdm(pipe(dataset, batch_size=32), total=len(dataset)):              
        out_ls.append(out)
    return [int(out['label'][-1]) for out in out_ls]

def main(args):

    data = pd.read_json('./datas/news_test.jsonl', lines=True)
    dataset = MyDataset_inference(data)
    labels = data['completion'].map(lambda x: label2int(x)).to_list()
    
    model_pathes = glob(os.path.join(args.pretrained_model_pathes, '*'))
    
    out_ls = []
    for model_path in model_pathes:
        print("Inference Start, ", model_path)
        out = make_metrcis(model_path, dataset)
        out_ls.append([os.path.basename(model_path), metrics.accuracy_score(labels, out)])
    
    df = pd.DataFrame(out_ls, columns=['model', 'accuracy'])
    df.to_csv(args.output_csv, index=False)
    
if __name__ == '__main__':
    args = define_argparser()
    main(args)