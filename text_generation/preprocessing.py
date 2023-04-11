import pandas as pd
import argparse, re
from sklearn.model_selection import train_test_split

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--json_path', type=str, default='./data/qa_train.jsonl')
    p.add_argument('--val_split', type=int, default=1)
    config = p.parse_args()

    return config
    
def main(config):
    data = pd.read_json(config.json_path, lines=True)
    data['prompt'] = data['prompt'].map(lambda x: re.sub("\n위 문의에 대한 답변 생성해줘\n\n###\n\n", '', x).strip())
    data['completion'] = data['completion'].map(lambda x: re.sub("###", '', x).strip())
    
    if config.val_split == 1:
        train, val = train_test_split(data, test_size=0.1, random_state=1004)
        train.to_csv('./data/train.tsv', sep='\t', header=None, index=False)
        val.to_csv('./data/val.tsv', sep='\t', header=None, index=False)
    else:
        data.to_csv('./data/test.tsv', sep='\t', header=None, index=False)
        
if __name__ == '__main__':
    config = define_argparser()
    main(config)