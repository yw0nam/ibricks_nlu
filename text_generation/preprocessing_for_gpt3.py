import pandas as pd
import argparse, re
from sklearn.model_selection import train_test_split

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--json_path', type=str, default='./datas/qa_train.jsonl')
    p.add_argument('--val_split', type=int, default=1)
    config = p.parse_args()

    return config
    
def main(config):
    data = pd.read_json(config.json_path, lines=True)
    input_seq = data['prompt'].map(lambda x: re.sub("\n위 문의에 대한 답변 생성해줘\n\n###\n\n", '', x).strip())
    output_seq = data['completion'].map(lambda x: x[:-4].strip())
    seq = input_seq + "[EOS]" + output_seq
    
    if config.val_split == 1:
        train, val = train_test_split(seq, test_size=0.1, random_state=1004)
        train.to_csv('./data/train.txt', sep='\t', header=None, index=False)
        val.to_csv('./data/val.txt', sep='\t', header=None, index=False)
    else:
        seq.to_csv('./data/test.txt', sep='\t', header=None, index=False)
        
if __name__ == '__main__':
    config = define_argparser()
    main(config)