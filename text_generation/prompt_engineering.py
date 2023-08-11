import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from custom_data import textDataset

def define_argparser():
    """Function to define the command line arguments
    
    Returns:
        argparse.Namespace: Command line arguments
    """
    p = argparse.ArgumentParser()
    p.add_argument('--tsv_path', type=str, default='./data/test.tsv')
    p.add_argument('--train_tsv_path', type=str, default='./data/train.tsv')
    p.add_argument('--out_path', type=str, default='./data/out/KoGPT6B-ryan1.5b-float16.tsv')
    p.add_argument('--max_new_tokens', type=int, default=60)

    config = p.parse_args()

    return config

def main(config):
    np.random.seed(1004)
    
    tokenizer = AutoTokenizer.from_pretrained(
        'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
        bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
    )
    model = AutoModelForCausalLM.from_pretrained(
        'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
        pad_token_id=tokenizer.eos_token_id,
        torch_dtype='auto', low_cpu_mem_usage=True
    ).to(device='cuda', non_blocking=True)
    _ = model.eval()

    df = pd.read_csv(config.tsv_path, sep='\t', names=['input_seq', 'output_seq'])
    dataset = textDataset(config.train_tsv_path)
    
    init_prompt = ""
    for i in np.random.randint(0, len(dataset), 5):
        init_prompt = init_prompt + tokenizer.bos_token +  "Q: " + dataset[i].replace("\t", "A: ")  + tokenizer.eos_token
    
    args = {
        "num_beams" : 3,
        "num_return_sequences" : 1,
        "repetition_penalty" : 1.2,
        "pad_token_id" :tokenizer.pad_token_id,
        "eos_token_id":tokenizer.eos_token_id,
        "bos_token_id":tokenizer.bos_token_id,
        # "temperature" : 0.8,
        "max_new_tokens": config.max_new_tokens,
        "do_sample" : True
    }
    out_ls = []
    with torch.no_grad():
        for i in tqdm(range(len(df))):
            prompt = init_prompt +  tokenizer.bos_token + "Q: " + df['input_seq'].iloc[i] +  "A: "
            tokens = tokenizer.encode(prompt, return_tensors='pt').to(device='cuda', non_blocking=True)
            gen_tokens = model.generate(tokens, **args)
            out_ls.append(tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0].split("A:")[-1])

    with open(config.out_path, 'w', encoding='UTF-8') as f:
        for i in range(len(df)):
            content = "{}\t{}\t{}".format(
                df['input_seq'].iloc[i], df['output_seq'].iloc[i], out_ls[i]
            )
            content = content.replace('\n', "") + '\n'
            f.write(content)

if __name__ == '__main__':
    config = define_argparser()
    main(config)