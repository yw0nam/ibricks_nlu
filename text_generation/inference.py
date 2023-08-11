import argparse, os
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from custom_data import textDataset_inference

def define_argparser():
    """Function to define the command line arguments
    
    Returns:
        argparse.Namespace: Command line arguments
    """
    p = argparse.ArgumentParser()
    p.add_argument('--tsv_path', type=str, default='./data/test.tsv')
    p.add_argument('--out_path', type=str, default='./data/out/skt_kogpt2_out.tsv')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--max_length', type=int, default=256)
    p.add_argument('--model_save_path', type=str, default='./models_zoo/skt_kogpt2-base-v2/')

    config = p.parse_args()

    return config

def main(config):
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(config.model_save_path, "tokenizer"), padding_side='left'
    )
    model = AutoModelForCausalLM.from_pretrained(os.path.join(config.model_save_path, "model_weights"))

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    dataset = textDataset_inference(config.tsv_path)

    df = pd.read_csv(config.tsv_path, sep='\t', names=['input_seq', 'output_seq'])
    
    args = {
        "batch_size" : config.batch_size,
        "num_beams" : 3,
        "num_return_sequences" : 1,
        "max_new_tokens" : 128,
        "repetition_penalty" : 1.5,
        "pad_token_id" :tokenizer.pad_token_id,
        "eos_token_id":tokenizer.eos_token_id,
        "bos_token_id":tokenizer.bos_token_id,
    }
    out_ls = []
    for out in tqdm(generator(dataset,**args),total=len(dataset)):
        out_ls.append(out[0]['generated_text'])

    out_seq = [out.split('</s>')[1] for out in out_ls]
    with open(config.out_path, 'w', encoding='UTF-8') as f:
        for i in range(len(out_seq)):
            content = "{}\t{}\t{}\n".format(
                df['input_seq'].iloc[i], df['output_seq'].iloc[i], out_seq[i] 
            )
            content = content.replace('\n', "") + '\n'
            f.write(content)

if __name__ == '__main__':
    config = define_argparser()
    main(config)