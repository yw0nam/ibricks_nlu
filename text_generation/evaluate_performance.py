import argparse
import evaluate
import pandas  as pd
def define_argparser():
    """Function to define the command line arguments
    
    Returns:
        argparse.Namespace: Command line arguments
    """
    p = argparse.ArgumentParser()
    p.add_argument('--tsv_path', type=str, default='./data/input.tsv')
    p.add_argument('--out_path', type=str, default='./data/out/metrics.txt')

    config = p.parse_args()

    return config

def main(config):
    with open(config.tsv_path, 'r') as f:
        lines = f.readlines()
    
    # query = [line.split('\t')[0] for line in lines]
    ref = [line.split('\t')[1] for line in lines]
    gen = [line.split('\t')[2] for line in lines]
    
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    rouge_results = rouge.compute(predictions=gen,
                            references=ref)
    bleu_results = bleu.compute(predictions=gen, references=ref)
    rouge_results['bleu'] = bleu_results['bleu']

    with open(config.out_path, 'w', encoding='UTF-8') as f:
        for key in rouge_results.keys():
            f.write("{}\t{}\n".format(key, rouge_results[key]))
        

if __name__ == '__main__':
    config = define_argparser()
    main(config)