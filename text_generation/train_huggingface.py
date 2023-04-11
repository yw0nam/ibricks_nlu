import torch
from transformers import AutoTokenizer, AutoModelForCausalLM#, TextDataset, DataCollatorForLanguageModeling
from custom_data import textDataset, dataCollator
from transformers import Trainer, TrainingArguments
import torch
import os
import argparse

def define_argparser():
    """Function to define the command line arguments
    
    Returns:
        argparse.Namespace: Command line arguments
    """
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, default='./data/')
    p.add_argument('--gradient_accumulation_steps', type=int, default=1)
    p.add_argument('--batch_size_per_device', type=int, default=128)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--warmup_ratio', type=float, default=.25)
    p.add_argument('--max_length', type=int, default=512)
    p.add_argument('--model_save_path', type=str, default='./models_zoo/skt_kogpt2-base-v2/')
    # p.add_argument('--model_address', type=str, default='klue/bert-base')

    config = p.parse_args()

    return config
    
def main(config):
    """Main function to train the language model

    Args:
        config (argparse.Namespace): Command line arguments
    """
    tokenizer = AutoTokenizer.from_pretrained(
        "skt/kogpt2-base-v2",
        bos_token='</s>', eos_token='</s>', unk_token='<unk>',
        pad_token='<pad>', mask_token='<mask>'
    )
    
    model = AutoModelForCausalLM.from_pretrained('skt/kogpt2-base-v2')
    
    
    # train_dataset = TextDataset(
    #     tokenizer=tokenizer,
    #     file_path=os.path.join(config.data_root,"train.txt"),
    #     block_size=128
    # )
    # val_dataset = TextDataset(
    #     tokenizer=tokenizer,
    #     file_path=os.path.join(config.data_root,"val.txt"),
    #     block_size=128
    # )
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    train_dataset = textDataset(os.path.join(config.data_root,"train.tsv"))
    val_dataset = textDataset(os.path.join(config.data_root,"val.tsv"))
    collator = dataCollator(tokenizer=tokenizer, 
                            max_length=128, 
                            with_text=False, 
                            model_type='causal_lm')
    
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
        evaluation_strategy='steps',
        logging_steps=10,
        save_strategy ='steps',
        save_steps=10,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        load_best_model_at_end=True,
        prediction_loss_only=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator
    )

    trainer.train()
    
    trainer.model.save_pretrained(os.path.join(config.model_save_path, 'model_weights'))
    tokenizer.save_pretrained(os.path.join(config.model_save_path, 'tokenizer'))

if __name__ == '__main__':
    config = define_argparser()
    main(config)