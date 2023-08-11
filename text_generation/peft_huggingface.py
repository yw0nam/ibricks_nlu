from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from custom_data import textDataset, dataCollator
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
import torch
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

def define_argparser():
    """Function to define the command line arguments
    
    Returns:
        argparse.Namespace: Command line arguments
    """
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, default='./data/')
    p.add_argument('--gradient_accumulation_steps', type=int, default=2)
    p.add_argument('--batch_size_per_device', type=int, default=64)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--n_epochs', type=int, default=5)
    p.add_argument('--warmup_ratio', type=float, default=.2)
    p.add_argument('--max_length', type=int, default=256)
    p.add_argument('--model_save_path', type=str, default='./models_zoo/kakaobrain_kogpt/')

    config = p.parse_args()

    return config

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def main(config):
    """Main function to train the language model

    Args:
        config (argparse.Namespace): Command line arguments
    """
    
    tokenizer = AutoTokenizer.from_pretrained(
        'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
        bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]', sep_token='[SEP]',
    )
    model = AutoModelForCausalLM.from_pretrained(
        'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
        pad_token_id=tokenizer.pad_token_id, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_8bit=True,
        device_map="auto",
    )
    
    target_modules = ['k_proj', 'v_proj', 'q_proj', 'out_proj', 'fc_in', 'fc_out']
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=target_modules, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = prepare_model_for_int8_training(
        model, output_embedding_layer_name='lm_head',
        layer_norm_names=['ln_1','ln_f']
    )
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    
    model.gradient_checkpointing_enable()
    
    train_dataset = textDataset(os.path.join(config.data_root,"train.tsv"))
    val_dataset = textDataset(os.path.join(config.data_root,"val.tsv"))
    collator = dataCollator(tokenizer=tokenizer, 
                            max_length=config.max_length, 
                            with_text=False, 
                            model_type='causal_lm')
    
    print(
        '|train| =', len(train_dataset),
        '|valid| =', len(val_dataset),
    )

    total_batch_size = config.batch_size_per_device * torch.cuda.device_count()
    n_total_iterations = int(len(train_dataset) / (total_batch_size * config.gradient_accumulation_steps) * config.n_epochs)
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    
    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.lr
    )
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer=optimizer,
        num_training_steps=n_total_iterations,
        num_warmup_steps=n_warmup_steps,
        num_cycles=1
    )
    
    training_args = TrainingArguments(
        output_dir=os.path.join(config.model_save_path, 'checkpoints'),
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.batch_size_per_device,
        per_device_eval_batch_size=config.batch_size_per_device,
        warmup_steps=n_warmup_steps,
        fp16=True,
        evaluation_strategy='steps',
        logging_steps=n_total_iterations // 5,
        save_strategy ='steps',
        save_steps=n_total_iterations // 5 ,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        # load_best_model_at_end=True,
        # prediction_loss_only=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        optimizers=[optimizer, scheduler]
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
    # model.push_to_hub(training_args.output_dir, use_auth_token=True)
    
    torch.save(trainer.model.state_dict(), os.path.join(config.model_save_path, 'model_weights'))
    trainer.model.save_pretrained(os.path.join(config.model_save_path, 'model_weights'))
    tokenizer.save_pretrained(os.path.join(config.model_save_path, 'tokenizer'))
    
if __name__ == '__main__':
    config = define_argparser()
    main(config)