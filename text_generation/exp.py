# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"]= "1"

from transformers import Trainer, TrainingArguments
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_int8_training
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup, pipeline
from custom_data import textDataset, dataCollator, textDataset_inference

# %%
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

train_dataset = textDataset('./data/train.tsv')
val_dataset = textDataset('./data/val.tsv')
collator = dataCollator(tokenizer=tokenizer, 
                        max_length=256, 
                        with_text=False, 
                        model_type='causal_lm')

print(
    '|train| =', len(train_dataset),
    '|valid| =', len(val_dataset),
)

total_batch_size = 64 * torch.cuda.device_count()
n_total_iterations = int(len(train_dataset) / (total_batch_size * 2) * 5)
n_warmup_steps = int(n_total_iterations * 0.2)

print(
    '#total_iters =', n_total_iterations,
    '#warmup_iters =', n_warmup_steps,
)

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=1e-4
)
scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer=optimizer,
    num_training_steps=n_total_iterations,
    num_warmup_steps=n_warmup_steps,
    num_cycles=1
)
#
training_args = TrainingArguments(
    output_dir='./models_zoo/kakaobrain_kogpt_3',
    num_train_epochs=5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_steps=n_warmup_steps,
    fp16=True,
    evaluation_strategy='steps',
    logging_steps=n_total_iterations // 5,
    save_strategy ='steps',
    save_steps=n_total_iterations // 5 ,
    gradient_accumulation_steps=2,
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
# %%
torch.save(model.state_dict(), './models_zoo/kakaobrain_kogpt_3/state_dict.pt')
trainer.model.save_pretrained(os.path.join('models_zoo/kakaobrain_kogpt_3', 'model_weights'), from_pt=True)
tokenizer.save_pretrained(os.path.join('models_zoo/kakaobrain_kogpt_3', 'tokenizer'))
# %%
trainer.model.save_pretrained('./temp/weights')
# %%
model.push_to_hub(training_args.output_dir, use_auth_token=True)
# %%
peft_model_id = training_args.output_dir
config = PeftConfig.from_pretrained(peft_model_id)
# %%
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, revision='KoGPT6B-ryan1.5b-float16', 
    return_dict=True, load_in_8bit=True, device_map="auto"
)
# %%
model.cuda()
model.load_state_dict(torch.load('./models_zoo/kakaobrain_kogpt_3/state_dict.pt'), strict=False)
# %%
tokenizer = AutoTokenizer.from_pretrained('./models_zoo/kakaobrain_kogpt_3/tokenizer/', padding_side='left') 
# %%
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
dataset = textDataset_inference('./data/test.tsv', eos_token='[SEP]')
# %%
from tqdm import tqdm
args = {
    "batch_size" : 8,
    "num_beams" : 3,
    "num_return_sequences" : 1,
    "max_new_tokens" : 128,
    "repetition_penalty" : 1.5,
    "pad_token_id" :tokenizer.pad_token_id,
    "eos_token_id":tokenizer.eos_token_id,
    "bos_token_id":tokenizer.bos_token_id,
}
out_ls = []
with torch.cuda.amp.autocast():
    for out in tqdm(generator(dataset,**args),total=len(dataset)):
        out_ls.append(out[0]['generated_text'])
# %%