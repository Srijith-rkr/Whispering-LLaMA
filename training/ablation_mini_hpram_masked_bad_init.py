import os
import sys
import time
import wandb
from pathlib import Path
import shutil
import argparse

import lightning as L
import numpy as np
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve() # does not work as jupyter notebook 
sys.path.append(str(wd))

import whisper_openAI.whisper as whisper
from lit_llama.ablation_mini_bad_init import LLaMA, LLaMAConfig, mark_only_adapter_as_trainable, adapter_state_from_state_dict
from lit_llama.tokenizer import Tokenizer
#from scripts.prepare_alpaca import generate_prompt
from lightning.fabric.strategies import DeepSpeedStrategy
from generate.generate_for_WL import generate

#cli setup 
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-2,help='learning rate for the model (default: 1e-2)')
parser.add_argument('--d', type=int, default=1,help='lNo of GPUs (default: 1)')
parser.add_argument('--data', type=str,help='st15, st17, arts, enter, ppl , atis') 

# parser.add_argument('--dataset', type=str, required=True,help='choose between wsj,slurp,atis')
args = parser.parse_args()

learning_rate = args.lr
dataset = args.data
# print('Using learning rate ',learning_rate)
# dataset =args.dataset
# print('Using dataset ',dataset)

# Hyperparameters
num_epochs = 25
weight_decay = 0.02

# Batch and device stuff
devices = args.d
batch_size = 32 / devices # trained atis with 32BS 1 gpu == 64BS with 2 GPUs
micro_batch_size = 4 # was 6 with 500
gradient_accumulation_steps = batch_size // micro_batch_size

# Dataset stuff
# data_dir: str = os.path.join("data",dataset)

# demo_config = { # you can only fit 1200 max_sequence_length with micro BS 8 in A100. These are the maximium no of demonstrations you can have without OOM
#     "wsj": 25,
#     "atis": 30,
#     "slurp": 45}
# num_demonstrations =  15# demo_config[dataset]

train_path = f'{dataset}_train.pt'
val_path = f'{dataset}_test.pt'

train_data = torch.load(train_path,map_location=torch.device('cpu'))
val_data   = torch.load(val_path,map_location=torch.device('cpu'))

train_data_len = len(train_data)
val_data_len = len(val_data)

print('loaded test data')

epoch_size = train_data_len // micro_batch_size // devices#50000  # train dataset size
max_iters = num_epochs * epoch_size 
eval_iters = val_data_len // micro_batch_size  // devices #100
warmup_steps = epoch_size * 0 // devices #// micro_batch_size // devices  # 2 epochs : changes becuase i changed epoch size from 5000 to train_data_len/micro BS

# Network stuff
max_seq_length = 2048  # see scripts/prepare_alpaca.py # 832 for v100
max_input_length = 1000 # 800 for v100 wo k,v ; 700 works for v100 w k,v

# Checkpointing stuff
#eval_interval =  val_data_len // micro_batch_size #600 # you are not using this as you changed the code to evaluate (output val loss) every epoch
save_interval = epoch_size  # save every epoch #1000
log_interval = 1
run_name = f'again_ablation_Masked_{dataset}_MINI_{learning_rate}_bad_init'
out_dir: str = 'runs/'+run_name

# wandb stuff
wandb.login()
wandb.init(
    # set the wandb project where this run will be logged
    project="GigaCat", #"LLAMA_ADAPTER",
    name=run_name,
    group=run_name,
    config={
    "learning_rate": learning_rate,
    "num_epochs": num_epochs,
    "weight_decay": weight_decay,
    "batch_size": (batch_size*devices),
    "micro_batch_size":micro_batch_size,
    "dataset":'gigaspeech',
    'devices':devices,
    "dataset":dataset,
    'max_input_length':max_input_length,
    }
)

ds_config = {
    "train_micro_batch_size_per_gpu": micro_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "zero_optimization": {"stage": 2},
}


def main():
    fabric = L.Fabric(
        accelerator="cuda", 
        devices=devices, 
        strategy= "ddp"  if devices > 1 else "auto" , #(DeepSpeedStrategy(config=ds_config) if devices > 1 else "auto"), 
        precision="bf16-true",
    )
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        
    config = LLaMAConfig(block_size=max_seq_length)
    
    pretrained_path: str = "model/Alpaca_PY/lit-llama.pth"
    if not os.path.isfile(pretrained_path):
        raise FileNotFoundError(
            f"Can't find the pretrained weights at {pretrained_path}."
            " Please follow the instructions in the README to download them."
        )
    checkpoint = torch.load(pretrained_path)
    print('loaded LLaMA checkpoint')
    with fabric.init_module():
        model = LLaMA(config)

    (_, w_ck_pt) = whisper.load_model("large-v2",device='cpu')
    print('loaded Whisper checkpoint')
    for n, p in model.named_parameters():
        if 'whisper' in n :
            #transformer.h.2.attn.whisper_value.weight
            layer = n.split('.')[2]
            suffix = n.split('.')[-1]
            kv = n.split('.')[4].split('_')[-1]
            #decoder.blocks.3.cross_attn.key.weight
            w_key = f'decoder.blocks.{layer}.cross_attn.{kv}.{suffix}'
            checkpoint[n] = w_ck_pt['model_state_dict'][w_key].cpu()

        
        
    with fabric.init_module():
         # strict=False because missing keys due to adapter weights not containted in state dict  
        model.load_state_dict(checkpoint, strict=False)
    print('loaded LAMMA model')
    mark_only_adapter_as_trainable(model)

    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Number of trainable parameters: {num_params/1e6}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model, optimizer = fabric.setup(model, optimizer)
    train(fabric, model, optimizer, train_data, val_data, out_dir)
    wandb.finish()

    # Save the final checkpoint at the end of training
    save_model_checkpoint(fabric, model, os.path.join(out_dir, "lit-llama-adapter-finetuned.pth"))


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    out_dir: str,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0 # gets updated each time you compleate a batch aka each time you take a step

    for iter_num in range(max_iters):


        #     lr = learning_rate * step_count / warmup_steps # what is happening here
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        #     wandb.log({"lr": lr})

        t0 = time.time()

        input_ids, targets, audio_features = get_batch(fabric, model, train_data)
        logits = model(input_ids, audio_features = audio_features)
        loss = loss_fn(logits, targets)
        with fabric.no_backward_sync(model, enabled=((iter_num + 1) % gradient_accumulation_steps != 0)): # Skip gradient synchronization during backward to avoid redundant communication overhead : so you skip it till you compleate your gradient accumulation steps
            fabric.backward(loss / gradient_accumulation_steps)

        if (iter_num + 1) % gradient_accumulation_steps == 0: # At step
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
            lr = learning_rate - ((learning_rate - 1e-5)/max_iters)*(iter_num)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            wandb.log({"lr": lr})

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt:.2f}s")
            wandb.log({"train_iter": iter_num, "train_Iter_loss": loss.item()})
       #validation takes too long - change to end of epoch  #MINEif (iter_num + 1) % gradient_accumulation_steps == 0:# WAS step_count % eval_interval == 0:
        if (iter_num + 1) % epoch_size == 0:# step_count % save_interval == 0:
            print(f"Saving adapter weights to {out_dir}")
            save_model_checkpoint(fabric, model, os.path.join(out_dir, f"iter-{int((iter_num+1)/epoch_size):06d}.pth"))

        # print loss on val set at end of epoch
            val_loss = validate(fabric, model, val_data)
            fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
            fabric.barrier()
            wandb.log({"val_step": iter_num, "val_step_loss": val_loss})

        # print loss on train set at the end of epoch
            print('End of epoch ',(iter_num+1)/epoch_size)
            # epoch_train_loss = validate(fabric, model, train_data)
            # fabric.print(f"step {iter_num}: epoch_train_loss {epoch_train_loss:.4f}")
            # fabric.barrier()
            # wandb.log({"train_epoch": (iter_num+1)/epoch_size, "train_epoch_loss":epoch_train_loss})
            

def generate_response(model, instruction, input=""):
    tokenizer = Tokenizer("model/tokenizer.model")
    sample = {"instruction": instruction, "input": input}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

    output = generate(
        model,
        idx=encoded,
        max_seq_length=max_seq_length,
        max_new_tokens=100,
        temperature=0.8,
    )
    output = tokenizer.decode(output)
    return output # output.split("### Response:")[1].strip()


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray) -> torch.Tensor:
    fabric.print("Validating ...")

    if len(val_data) == val_data_len :
        eval_iters =  val_data_len // micro_batch_size  // devices
    else :
        eval_iters =  epoch_size // devices

    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets, audio_features = get_batch(fabric, model, val_data)
        logits = model(input_ids, audio_features = audio_features)
        loss = loss_fn(logits, targets)
        losses[k] = loss.item()
    val_loss = losses.mean()

    # produce an example:
    # instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    # output = generate_response(model, instruction)
    # fabric.print(instruction)
    # fabric.print(output)

    model.train()
    return val_loss.item()

def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss
    

def get_batch(fabric: L.Fabric, model ,data: list):
    ix = torch.randint(len(data), (micro_batch_size,))

    input_ids = [data[i]["input_ids"][:max_input_length].type(torch.int64) for i in ix]
    labels = [data[i]["labels_with_masked_input"][:max_input_length].type(torch.int64) for i in ix]
    audio_features = [data[i]["audio_features"].type(model.dtype) for i in ix]
    #values = [data[i]["value"].type(model.dtype) for i in ix]   

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    af = torch.cat([x for x in audio_features], dim =0)
    # vs = torch.cat([x for x in values], dim =0)
    x, y , af  = fabric.to_device((x.pin_memory(), y.pin_memory(), af.pin_memory()))
    #ks , vs = fabric.to_device((ks.pin_memory(),vs.pin_memory() ))
    return x, y , af



def save_model_checkpoint(fabric, model, file_path):
    file_path = Path(file_path)

    if isinstance(fabric.strategy, DeepSpeedStrategy):
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

        tmp_path = file_path.with_suffix(".tmp")
        fabric.save(tmp_path, {"model": model})
        fabric.barrier()
        if fabric.global_rank == 0:
            # Create a consolidated checkpoint with the same name next to the deepspeed checkpoint
            # and only keep the adapter weights
            state_dict = get_fp32_state_dict_from_zero_checkpoint(tmp_path)
            state_dict = adapter_state_from_state_dict(state_dict)
            torch.save(state_dict, file_path)
            shutil.rmtree(tmp_path)
    else:
        state_dict = adapter_state_from_state_dict(model.state_dict())
        if fabric.global_rank == 0:
            torch.save(state_dict, file_path)
        fabric.barrier()


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    # from jsonargparse.cli import CLI

    # CLI(main)
    main()
    
