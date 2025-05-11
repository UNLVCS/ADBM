# A distributed training pipeline that uses deepspeed
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    StateDictType
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap
)
from functools import partial
from functools import partial
from transformers.models.t5.modeling_t5 import T5Block
from torch.utils.data import DataLoader, DistributedSampler, random_split
from transformers import T5ForConditionalGeneration, T5TokenizerFast, BitsAndBytesConfig, AutoConfig
from transformers.optimization import Adafactor, get_scheduler
from transformers.models.t5.modeling_t5 import T5Block
from dataset_utils import T5Dataset, collator
from tqdm import tqdm
import deepspeed
import os
import re
import json
import argparse
from peft import LoraConfig, get_peft_model
import numpy as np
import math
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt



RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
BOLD = "\033[1m"
RESET = "\033[0m"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup(rank, world_size):
    """
    Initialize the distributed environment.
    """
    # Setting up the addr and port of the coordinating process
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'

    os.environ["RANK"]       = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    # Initialize the process group
    # nccl is the backend that handles distributed computing accross nvidia cuda gpu's
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def get_deepspeed_config(config_path = '/home/tadesa1/research/ADBMO-UNLV/t5_lora_finetuning/ds_config.json'):
    """
    Load DeepSpeed configurations from json file
    """

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"{YELLOW}[INFO]: Deepspeed config loaded successfully from {config_path}{RESET}")
        return config
    except Exception as e:
        print(f"{RED}[ERROR]: Error loading Deepspeed configuration: {e}{RESET}")

def cleanup():
    """
    Clean up the distributed environment.
    """
    dist.destroy_process_group()

def train_t5_with_deepspeed(
    rank,
    world_size,
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    ds_config,
    collate_fn,
    batch_size=1,
    num_epochs=7,
    save_path="t5_finetuned.pt",
    resume_from_checkpoint=None,
    resume_optimizer=False,
    resume_scheduler=False
):
    """
    Train T5 with DeepSpeed, with checkpoint resumption
    """

    # Setting up distributed env
    setup(rank, world_size)

    # For training
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # For validation
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # Move model to the correct device
    device = torch.device(f"cuda:{rank}")

    # Creating torch optimizer and scheduler in case if not using deepspeed built in optim and sched

    # Calculate total steps for scheduler
    total_steps = len(train_loader) // ds_config['gradient_accumulation_steps'] * num_epochs

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=0.01
    )
    
    # Create scheduler
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=total_steps
    )

    steps_per_epoch = math.ceil(len(train_dataset) / (batch_size * world_size))
    effective_steps_per_epoch = math.ceil(steps_per_epoch / ds_config['gradient_accumulation_steps'])
    total_num_steps = effective_steps_per_epoch * num_epochs
    ds_config['scheduler']['params']['total_num_steps'] = total_num_steps

    # Initialize DeepSpeed
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        # optimizer = optimizer,
        # lr_scheduler=lr_scheduler,
        config=ds_config,
        dist_init_required=False  # Already initialized
    )

    # Tracking the starting epoch
    start_epoch = 0

    # Load from checkpoint if specified
    if resume_from_checkpoint and rank == 0:
        print(f"{BOLD}Loading DeepSpeed checkpoint from {resume_from_checkpoint}{RESET}")
        # DeepSpeed has its own checkpoint loading mechanism
        model.load_checkpoint(resume_from_checkpoint)
        # Extract epoch information if available
        try:
            checkpoint_name = os.path.basename(resume_from_checkpoint)
            if checkpoint_name.startswith("epoch"):
                # Try to extract epoch number from checkpoint name
                start_epoch = int(re.search(r"epoch(\d+)", checkpoint_name).group(1))
                print(f"{BOLD}Resuming from epoch {start_epoch}{RESET}")
        except:
            print(f"{RED}Could not determine start epoch from checkpoint name, starting from 0{RESET}")
            start_epoch = 0

    best_val_loss = float('inf')

    # Create metrics directory if it doesn't exist
    metrics_dir = os.path.join(os.path.dirname(save_path), "metrics")
    if rank == 0:
        os.makedirs(metrics_dir, exist_ok=True)
    
    # Track metrics for plotting - load previous metrics if resuming
    metrics_file = os.path.join(metrics_dir, "training_metrics.npz")
    if resume_from_checkpoint and os.path.exists(metrics_file) and rank == 0:
        try:
            saved_metrics = np.load(metrics_file)
            train_losses = saved_metrics['train_losses'].tolist()
            val_losses = saved_metrics['val_losses'].tolist()
            train_perplexities = saved_metrics['train_perplexities'].tolist()
            val_perplexities = saved_metrics['val_perplexities'].tolist()
            learning_rates = saved_metrics['learning_rates'].tolist()
            print(f"{GREEN}Loaded previous training metrics from {metrics_file}{RESET}")
        except Exception as e:
            print(f"{RED}Error loading metrics: {e}. Starting with empty metrics.{RESET}")
            train_losses, val_losses = [], []
            train_perplexities, val_perplexities = [], []
            learning_rates = []
    else:
        train_losses, val_losses = [], []
        train_perplexities, val_perplexities = [], []
        learning_rates = []
    
    # Ensure all processes are synchronized before entering training loop
    dist.barrier()

    for epoch in range(start_epoch, num_epochs):
        # Setting the epoch for the sampler
        train_sampler.set_epoch(epoch)

        # Switching model to train mode
        model.train()
        train_loss = 0.0


        loop = tqdm(train_loader, desc=f"{GREEN}Epoch {epoch+1}/{num_epochs}{RESET}", disable=(rank!=0))
        for batch in loop:


            # Move batch to device
            for k in ("input_ids", "attention_mask", "labels"):
                batch[k] = batch[k].to(device)

            # Forward pass
            loss = model(**batch).loss
            
            # Backward pass: DeepSpeed internally tracks gradient accumulation
            model.backward(loss)
            model.step()

            train_loss += loss.item()

            if rank == 0:
                # Update progress bar with current loss
                current_lr = lr_scheduler.get_last_lr()[0]

                loop.set_postfix(loss=loss.item(), lr=current_lr)

        # — aggregate training loss —
        avg_train = train_loss / len(train_loader)
        t_tensor = torch.tensor(avg_train, device=device)
        dist.all_reduce(t_tensor)
        avg_train = t_tensor.item() / world_size

        # Calculate training perplexity
        train_perplexity = torch.exp(torch.tensor(avg_train))

        # — validation pass —
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                for k in ("input_ids", "attention_mask", "labels"):
                    batch[k] = batch[k].to(device)
                val_loss += model(**batch).loss.item()

        avg_val = val_loss / len(val_loader)
        v_tensor = torch.tensor(avg_val, device=device)
        dist.all_reduce(v_tensor)
        avg_val = v_tensor.item() / world_size

        # Calculate validation perplexity
        val_perplexity = torch.exp(torch.tensor(avg_val))
        # Store metrics and save plot ONLY on rank 0
        if rank == 0:

            # Save checkpoint after every epoch
            checkpt_dir = os.path.join(save_path, f"checkpoint-epoch{epoch}")
            model.save_checkpoint(str( checkpt_dir) )
            print(f"{BOLD}{GREEN}-> Saved checkpoint at epoch: {epoch} to {checkpt_dir}{RESET}")
            
            # Update metrics lists
            train_losses.append(avg_train)
            val_losses.append(avg_val)
            train_perplexities.append(train_perplexity.item())
            val_perplexities.append(val_perplexity.item())
            learning_rates.append(lr_scheduler.get_last_lr()[0])
            
            # Save metrics to disk (atomic operation)
            tmp_metrics_file = os.path.join(metrics_dir, f"tmp_metrics_{epoch}.npz")
            np.savez(
                tmp_metrics_file,
                train_losses=np.array(train_losses),
                val_losses=np.array(val_losses),
                train_perplexities=np.array(train_perplexities),
                val_perplexities=np.array(val_perplexities),
                learning_rates=np.array(learning_rates)
            )
            # Atomic rename to avoid partial writes
            os.replace(tmp_metrics_file, metrics_file)
            
            # Save the plot
            try:
                save_metrics_plot(
                    train_losses, 
                    val_losses,
                    train_perplexities,
                    val_perplexities,
                    learning_rates, 
                    os.path.join(metrics_dir, "training_progress.png")
                )
            except Exception as e:
                print(f"{RED}Error saving plot: {e}{RESET}")
        
        # Make sure all processes are synced before moving to the next epoch
        dist.barrier()
        
        print(f"{BLUE}{BOLD} Epoch {epoch+1} ▹ Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}{RESET}")

        # Taking a snap shot of the best-model if validation loss is good 
        if avg_val < best_val_loss:
            
            best_val_loss = avg_val

            with deepspeed.zero.GatheredParameters(model.module.parameters(), modifier_rank=0):
                # only rank 0 actually dumps the files
                if rank == 0:
                    model.module.save_pretrained(f"{save_path}_best/")
                    tokenizer.save_pretrained(f"{save_path}_best/")
                    print(f"{GREEN}{BOLD} → New best model saved with Val Loss {best_val_loss:.4f}{RESET}")
        
        # Sync up before the next epoch
        dist.barrier()

    # Clean up distributed environment
    cleanup()

def train_t5_distributed(
    model,
    tokenizer,
    dataset,
    collate_fn,
    batch_size=1,
    num_gpus=2,
    num_epochs=20,
    save_path="t5_finetuned",
    resume_from_checkpoint=None,
    resume_optimizer=False,
    resume_scheduler=False        
):
    """
        Main function that spawns up multiple processes for distributed training using deepspeed
    """
    assert torch.cuda.is_available(), f"{RED}{BOLD}No GPU available!{RESET}"
    world_size = min(torch.cuda.device_count(), num_gpus)

    ds_config = get_deepspeed_config()

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"{GREEN}{BOLD}Starting distributed training on {world_size} GPUs using deepspeed...{RESET}")

    mp.spawn(
        train_t5_with_deepspeed,
        args=(world_size, model, tokenizer, train_dataset, val_dataset, ds_config, collate_fn, batch_size, num_epochs, save_path,
              resume_from_checkpoint, resume_optimizer, resume_scheduler),
        nprocs=world_size,
        join=True
    )

    print(f"{GREEN}{BOLD}Training Completed!{RESET}")

def save_metrics_plot(train_losses, val_losses, train_perplexities, val_perplexities, learning_rates, save_path):
    """
    Create and save a plot of training metrics with proper error handling
    """
    try:
        # Create a temporary file first
        tmp_path = f"{save_path}.tmp"
        
        plt.figure(figsize=(12, 15))
        
        # Create subplot for losses
        plt.subplot(3, 1, 1)
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Create subplot for perplexities
        plt.subplot(3, 1, 2)
        plt.plot(epochs, train_perplexities, 'b-', label='Training Perplexity')
        plt.plot(epochs, val_perplexities, 'r-', label='Validation Perplexity')
        plt.title('Training and Validation Perplexity')
        plt.xlabel('Epochs')
        plt.ylabel('Perplexity')
        plt.legend()
        plt.grid(True)
        
        # Create subplot for learning rate
        plt.subplot(3, 1, 3)
        plt.plot(epochs, learning_rates, 'g-', label='Learning Rate')
        plt.title('Learning Rate')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save to temporary file first
        plt.savefig(tmp_path, format='png')
        plt.close()
        
        # Then do an atomic rename
        os.replace(tmp_path, save_path)
        
        print(f"{GREEN}{BOLD}Training metrics plot saved to {save_path}{RESET}")
    except Exception as e:
        print(f"{RED}{BOLD}Error in save_metrics_plot: {e}{RESET}")

def load_training_dt(input_file):
    # Load JSON
    with open(input_file, 'r') as f:
        data = json.load(f)
        
    texts = []
    annotations = []
    # Flatten json structure
    for pmid, sections in data.items():
        for section_type, content in sections.items():
            section_text = " ".join(content.get("text", []))
            section_annotations = []
            for ann_group in content.get("annotation", []):
                try:
                    start = int(ann_group['offset'])
                    end = start + int(ann_group['length'])
                    section_annotations.append((start, end))
                except:
                    continue
            texts.append(section_text)
            annotations.append(section_annotations)

    with open('/home/tadesa1/research/ADBMO-UNLV/t5_training/gene_species_tagged_articles.json', 'r') as f:
        data = json.load(f)
    # Flatten json structure
    for pmid, sections in data.items():
        for section_type, content in sections.items():
            section_text = " ".join(content.get("text", []))

            section_annotations = []
            for ann_group in content.get("annotation", []):
                for ann_id, ann in ann_group.items():
                 
                    try:
                        start = int(ann["offset"])
                        end = start + int(ann["length"])
                        section_annotations.append((start, end))
                    except:
                        continue

            texts.append(section_text)
            annotations.append(section_annotations)

    with open('/home/tadesa1/research/ADBMO-UNLV/data/disease_tagged_articles.json', 'r') as f:
        data = json.load(f)

    for pmid, sections in data.items():
        for section_type, content in sections.items():
            try:
                section_text = " ".join(content.get("text", []))
                section_annotations = []
                for ann_group in content.get("annotation", []):
                    for ann_id, ann in ann_group.items():
                    
                        try:
                            start = int(ann["offset"])
                            end = start + int(ann["length"])
                            section_annotations.append((start, end))
                        except:
                            continue
                texts.append(section_text)
                annotations.append(section_annotations)
            except AttributeError as e:
                print(f"{RED} Attribute error when loading disease data. Ignore {RESET}")
    return (texts, annotations)

class CollatorWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        from dataset_utils import collator
        return collator(batch, self.tokenizer)

if __name__ == '__main__':

    # Parsing command line arguments

    parser = argparse.ArgumentParser(description=f'{BOLD}{BLUE}Train Flan-T5 Models with DeepSpeed{RESET}')
    parser.add_argument('--model_name', type=str, default="google/flan-t5-base",
        choices=["google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large", "google/flan-t5-xl", "google/flan-t5-xxl"],
        help=f'{BOLD}T5 model variant to use{RESET}')

    parser.add_argument('--batch_size', type=int, default=1, 
        help=f'{BOLD}Batch size per GPU{RESET}')

    parser.add_argument('--num_gpus', type=int, default=2, 
        help=f'{BOLD}Number of GPUs to use{RESET}')

    parser.add_argument('--num_epochs', type=int, default=20, 
        help=f'{BOLD}Number of training epochs{RESET}')

    parser.add_argument('--input_file', type=str, 
        default='/home/tadesa1/research/ADBMO-UNLV/t5_training/converted_from_biocjson.json',
        help=f'{BOLD}Input text file for training{RESET}')

    parser.add_argument('--save_path', type=str, default='/home/tadesa1/research/ADBMO-UNLV/t5_lora_finetuning/t5_finetuned',
        help=f'{BOLD}Path to save model checkpoints{RESET}')

    parser.add_argument('--resume_from_checkpoint', type=str, default=False,
        help=f'{BOLD}Path to checkpoint to resume training from{RESET}')

    parser.add_argument('--resume_optimizer', action='store_true',
        help=f'{BOLD}Whether to also load optimizer state from checkpoint{RESET}')

    parser.add_argument('--resume_scheduler', action='store_true',
        help=f'{BOLD}Whether to also load scheduler state from checkpoint{RESET}')
    
    args = parser.parse_args()

    # Suppress TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Print PyTorch version for debugging
    print(f"{YELLOW}PyTorch version: {torch.__version__}{RESET}")
    
    # Set NCCL_DEBUG for better CUDA debugging
    os.environ['NCCL_DEBUG'] = 'WARN'


    print(f"***********{BOLD}{GREEN}Loading Model: {args.model_name}{RESET}***********")
    print(f"{BOLD}Loading in bfloat16 precision{RESET}")
    model = T5ForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        token=os.getenv('HF_ACCESS_TOKEN')
    )
    # with deepspeed.zero.Init(
    #     remote_device='cpu',
    #     pin_memory=True,
    #     config_dict_or_path="ds_config.json"
    # ):
    #     model = T5ForConditionalGeneration.from_pretrained(
    #         args.model_name,
    #         torch_dtype=torch.bfloat16,    # or float16
    #         device_map=None,                # very important
    #     )


    # Configuring LoRA to target all layers of the transformer. 
    # config = LoraConfig(
    #     task_type = "SEQ_2_SEQ_LM",
    #     # target_modules=["q", "v"],
    #     target_modules=["q", "k", "v", "o", "wi", "wo"],
    #     r=16,
    #     lora_alpha=32,
    #     lora_dropout=0.1
    # )
    
    # model = get_peft_model(model, config)
    # print(f"***********{BOLD}")
    # model.print_trainable_parameters()
    # print(f"{RESET}***********")


    # Initialize tokenizer
    tokenizer = T5TokenizerFast.from_pretrained(args.model_name)

    texts, annotations = load_training_dt(args.input_file)

    dataset = T5Dataset(
        texts=texts,
        tokenizer=tokenizer,
        annotations=annotations
    )

    my_collator = CollatorWrapper(tokenizer)

    # Start training
    train_t5_distributed(
        model=model,
        tokenizer = tokenizer,
        dataset=dataset,
        collate_fn=my_collator,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        num_epochs=args.num_epochs,
        save_path=args.save_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
        resume_optimizer=args.resume_optimizer,
        resume_scheduler=args.resume_scheduler
    )