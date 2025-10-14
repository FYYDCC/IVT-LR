import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.distributed
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import timedelta
import deepspeed
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from torch.optim import AdamW
import shutil
import numpy as np
from torch.utils.data import Subset
from collections import OrderedDict
import re

import wandb

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
from datasets import load_dataset
import logging
logging.basicConfig(
    filename='chameleon_sqa_train.log',  
    level=logging.DEBUG,          
    format='[%(asctime)s] %(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S'   
)

from chameleon_ivtlr import IVTLR
from chameleon_dataset import (
    get_dataset,
    get_cot_latent_dataset,
    MyCollator,
)

from tqdm import tqdm
from copy import copy
import itertools
import os, sys
import yaml
import json
import gc
import argparse
import functools
from utils import Config, set_seed
import pdb
from peft import LoraConfig, get_peft_model

# LoRA 
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    inference_mode=False
)

def main():
    print("Initializing DeepSpeed Training!")
    parser = argparse.ArgumentParser(description="IVTLR")
    parser.add_argument("config_file")
    parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed")
    parser.add_argument("--deepspeed_config", default="ds_config.json", help="DeepSpeed config path")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by DeepSpeed")
    args = parser.parse_args()

    # Initialize DeepSpeed
    deepspeed.init_distributed()
    local_rank = args.local_rank
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    print("line 57")
    # load the configuration file
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    configs = Config(config_dict)
    set_seed(configs.seed)
    save_dir = os.path.join(configs.save_path, configs.name)

    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)

    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    
    cur_ckpts = os.listdir(save_dir)
    

    # check if the job is preempted and resumed.
    if len(cur_ckpts) > 0 and rank == 0:
        raise ValueError(
            f"Save directory {save_dir} is not empty! "
        )

    if configs.resume != 0:
        # by setting `resume`, we can skip a few epoches at the beginning.
        print(
            f"Loading from {configs.load_model_path} and skip the first {configs.resume} epochs"
        )
        
    
    print("start loading model")

    processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")
    model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True, attn_implementation="eager")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=configs.lr)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    print("latent_id: ", latent_id)
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    image_token_id = tokenizer.convert_tokens_to_ids(processor.image_token)
    logging.debug(f"image_token_id: {image_token_id}")
    logging.debug(f"image_token: {processor.image_token}")

    model = get_peft_model(model, lora_config)

    
    model.resize_token_embeddings(len(tokenizer))
    embeddings = model.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids("<<")
    # initialize the new token embeddings with a known token
    # it helps stablize the training
    for token_id in [latent_id, start_id, end_id]:
        target_embedding = embeddings.weight.data[token_id]
        embeddings.weight.data[token_id] = target_embedding
        lm_head = model.lm_head
        lm_head.weight.data[token_id] = lm_head.weight.data[target_id]
    
    model.print_trainable_parameters()

    model = IVTLR(model, latent_id, start_id, end_id, tokenizer.eos_token_id, image_token_id)

    print(f"Running Deepspeed on rank = {rank}, world size = {world_size}")
    model = model.to(rank)


    if configs.bf16:
        model.to(torch.bfloat16)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=args.deepspeed_config,
        # optimizer = optimizer,
        model_parameters=filter(lambda p: p.requires_grad, model.parameters())
    )

    del model

    dataset = load_dataset("derek-thomas/ScienceQA")

    def process_example(example):
        example["answer"] = str(example["answer"])
        # rationale：merge lecture and solution
        lecture = example.get("lecture", "") or ""
        solution = example.get("solution", "") or ""
        
        # merge lecture and solution
        if lecture and solution:
            rationale = (lecture.strip() + " " + solution.strip()).strip()
        elif lecture:
            rationale = lecture.strip()
        elif solution:
            rationale = solution.strip()
        else:
            # both is null
            rationale = example["answer"]
            print(f"Warning: Both lecture and solution are empty for question: {example['question']}")
            rationale = str(rationale)
        
        rationale = rationale.replace("\n", " ").strip()
        example["steps"] = rationale.split(". ")
        if example["steps"][-1] == "":
            example["steps"].pop()

        # multi-steps
        if len(example["steps"]) > 3:
            total_steps = len(example["steps"])
            
            step_size = total_steps // 3
            remainder = total_steps % 3

            
            new_steps = []
            start = 0

           
            for i in range(3):
                end = start + step_size + (1 if i < remainder else 0)
                new_steps.append(". ".join(example["steps"][start:end]))
                start = end

            example["steps"] = new_steps
        
        question = example["question"]
        choices = example["choices"]
        image_token_str = "<image>" * 1024
        question_prefix = f"{image_token_str}"
        
        choices_str = "[Options]:\n"+"\n".join([
            f"({chr(65 + i)}).{{{choice.strip()}}}"  
            for i, choice in enumerate(choices)
        ])
        question = question_prefix + question
        question_with_braces = f"{{{question.strip()}}}"
        
        example["question"] = f"[Question]:{question_with_braces}\n{choices_str}"
        
        # 删除不需要的列
        if "lecture" in example:
            del example["lecture"]
        if "solution" in example:
            del example["solution"]
        if "choices" in example:
            del example["choices"]
        
        return example
    print("start dataset")
    # 处理 train 数据集

    
    def has_image(example):
        # 检查 "image" 和 "choice" 是否存在且非空
        return (
            "image" in example and example["image"] is not None
        )

    train_dataset = dataset["train"].select(range(10)).filter(has_image)
    train_dataset = train_dataset.map(process_example, num_proc=32)

    base_dataset_train = get_dataset(
        train_dataset, tokenizer, processor, max_size=5000 if configs.debug else 100000000
    )


    total_train_steps = 0

    if not configs.debug and rank == 0:
        wandb_run = wandb.init(project=configs.project, name=configs.name)
        wandb_run.config.update(configs, allow_val_change=True)
        text_table = wandb.Table(columns=["step", "text"])

    else:
        wandb_run = None


    best_acc = 0

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
    # pdb.set_trace()
    for epoch in range(configs.resume, configs.num_epochs):

        scheduled_stage = epoch // configs.epochs_per_stage
    
        np.random.seed(epoch) 

        dataset_train = get_cot_latent_dataset(
            scheduled_stage,
            base_dataset_train,
            configs,
            start_id,
            latent_id,
            end_id,
            no_special_marker=True,
            shuffle=True,
        )

        train_dataloader = torch.utils.data.DataLoader(
            dataset_train,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            batch_size=configs.batch_size_training,
            collate_fn=collator,
            sampler=DistributedSampler(dataset_train, shuffle=True),
        )
        
        model_engine.train()
        total_length = len(train_dataloader) // configs.gradient_accumulation_steps
        pbar = tqdm(
            colour="blue",
            desc=f"Training Epoch: {epoch+1}",
            total=total_length,
            dynamic_ncols=True,
        )
        for step, batch in enumerate(train_dataloader):
            if step == 0 and wandb_run and rank == 0:
                print("logging training data")
                cur_bs = len(batch["input_ids"])
                text_str = ""
                for data_idx in range(cur_bs):
                    for token_idx in range(len(batch["input_ids"][data_idx])):
                        text_str += (
                            str(batch["input_ids"][data_idx][token_idx].item())
                            + " "
                            + str(batch["labels"][data_idx][token_idx].item())
                            + " "
                            + tokenizer.decode(
                                batch["input_ids"][data_idx][token_idx]
                            )
                            + "\n"
                        )
                        
                    text_str += "====" * 10 + "\n"
                    
                text_table.add_data(total_train_steps, text_str)

            total_train_steps += 1
            batch = {
                key: batch[key].to(rank) for key in batch.keys() if key != "idx"
            }

            outputs = model_engine(**batch)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()
            
            if wandb_run and rank == 0:
                log_dict = {
                    "train/epoch": epoch + 1,
                    "train/step": epoch * len(train_dataloader) + step,
                    "train/loss": loss.detach().float()
                }
                wandb_run.log(log_dict)
            
            pbar.set_description(
                f"Training Epoch: {epoch+1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} "
                f"completed (loss: {round(float(loss.detach().float()), 4)}"
            )
        pbar.close()
        dist.barrier()

        if (
            not configs.debug
            and (epoch + 1) % 4 == 0
        ):
            epoch_save_dir = os.path.join(save_dir, f"epoch_{epoch+1}_checkpoint")

            model_engine.save_checkpoint(
                save_dir=epoch_save_dir,  
                tag=f"epoch_{epoch+1}_zero3_bf32",
                client_state={"best_acc": best_acc, "current_epoch": epoch+1}
            )

            if rank == 0:
                # save full model
                fp32_state_dict = get_fp32_state_dict_from_zero_checkpoint(epoch_save_dir, tag=f"epoch_{epoch+1}_zero3_bf32")
                fp32_output = os.path.join(save_dir, f"epoch_{epoch+1}_full_model_fp32.pth")

                torch.save(fp32_state_dict, fp32_output)
                
                print(f"Epoch {epoch+1} save to {fp32_output}")

                if os.path.exists(epoch_save_dir):
                    shutil.rmtree(epoch_save_dir)

            dist.barrier()
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
