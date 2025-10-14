"""
Chameleon infer on ScienceQA
"""

from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
from chameleon_ivtlr import IVTLR
import torch
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import re
import logging
import json
import os
import time
from datetime import timedelta

logging.basicConfig(
    filename='chameleon_sqa_infer_64_full.log',
    level=logging.DEBUG,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_inference_model(checkpoint_path):

    print("Loading Chameleon model...")
    

    processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    

    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<|start-latent|>",
            "<|end-latent|>",
            "<|latent|>"
        ]
    })
    
    base_model = ChameleonForConditionalGeneration.from_pretrained(
        "facebook/chameleon-7b",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    base_model.resize_token_embeddings(len(tokenizer))
    processor.tokenizer = tokenizer
    
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        inference_mode=False
    )
    base_model = get_peft_model(base_model, lora_config)
    
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    image_token_id = tokenizer.convert_tokens_to_ids(processor.image_token)
    
    model = IVTLR(
        base_model,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
        image_token_id=image_token_id,
    )
    
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=True)
    print("✓ Model loaded successfully")
    
    model = model.to(device)
    model.eval()
    return model, processor, tokenizer

model, processor, tokenizer = load_inference_model("your_pth_path")

os.makedirs("sqa_output", exist_ok=True)

def format_prompt(example):
    question = example["question"].strip()
    answer = example["answer"] 
    choices = example.get("choices", [])
    image = example["image"]

    image_token_str = "<image>"
    
    if choices:
        choices_str = "\n".join([f"({chr(65+i)}).{{{choice.strip()}}}" for i, choice in enumerate(choices)])
        user_prompt = (
            f"{image_token_str}[Question]:{{{question}}}\n"
            f"[Options]:\n{choices_str}\n"
            f"Answer:"
        )
    else:
        user_prompt = f"{image_token_str}[Question]:{{{question}}}\nAnswer:"
    
    return user_prompt, answer, image

def process_func(example, idx):

    prompt, answer, image = format_prompt(example)
    
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to("cuda")
    
    return {
        "idx": idx,
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": inputs["pixel_values"],
        "question_raw": prompt,
        "image_raw": image,
        "gt_answer": answer,
    }

dataset = load_dataset("derek-thomas/ScienceQA")
test_dataset = dataset["test"]

# To test with images
def has_image(example):
    return "image" in example and example["image"] is not None

test_dataset = test_dataset.filter(has_image)
test_dataset = test_dataset.map(lambda example, idx: process_func(example, idx), with_indices=True)

print(f"Total test samples with images: {len(test_dataset)}")

def extract_answer(text):

    digit_patterns = [
        r'Therefore,?\s*the\s+answer\s+is\s+(\d)',
        r'the\s+answer\s+is\s+(\d)',
        r'answer\s+is:?\s*(\d)',
    ]
    
    for pattern in digit_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer_idx = int(match.group(1))
            logging.debug(f"Extracted answer (digit): {answer_idx}")
            return answer_idx
        
    letter_patterns = [
        r'Therefore,?\s*the\s+answer\s+is\s+([A-Z])',
        r'the\s+answer\s+is\s+([A-Z])',
        r'answer\s+is:?\s*([A-Z])',
    ]
    
    for pattern in letter_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            answer_idx = ord(letter) - ord('A')
            logging.debug(f"Extracted answer (letter): {letter} -> index {answer_idx}")
            return answer_idx
    
    logging.warning(f"No answer pattern found in text: {text[:200]}")
    return -1

def evaluate_and_save(eval_dataset, model, processor):
    model.eval()
    correct = 0
    total = 0
    results = {}  
    total_generated_tokens = 0
    total_generate_time = 0.0
    
    output_json_path = "sqa_output/chameleon_scienceqa_results_full.json"
    
    for ex in eval_dataset:
        idx = str(ex["idx"])
        prompt = ex["question_raw"] + "<|latent|>" + "<|latent|>" + "<|latent|>"
        
        inputs = processor(
            images=ex["image_raw"],
            text=prompt,
            return_tensors="pt"
        ).to(device)
        
        prompt_length = inputs["input_ids"].shape[1]
        
        generate_start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512
            )
        
        generate_end_time = time.time()
        sample_generate_time = generate_end_time - generate_start_time
        total_generate_time += sample_generate_time
        
        generated_tokens = outputs[0, prompt_length:]
        generated_text = processor.decode(generated_tokens, skip_special_tokens=True)
        
        num_generated_tokens = len(generated_tokens)
        total_generated_tokens += num_generated_tokens
        
        logging.debug(f"[IDX {idx}] Generated {num_generated_tokens} tokens in {sample_generate_time:.3f}s: {generated_text}")
        
        pred_answer = extract_answer(generated_text)
        results[idx] = pred_answer
        
        gt_answer = ex["gt_answer"]
        if pred_answer == gt_answer:
            correct += 1
        
        total += 1
        
        if total % 100 == 0:
            avg_tokens = total_generated_tokens / total
            avg_time_per_sample = total_generate_time / total
            logging.info(f"Processed {total} samples, Current Acc: {correct/total:.2%}, Avg tokens: {avg_tokens:.1f}, Avg time: {avg_time_per_sample:.3f}s/sample")
            print(f"Processed {total} samples, Current Acc: {correct/total:.2%}, Avg tokens: {avg_tokens:.1f}, Avg time: {avg_time_per_sample:.3f}s/sample")
    

    output_data = {"results": results}
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    accuracy = correct / total if total > 0 else 0
    avg_generated_tokens = total_generated_tokens / total if total > 0 else 0
    avg_time_per_sample = total_generate_time / total if total > 0 else 0
    

    logging.info(f"[FINAL] Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2%}")
    logging.info(f"[FINAL] Avg generated tokens per sample: {avg_generated_tokens:.1f}")
    logging.info(f"[FINAL] Total generate time: {total_generate_time:.2f}s ({timedelta(seconds=int(total_generate_time))})")
    logging.info(f"[FINAL] Avg generate time per sample: {avg_time_per_sample:.3f}s")
    

    print(f"\n{'='*60}")
    print(f"[CHAMELEON - ScienceQA - FINAL RESULTS]")
    print(f"{'='*60}")
    print(f"  Total samples: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Average generated tokens per sample: {avg_generated_tokens:.1f}")
    print(f"\n Generate Time Statistics:")
    print(f"  Total generate time: {timedelta(seconds=int(total_generate_time))} ({total_generate_time:.2f}s)")
    print(f"  Average generate time per sample: {avg_time_per_sample:.3f}s")
    print(f"{'='*60}")
    print(f"Results saved to: {output_json_path}\n")
    
    return accuracy

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"Starting Chameleon inference on ScienceQA test set")
    print(f"{'='*60}\n")
    
    evaluate_and_save(test_dataset, model, processor)

