from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
import os
import argparse

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.optim import Adam
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    AutoModelForSequenceClassification
)

from accelerate import Accelerator
import numpy as np

from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    create_reference_model,
    set_seed,
)
import functools
import gc
from collections import defaultdict
import warnings
import pandas as pd
import transformers
transformers.logging.set_verbosity_error()

def build_dataset(
    tokenizer, input_min_text_length=5, input_max_text_length=100, train=True
):
    """각 파라미터에 따른 데이터 개수

    input_min_text_length=5
    input_max_text_length=50
    >>> 235672

    input_min_text_length=0
    input_max_text_length=50
    >>> 236859

    input_min_text_length=5
    input_max_text_length=100
    >>> 262041

    input_min_text_length=0
    input_max_text_length=100
    >>> 263228
    """

    TRAIN_PATH = "datasets/train"
    TEST_PATH = "datasets/test"
    train_set = [
        os.path.join(TRAIN_PATH, d)
        for d in os.listdir(TRAIN_PATH)
        if d.endswith(".csv")
    ]
    test_set = [
        os.path.join(TEST_PATH, d) for d in os.listdir(TEST_PATH) if d.endswith(".csv")
    ]

    # NOTE: OASST 먼저 처리하기
    train_set = sorted(train_set, key=lambda x: '0' if 'OASST' in x else x)
    test_set = sorted(test_set, key=lambda x: '0' if 'OASST' in x else x)

    if train:
        ds = load_dataset("csv", data_files={"train": train_set})
        ds = ds["train"]
    else:
        ds = load_dataset("csv", data_files={"test": test_set})
        ds = ds["test"]

    def tokenize(sample):
        prompt = sample["text"] + tokenizer.eos_token

        sample.update(tokenizer(prompt))
        return sample

    def filter_data(sample):
        return input_min_text_length <= len(sample['input_ids']) <= input_max_text_length

    ds = ds.map(tokenize, batched=False)
    ds = ds.filter(filter_data)
    ds.set_format(type="torch")

    return ds


def make_batch(data, tokenizer, device):
    batch = dict((key, [d[key] for d in data]) for key in data[0])
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    maxlen = max([len(i) for i in input_ids])

    def get_id_pads(x):
        return torch.full(
            (maxlen-len(x),), tokenizer.pad_token_id).to(x.device)

    def get_attn_pads(x):
        return torch.full(
            (maxlen-len(x),), 0).to(x.device)

    if tokenizer.padding_side == 'right':
        batch_input_ids = [torch.cat((x, get_id_pads(x))) for x in input_ids]
        batch_attn_mask = [torch.cat((x, get_attn_pads(x))) for x in attention_mask]
    else:
        batch_input_ids = [torch.cat((get_id_pads(x), x)) for x in input_ids]
        batch_attn_mask = [torch.cat((get_attn_pads(x), x)) for x in attention_mask]

    update_this = dict(
        data_id=list(batch['data_id']),
        input_ids=torch.stack(batch_input_ids),
        attention_mask=torch.stack(batch_attn_mask)
    )
    batch.update(update_this)
    return batch


@torch.no_grad()
def evaluate(dataloader, model, tokenizer, rank_model, rank_tokenizer, device, generation_kwargs):
    warnings.filterwarnings(action='ignore')
    table = defaultdict(list)
    for batch in tqdm(dataloader):

        # generate answers
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        maxlen = input_ids.shape[1]
        output_ids = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  **generation_kwargs)[:, maxlen:]
        answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        # compute rewards
        question = batch['text']
        inputs = rank_tokenizer(question, answer,
                                padding=True, truncation=True, return_tensors='pt')
        score = rank_model(**inputs.to(device)).logits.cpu().flatten().tolist()

        table['data_id']  += batch['data_id']
        table['source']   += batch['source']
        table['question'] += question
        table['answer']   += answer
        table['score']    += score

    warnings.filterwarnings(action='default')
    return table

@torch.no_grad()
def param_diff(model1, model2, rtol=1e-05):
    param1 = torch.cat([p.flatten() for p in model1.parameters()])
    param2 = torch.cat([p.flatten() for p in model2.parameters()])
    print("Is close?", torch.allclose(param1, param2, rtol=rtol))

def parsing_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', dest='model_name', type=str, default="microsoft/DialoGPT-medium")
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)

    args = parser.parse_args()
    return args


def main(args):
    
    # set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    
    # load LM
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    )

    baseline = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium", torch_dtype=torch.bfloat16)
    param_diff(model, baseline)
    
    # load RM
    reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
    rank_model = AutoModelForSequenceClassification.from_pretrained(reward_name)
    rank_tokenizer = AutoTokenizer.from_pretrained(reward_name)
     
    # load dataset
    min_input_length = 5
    max_input_length = 100
    dataset = build_dataset(
        tokenizer,
        input_min_text_length=min_input_length,
        input_max_text_length=max_input_length,
        train=False,
    )
    collate_fn = functools.partial(make_batch, tokenizer=tokenizer, device=device)
    dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                collate_fn=collate_fn,
                num_workers=4,
                shuffle=False,
                drop_last=False,
            )

    gc.collect()
    torch.cuda.empty_cache()
    
    generation_kwargs = {
        "min_length": -1,
        "max_length": 1024,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    model.to(device)
    rank_model.to(device)
    results = evaluate(dataloader, model, tokenizer, rank_model, rank_tokenizer, device, generation_kwargs)
    
    df = pd.DataFrame(results)
    df.to_csv('baseline.csv', index=False)

    print("#"*100)
    print("Min Score Example")
    print(df.loc[np.argmin(df.score)])

    print("#"*100)
    print("Max Score Example")
    print(df.loc[np.argmax(df.score)])


    baseline = pd.read_csv("baseline.csv")
    print("Improvement rate:", (df.score > baseline.score).sum() / len(df))

if __name__ == '__main__':
    args = parsing_arguments()
    main(args)
