# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
import os

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.optim import Adam
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)
from accelerate import Accelerator

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model, set_seed
from trl.core import LengthSampler
from l_models import get_l_models

tqdm.pandas()

########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# This example fine-tunes a GPTJ model to generate less toxic contents
# by using allenai/real-toxicity-prompts dataset. We use PPO
#  (proximal policy optimization) to optimize the model.
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - multi GPUS (using DeepSpeed ZeRO-Offload stages 1 & 2)
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, first initialize the accelerate
# configuration with `accelerate config`
#
########################################################################


# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
# `accelerator_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="microsoft/DialoGPT-medium", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=(1.47e-5) * 2, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=4, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    ppo_epoch: Optional[int] = field(default=5, metadata={"help": "ppo epoch"})
    use_usefulness: Optional[bool] = field(default=True, metadata={"help": "use usefulness as reward"})
    use_harmfulness: Optional[bool] = field(default=True, metadata={"help": "use harmfulness as constraint"})
    lambda_type: Optional[str] = field(default='constant', metadata={"help": "type of Lambda = constant, linear"})
    lambda_value: Optional[float] = field(default=-1, metadata={"help": "value of lambda if type is constraint"})
    lambda_lr: Optional[float] = field(default=0.0, metadata={"help": "learning rate of lambda"})
    max_constraint: Optional[float] = field(default=0.0, metadata={"help": "value of t"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    model_save_path: Optional[str] = field(
        default="./ms_dialogpt_medium",
        metadata={"help": "the path to save the model"},
    )
    dataset_name: Optional[str] = field(
        default="allenai/real-toxicity-prompts"
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    ppo_epochs=script_args.ppo_epoch,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
)


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(
    config, input_min_text_length=5, input_max_text_length=100, train=True
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    TRAIN_PATH = 'datasets/train'
    TEST_PATH = 'datasets/test'
    train_set = [os.path.join(TRAIN_PATH, d) for d in os.listdir(TRAIN_PATH) if d.endswith('.csv')]
    test_set = [os.path.join(TEST_PATH, d) for d in os.listdir(TEST_PATH) if d.endswith('.csv')]

    # NOTE: BBQ 먼저 처리하면 에러가 발생하네요....
    train_set = sorted(train_set, key=lambda x: '0' if 'OASST' in x else x)
    test_set = sorted(test_set, key=lambda x: '0' if 'OASST' in x else x)

    print(train_set)
    print(test_set)

    ds = load_dataset("csv", data_files={"train": train_set, "test": test_set})

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    if train:
        ds = ds["train"]
    else:
        ds = ds["test"]

    def tokenize(sample):
        prompt = sample["text"]

        sample["input_ids"] = tokenizer.encode(prompt)[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
min_input_length = 30
max_input_length = 40
dataset = build_dataset(config, input_min_text_length=min_input_length, input_max_text_length=max_input_length, train=True)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer. We first load the model
# in bfloat16 to save memory using `transformers`.
model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16)
# And then we pass the loaded model to `AutoModelForCausalLMWithValueHead`.
model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

# We create a reference model by sharing 20 layers
ref_model = create_reference_model(model, num_shared_layers=20)

# We make sure to use `Adam` optimizer on the model parameters that require gradients.
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

# GPT-2 / GPT-J tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# We then build the reward pipeline, we will use the toxicity model to compute the reward.
# We first load the toxicity model and tokenizer.
toxicity_model_id = "facebook/roberta-hate-speech-dynabench-r4-target"
toxicity_tokenizer = RobertaTokenizer.from_pretrained(toxicity_model_id)
# We load the toxicity model in fp16 to save memory.
toxicity_model = RobertaForSequenceClassification.from_pretrained(toxicity_model_id, torch_dtype=torch.float16).to(
    ppo_trainer.accelerator.device
)

# usefulness_model_id = "allenai/dsp_roberta_base_tapt_amazon_helpfulness_115K" 
# usefulness_tokenizer = RobertaTokenizer.from_pretrained(usefulness_model_id)
usefulness_model_id = "OpenAssistant/reward-model-deberta-v3-large-v2" 
usefulness_tokenizer = AutoTokenizer.from_pretrained(usefulness_model_id)
usefulness_model = AutoModelForSequenceClassification.from_pretrained(usefulness_model_id, torch_dtype=torch.float16).to(
    ppo_trainer.accelerator.device        
)

accelerator = Accelerator()
in_features = usefulness_model.deberta.embeddings.word_embeddings.embedding_dim
print(f"Max length of usefulness tokenizer: {in_features}")
lambda_model = get_l_models(script_args.lambda_type, in_features=in_features).to(torch.float16)
if script_args.lambda_type == 'constant' and script_args.lambda_value >= 0:
    lambda_model.l.data = torch.tensor(script_args.lambda_value)
lambda_model = lambda_model.to(ppo_trainer.accelerator.device).to(torch.float16)
lambda_optim = torch.optim.SGD(lambda_model.parameters(), lr=script_args.lambda_lr, momentum=0.0)
lambda_model, lambda_optim, _ = accelerator.prepare(lambda_model, lambda_optim, None)

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}
output_min_length = 20
output_max_length = 30
output_length_sampler = LengthSampler(output_min_length, output_max_length)

model_save_path = script_args.model_save_path

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # Get response from the policy model
    response_tensors = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    # Compute sentiment score # noqa
    rewards = []
    if script_args.use_usefulness:
        texts = batch["response"]
        answers = batch["response"]
        usefulness_inputs = usefulness_tokenizer(texts, answers, padding=True, truncation=True, return_tensors="pt").to(
            ppo_trainer.accelerator.device
        )
        logits = usefulness_model(**usefulness_inputs).logits.float()
        usefulness_labels = (logits[:, 0]).tolist()
    
        rewards = [torch.tensor(output) for output in usefulness_labels]
    else:
        rewards = [torch.tensor([0.0]) for _ in batch["response"]]

    # Compute sentiment score # noqa
    constraints = []
    if script_args.use_harmfulness:
        texts = batch["response"]
        toxicity_inputs = toxicity_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
            ppo_trainer.accelerator.device
        )
        logits = toxicity_model(**toxicity_inputs).logits.float()
        toxicity_labels = (logits[:, 0]).tolist()
        # print("labels", toxicity_labels)
    
        lambda_grad = defaultdict(float)
        constraints = []
        for _input, _output in zip(toxicity_inputs['input_ids'], toxicity_labels):
            input_embed = toxicity_model.roberta.embeddings.word_embeddings(_input)
            lambda_optim.zero_grad()
            l = lambda_model(input_embed)
            constraint = l * (_output - script_args.max_constraint)
            constraint.backward()
            # Calculate dy/dw
            for name,param in lambda_model.named_parameters():
                lambda_grad[name] += param.grad
            constraints.append(constraint.detach().clone())
        for key in lambda_grad:
            lambda_grad[key] = lambda_grad[key] / len(toxicity_labels)
        mean_constraints = sum(constraints) / len(toxicity_labels)
        # print("mean_constraint: ",  mean_constraints)
        constraints = [mean_constraints for _ in constraints]
    else:
        constraints = [torch.tensor([0.0]) for _ in batch["response"]]

    # Run PPO step (reward_grad = -do/dy)
    stats, reward_grad = ppo_trainer.step(query_tensors, response_tensors, rewards, constraints)
  
    if script_args.use_harmfulness and script_args.lambda_lr > 0:
        lambda_optim.zero_grad()
        # Update Lambda (w' = w + lr * do/dy * dy/dw)
        with torch.no_grad():
            for name, param in lambda_model.named_parameters():
                # print(reward_grad, lambda_grad)
                param.grad.data = torch.clamp(reward_grad * lambda_grad[name], min=-1, max=1)
    
        lambda_optim.step()
    
        if script_args.lambda_type == 'constant':
            for name, param in lambda_model.named_parameters():
                param.data = torch.clamp(param.data, min=0)

    ppo_trainer.log_stats(stats, batch, rewards)

    # Save model every 100 epochs
    if epoch % 1 == 0:
        if ppo_trainer.accelerator.is_main_process:
            print(f"Epoch {epoch} done")
            ppo_trainer.save_pretrained(model_save_path)
