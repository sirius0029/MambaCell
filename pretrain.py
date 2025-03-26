from model import MambaCell,MambaCellPretrainer
import datetime
import os
import pickle
import random
import subprocess

import numpy as np
import pytz
import torch
from datasets import load_from_disk
from transformers import TrainingArguments

seed_num = 0
random.seed(seed_num)
np.random.seed(seed_num)
seed_val = 42
torch.manual_seed(seed_val)
num_layers=12
timezone = pytz.timezone("US/Eastern")
rootdir = "/parent_ouput_directory"

os.environ["CONDA_OVERRIDE_GLIBC"] = "2.56"


with open("token_dictionary.pkl", "rb") as fp:
    token_dictionary = pickle.load(fp)

seed_num = 0
random.seed(seed_num)
np.random.seed(seed_num)
seed_val = 42
torch.manual_seed(seed_val)

timezone = pytz.timezone("US/Eastern")
rootdir = "./parent_ouput_directory"


num_examples = 27_406_208
# number gpus
num_gpus = 2
# batch size for training and eval
batch_size = 12
# max learning rate
max_lr = 1e-4
# learning schedule
lr_schedule_fn = "linear"
# warmup steps
warmup_steps = 10_000
# number of epochs
epochs = 3
# optimizer
optimizer = "adamw"
# weight_decay
weight_decay = 0.001

max_seq_length=2048

# output directories
current_date = datetime.datetime.now(tz=timezone)
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}_{current_date.strftime('%X').replace(':','')}"
run_name = f"{datestamp}_geneMamba{num_layers}_SL{max_seq_length}_E{epochs}_B{batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_O{optimizer}_DS{num_gpus}"
training_output_dir = f"{rootdir}/models/{run_name}/"
logging_dir = f"{rootdir}/runs/{run_name}/"
model_output_dir = os.path.join(training_output_dir, "models/")

# ensure not overwriting previously saved model
model_output_file = os.path.join(model_output_dir, "pytorch_model.bin")
if os.path.isfile(model_output_file) is True:
    raise Exception("Model already saved to this directory.")

# make training and model output directories
subprocess.call(f"mkdir {training_output_dir}", shell=True)
subprocess.call(f"mkdir {model_output_dir}", shell=True)


model = MambaCell(
        vocab_size=25426,
        d_model=768,
        n_layer=12,
        d_state=16,
        d_conv=4,
        expand=2,
        max_seq_length=2048,
        dropout=0.1,
        task='joint',

    )


if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model = model.to('cuda')
model = model.train()

# define the training arguments
training_args = {
    "learning_rate": max_lr,
    "do_train": True,
    "do_eval": False,
    "group_by_length": True,
    "length_column_name": "length",
    "disable_tqdm": False,
    "lr_scheduler_type": lr_schedule_fn,
    "warmup_steps": warmup_steps,
    "weight_decay": weight_decay,
    "per_device_train_batch_size": batch_size,
    "num_train_epochs": epochs,
    "save_strategy": "steps",
    "save_steps": np.floor(
        num_examples / batch_size / 8
    ),  # 8 saves per epoch
    "logging_steps": 1000,
    "output_dir": training_output_dir,
    "logging_dir": logging_dir,
     "remove_unused_columns":False, 
    "max_grad_norm": 1.0,
}
training_args = TrainingArguments(**training_args)

print("Starting training.")
    
# define the trainer
trainer = MambaCellPretrainer(
    model=model,
    args=training_args,
    # pretraining corpus (e.g. https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/genecorpus_30M_2048.dataset)
    train_dataset=load_from_disk("genecorpus_30M_2048.dataset"),
    # file of lengths of each example cell (e.g. https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/blob/main/genecorpus_30M_2048_lengths.pkl)
    example_lengths_file="genecorpus_30M_2048_lengths.pkl",
    token_dictionary=token_dictionary,
    task='joint',
    alpha=0.7,  
    beta=0.3,
)

# train
trainer.train()

# save model
trainer.save_model(model_output_dir)

