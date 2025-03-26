import os
import subprocess
import datetime
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments
import multiprocessing
import torch
from model import MambaCellClassifier, DataCollatorForCellClassification

GPU_ID = 0
NUM_PROC = multiprocessing.cpu_count() // 2


# 数据准备 --------------------------------------------------
def load_and_prepare_data():
    # 加载数据集
    train_dataset = load_from_disk("data/cardiomyocyte.dataset")

    # 过滤心肌细胞
    def filter_cardiomyocyte(example):
        return example["cell_type"].startswith("Cardiomyocyte")

    target_names = ["nf", "hcm", "dcm"]
    valid_diseases = set(target_names)

    filtered_data = train_dataset.filter(filter_cardiomyocyte, num_proc=NUM_PROC)
    filtered_data = filtered_data.filter(
        lambda x: x["disease"] in valid_diseases,
        num_proc=NUM_PROC,
        desc="filter invalid sample"
    )

    label_map = {name: idx for idx, name in enumerate(target_names)}

    def encode_labels(example):
        example["label"] = label_map[example["disease"]]
        return example

    labeled_data = filtered_data.map(encode_labels, num_proc=NUM_PROC)

    # 按个体划分数据集
    all_individuals = list(set(labeled_data["individual"]))
    random.seed(42)
    train_indiv = random.sample(all_individuals, int(0.7 * len(all_individuals)))
    remaining = [i for i in all_individuals if i not in train_indiv]
    valid_indiv = random.sample(remaining, int(0.15 * len(all_individuals)))
    test_indiv = [i for i in remaining if i not in valid_indiv]

    # 数据集拆分
    train_data = labeled_data.filter(lambda x: x["individual"] in train_indiv, num_proc=NUM_PROC)
    valid_data = labeled_data.filter(lambda x: x["individual"] in valid_indiv, num_proc=NUM_PROC)
    test_data = labeled_data.filter(lambda x: x["individual"] in test_indiv, num_proc=NUM_PROC)

    return train_data, valid_data, test_data


def model_init():

    vocab_size = 35426
    return MambaCellClassifier(
        vocab_size=vocab_size,
        num_labels=3,
        d_model=128,
        n_layer=6,
        freeze_backbone=True,
    )

# 训练配置 --------------------------------------------------
def get_training_args(params):
    return TrainingArguments(
        output_dir="/tmp",
        evaluation_strategy="epoch",
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"] * 2,
        num_train_epochs=params["epochs"],
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        warmup_steps=params["warmup_steps"],
        lr_scheduler_type=params["lr_scheduler_type"],
        fp16=True,
        gradient_accumulation_steps=2,
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=4,
        no_cuda=False,
        remove_unused_columns=False
    )

def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}


def train(params_list):
    torch.cuda.set_device(GPU_ID)
    torch.backends.cudnn.benchmark = True

    # 加载数据
    train_data, valid_data, _ = load_and_prepare_data()

    results = []
    for i, params in enumerate(params_list):
        # 显存清理
        torch.cuda.empty_cache()

        # 初始化训练参数
        training_args = get_training_args(params)

        # 数据整理器
        data_collator = DataCollatorForCellClassification(
            max_length=2048,
            pad_to_multiple_of=2
        )

        # 初始化Trainer
        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=valid_data,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # 训练与评估
        print(f"\n{'=' * 40}")
        print(f"start {i + 1}/{len(params_list)}")
        print(f"params: {params}")
        train_result = trainer.train()
        eval_result = trainer.evaluate()

        # 记录结果
        results.append({
            "params": params,
            "train_loss": train_result.training_loss,
            "eval_accuracy": eval_result["accuracy"]
        })

        print(f"\ntest {i + 1} result:")
        print(f"loss: {train_result.training_loss:.4f}")
        print(f"accc: {eval_result['accuracy']:.2%}")
        print(f"GPU Memory usage: {torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB")

    return results


if __name__ == "__main__":

    hyperparameter_combinations = [
        {
            "learning_rate": 3e-4,
            "weight_decay": 0.1,
            "batch_size": 32,
            "epochs": 10,
            "warmup_steps": 1000,
            "lr_scheduler_type": "cosine",
        },
        {
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "batch_size": 64,
            "epochs": 15,
            "warmup_steps": 500,
            "lr_scheduler_type": "linear",
        }
    ]

    final_results = train(hyperparameter_combinations)


    for idx, res in enumerate(final_results):
        print(f"\ntest {idx + 1}:")
        print(f"lr: {res['params']['learning_rate']:.0e}")
        print(f"acc: {res['eval_accuracy']:.2%}")
        print(f"loss: {res['train_loss']:.4f}")