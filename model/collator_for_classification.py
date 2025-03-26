import numpy as np
import torch
from typing import Dict, List, Optional, Union
from transformers import SpecialTokensMixin, BatchEncoding

class PrecollatorForCellClassification(SpecialTokensMixin):
    """
    Precollator for single-cell classification.
    Handles padding and processing of tokenized data for cell classification tasks.
    """
    pad_token = "<pad>"
    pad_token_id = 0  # Default pad_token_id
    padding_side = "right"
    model_input_names = ["input_ids"]

    def pad(
        self,
        encoded_inputs: Union[List[Dict[str, List[int]]], Dict[str, List[List[int]]]],
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: bool = True,
        return_tensors: Optional[str] = "pt",
    ) -> BatchEncoding:
        """
        Pads a batch of tokenized inputs to the same length.
        Args:
            encoded_inputs: A batch of tokenized inputs (list of dictionaries or dictionary of lists).
            max_length: Maximum length to pad/truncate sequences.
            pad_to_multiple_of: If set, pad sequences to a multiple of this value.
            return_attention_mask: Whether to generate attention masks.
            return_tensors: Return type of the padded batch (e.g., PyTorch tensors).
        Returns:
            A `BatchEncoding` object containing padded inputs.
        """
        # Convert to dict of lists if input is a list of dicts
        if isinstance(encoded_inputs, list):
            encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

        # The model's main input name must be in the inputs
        if self.model_input_names[0] not in encoded_inputs:
            raise ValueError(
                f"Input batch must include {self.model_input_names[0]}, but got {list(encoded_inputs.keys())}"
            )

        # Get the required input (e.g., input_ids)
        input_ids = encoded_inputs[self.model_input_names[0]]
        batch_size = len(input_ids)
        max_seq_length = max(len(seq) for seq in input_ids) if max_length is None else max_length

        if pad_to_multiple_of is not None:
            max_seq_length = ((max_seq_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

        # Pad input_ids and generate attention masks
        padded_input_ids = []
        attention_masks = []
        for seq in input_ids:
            padding_length = max_seq_length - len(seq)
            if self.padding_side == "right":
                padded_seq = seq + [self.pad_token_id] * padding_length
                attention_mask = [1] * len(seq) + [0] * padding_length
            else:
                padded_seq = [self.pad_token_id] * padding_length + seq
                attention_mask = [0] * padding_length + [1] * len(seq)

            padded_input_ids.append(padded_seq)
            attention_masks.append(attention_mask)

        # Prepare output batch
        batch = {"input_ids": padded_input_ids}
        if return_attention_mask:
            batch["attention_mask"] = attention_masks

        # Convert to tensors if required
        if return_tensors == "pt":
            batch = {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}

        return BatchEncoding(batch)


class DataCollatorForCellClassification:
    """
    Data collator for cell classification.
    Dynamically pads inputs and prepares the batch for cell classification tasks.
    """
    def __init__(self, tokenizer=None, max_length=None, pad_to_multiple_of=None):
        self.tokenizer = PrecollatorForCellClassification()
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Prepares a batch of features by padding inputs and processing labels.
        Args:
            features: A list of dictionaries containing tokenized inputs and labels.
        Returns:
            A dictionary of padded inputs and labels as PyTorch tensors.
        """
        # Separate labels from input features
        labels = [feature.pop("label") for feature in features] if "label" in features[0] else None

        # Pad the inputs
        batch = self.tokenizer.pad(
            features,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Process labels if present
        if labels is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.long)

        return batch