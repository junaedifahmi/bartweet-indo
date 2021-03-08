from pathlib import Path

from transformers import DataCollatorForSeq2Seq

from datasets import load_dataset
from datasets import logging

logging.set_verbosity_warning()


def get_train_test(train, test, tokenizer):
    padding = "max_length"
    ignore_pad_token_for_loss = True
    label_pad_token_id = -100 if ignore_pad_token_for_loss else tokenizer.pad_token_id

    def preprocess_function(examples):
        inputs = examples['input_text']
        targets = examples['target_text']
        model_inputs = tokenizer(inputs, max_length=256, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=256, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset = load_dataset('csv', delimiter='|', data_files={'train': train, 'test': test})
    dataset = dataset.map(
        preprocess_function,
        batched=True
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=label_pad_token_id
    )

    return dataset, data_collator
