from get_dataset import get_train_test
from utils import wer

from transformers import logging
from transformers import BartConfig, BartTokenizerFast, BartForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

logging.set_verbosity_info()

vocab_path = '../result/tokenizer/vocab.json'
merge_file = '../result/tokenizer/merges.txt'
train_file = '../data/clean/train.csv'
test_file = '../data/clean/test.csv'


if __name__ == '__main__':
    tokenizer = BartTokenizerFast(
        vocab_file=vocab_path,
        merges_file=merge_file,
    )
    bartconfig = BartConfig(
        vocab_size=tokenizer.vocab_size
    )
    model = BartForConditionalGeneration(config=bartconfig)
    dataset, cullator = get_train_test(train_file, test_file, tokenizer)

    args = Seq2SeqTrainingArguments(
        output_dir='../result/bart',
        do_train=True,
        num_train_epochs=10
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        data_collator=cullator,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer,
        compute_metrics=wer
    )
    trainer.train()

