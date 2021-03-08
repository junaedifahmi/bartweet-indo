import shutil
import logging
from pathlib import Path
from tokenizers.implementations import ByteLevelBPETokenizer

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

vocab_size = 52_000
min_freq = 5
special_tokens = [
    "<s>",
    "</s>",
    "<unk>",
    "<sep>",
    "<pad>",
    "<cls>",
    "<mask>",
    "[USER]",
    "[URL]"

]

output = Path('../result/tokenizer')
if output.is_dir():
    shutil.rmtree(output)
output.mkdir()


if __name__ == '__main__':
    paths = [str(x) for x in Path('../data/raw').glob('**/*.txt')]

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=paths,
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=special_tokens
    )
    tokenizer.save_model(str(output))
    print("Model is saved at", output.absolute().resolve())

