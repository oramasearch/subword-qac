import os
import sentencepiece as spm

INPUT = "data/aol/full/train.query.txt"
SPM_DIR = "spm"
MAX_NUM_SENTS = str(100000000)
VOCAB_SIZE = str(256)
MODEL_TYPES = ["char", "unigram", "bpe"]

# Create spm directory if it doesn't exist
os.makedirs(SPM_DIR, exist_ok=True)

# Train other models
for MODEL_TYPE in MODEL_TYPES:
    sub_folder = "spm" if MODEL_TYPE == "char" else VOCAB_SIZE
    model_dir = os.path.join(SPM_DIR, MODEL_TYPE, sub_folder)

    if os.path.exists(model_dir):
        print(f"Folder {model_dir} exists; skipping!")
        continue
    os.makedirs(model_dir)

    model_path = os.path.join(model_dir, "spm")

    train_command = f"--input={INPUT} --input_sentence_size={MAX_NUM_SENTS} --model_type=char --model_prefix={model_path}"

    if MODEL_TYPE != "char":
        train_command = f"{train_command} --vocab_size={VOCAB_SIZE}"
    spm.SentencePieceTrainer.Train(train_command)

print("All done!")
