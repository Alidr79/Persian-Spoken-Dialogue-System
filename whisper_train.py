
#This code is meant to be used for training the whisper model

debug=False

import torch
from torch.utils.data import Dataset, DataLoader
import librosa
from transformers import WhisperFeatureExtractor
from datasets import load_dataset
from datasets import load_from_disk
import datasets
work_dir="pourmand1376/asr-farsi-youtube-chunked-30-seconds"
output_dir = "whisper/model/whisper-small"
common_voice_dir="mozilla-foundation/common_voice_17_0"
model_directory = "openai/whisper-small"

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_directory)
ds = load_from_disk(work_dir)
ds_common = load_from_disk(common_voice_dir).rename_column("sentence", "transcription")
# raise ValueError(next(iter(ds['train'])))
# load the whisper model we want to use
import torch
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
from transformers import WhisperProcessor

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from evaluate import load
import pandas as pd
wer = load("wer")
# processor = WhisperProcessor.from_pretrained(model_directory, language="Persian", task="transcribe",cache_dir='v3')
processor = WhisperProcessor.from_pretrained(model_directory, language="Persian")
model = WhisperForConditionalGeneration.from_pretrained(model_directory)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)



from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained(model_directory, language="Persian")

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


import torch
from torch.utils.data import Dataset, DataLoader
import librosa
from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_directory)

import re

# Sample text
def normalize_text(text):

    text = text.lower()

    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_audio(example):
    target_sr=16000
    audio = example['audio']['array']
    original_sr = example['audio']['sampling_rate']

    if original_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    audio = torch.tensor(audio, dtype=torch.float32)
    
    feature_extract = feature_extractor(audio, sampling_rate=target_sr, return_tensors="pt")['input_features'][0]
    # processed_audio = processor(audio, sampling_rate=target_sr, return_tensors="pt")['input_features']
    text = example['transcription']

    tokenized = tokenizer(text, return_tensors="pt",truncation=True, max_length=200)['input_ids']
    return {'labels':tokenized,'input_features':feature_extract}



import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features) :
        target_sr=16000
        input_features = []
        labels = []
        for example in features:
            audio = example['audio']['array']
            original_sr = example['audio']['sampling_rate']

            if original_sr != target_sr:
                audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
            audio = torch.tensor(audio, dtype=torch.float32)
        
            feature_extract = feature_extractor(audio, sampling_rate=target_sr, return_tensors="pt")['input_features'][0]
            text = example['transcription']
            text = normalize_text(text)
            tokenized = tokenizer(text, return_tensors="pt",truncation=True, max_length=1000)['input_ids']
            input_features.append(feature_extract)
            labels.append(tokenized)
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature} for feature in input_features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        # get the tokenized label sequences
        label_features = [{"input_ids": feature[0]} for feature in labels]
        
        # label_features = [feature["labels"] for feature in features]
        # pad the labels to max length
        # raise ValueError(self.processor)
        labels_batch = self.processor.tokenizer.pad(label_features,return_tensors="pt")
        # labels_batch = tokenizer(label_features, return_tensors="pt", padding="max_length", truncation=True, max_length=200)
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        


        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)
import evaluate

metric = evaluate.load("wer")


# Evaluate with the 'normalized' WER
do_normalize_eval = True

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Replace -100 with the pad_token_id
    label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)

    # Decode
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens = True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens = True)

    if do_normalize_eval:
        pred_str = [normalize_text(s) for s in pred_str]
        label_str = [normalize_text(s) for s in label_str]
    with open("whisper/results.md","a") as f:
        f.write("result:\n")
        for pred,label in zip(pred_str[:20],label_str[:20]):
            f.write(f'{pred}\n')
            f.write(f'{label}\n\n\n')

    print("predicted:",pred_str[0])
    print("label:",label_str[0])
    # Compute the WER
    wer = 100 * metric.compute(predictions = pred_str, references = label_str)
    return {"wer": wer}



# Actually i defined it in previous code cells
def normalize_text(text):
    import string
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False


# from transformers import WhisperTokenizer

# tokenizer = WhisperTokenizer.from_pretrained(model_directory)


from transformers import Seq2SeqTrainingArguments


training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,  # your dir name
    per_device_train_batch_size=16,
    learning_rate=1e-5,
    warmup_steps=500,
    gradient_checkpointing=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps = 1000,
    logging_steps=5,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    num_train_epochs = 2,
    dataloader_num_workers=6,
    remove_unused_columns=False
)


from transformers import Seq2SeqTrainer

subset_indices = list(range(1024))

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=ds_common['train'],
    eval_dataset=ds['val'].select(subset_indices),
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)


trainer.train()

trainer = Seq2SeqTrainer(
    args=training_args,
    model=trainer.model,
    train_dataset=ds['train'],
    eval_dataset=ds['val'].select(subset_indices),
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)


trainer.train()




