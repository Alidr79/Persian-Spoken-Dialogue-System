
# !pip install datasets soundfile speechbrain
# !pip install --upgrade accelerate


import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython import display
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from transformers import SpeechT5HifiGan
from datasets import load_dataset, Audio
from PersianG2p import Persian_g2p_converter


processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

PersianG2Pconverter = Persian_g2p_converter(use_large = True)

ds = load_dataset('SeyedAli/Persian-Speech-Dataset')


# from IPython.display import Audio
# Audio(ds['train'][0]['audio']['array'], rate=16000)


# tokenizer = processor.tokenizer


# def extract_all_chars(batch):
#     all_text = PersianG2Pconverter.transliterate(batch["transcript"][0], tidy = False, secret = True)
#     vocab = list(set(all_text))
#     return {"vocab": [vocab], "all_text": [all_text]}

# vocabs = ds['train'].map(
#     extract_all_chars,
#     batched=True,
#     batch_size=1,  # Don't change this just 1 (faster for PersianG2Pconverter)
#     keep_in_memory=True,
#     remove_columns=ds['train'].column_names,
#     num_proc = 12
# )

# dataset_vocab = set(vocabs["vocab"][0])
# tokenizer_vocab = {k for k,_ in tokenizer.get_vocab().items()}


# tokenizer_vocab


# print(dataset_vocab - tokenizer_vocab)


# Nothing to change
# replacements = [
#     ('ʔ', 'æ'),
#     ('ӕ', 'æ'),
#     ('ʃ', 'sh'),
#     ('\ufeff', ''),
#     ('ɑ', 'a')
# ]


# def cleanup_text(inputs):
#     for src, dst in replacements:
#         inputs["text"] = inputs["text"].replace(src, dst)
#     return inputs

# dataset = ds.map(cleanup_text)


# No id for speakers

# from collections import defaultdict
# speaker_counts = defaultdict(int)

# for speaker_id in dataset['train']["client_id"]:
#     speaker_counts[speaker_id] += 1


# import matplotlib.pyplot as plt

# plt.figure()
# plt.hist(speaker_counts.values(), bins=20)
# plt.ylabel("Speakers")
# plt.xlabel("Examples")
# plt.show()


# len(set(dataset['train']["client_id"]))


# len(dataset['train'])


dataset = ds

# import librosa
# def calculate_duration(audio, sr):
#     return librosa.get_duration(y=audio, sr=sr)

# train_audio_duration_list = [calculate_duration(dataset['train'][i]['audio']['array'], dataset['train'][i]['audio']['sampling_rate'])
#                                  for i in tqdm(range(len(dataset['train'])))]

# val_audio_duration_list = [calculate_duration(dataset['test'][i]['audio']['array'], dataset['test'][i]['audio']['sampling_rate'])
#                                  for i in tqdm(range(len(dataset['test'])))]


# # sort from lowest to highest
# print(f"Longest sample:\nSample index: {np.argsort(train_audio_duration_list)[-1]}, duration: {train_audio_duration_list[np.argsort(train_audio_duration_list)[-1]]} seconds")


# # # Audio(dataset['train'][32636]['audio']['array'], rate=16000)


# # # sort from lowest to highest
# print(f"Shortest sample:\nSample index: {np.argsort(train_audio_duration_list)[0]}, duration: {train_audio_duration_list[np.argsort(train_audio_duration_list)[0]]} seconds")


# # Audio(dataset['train'][26320]['audio']['array'], rate=16000)


import librosa
# import noisereduce as nr


def calculate_duration(audio, sr):
    return librosa.get_duration(y=audio, sr=sr)

# def denoise_audio(audio, sr):
#     # Perform noise reduction
#     denoised_audio = nr.reduce_noise(y=audio, sr=sr)
#     return denoised_audio

def trim_silence(audio, sr, top_db=20):
    # Trim silence from the beginning and end
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed_audio

def remove_short_long(example):
    
    audio = example['audio']['array']
    sr = example['audio']['sampling_rate']
    
    duration = calculate_duration(audio, sr)
    if duration < 1: 
        return False
    
    return True


def preprocess_audio(example):
    
    audio = example['audio']['array']
    sr = example['audio']['sampling_rate']
    
    # Trim silence
    trimmed_audio = trim_silence(audio, sr)
    
    example['audio']['array'] = trimmed_audio
    
    return example


# Apply filtering to the dataset
dataset_bad_duration_removed = dataset.filter(remove_short_long)

print("start trim")
preprocessed_dataset = dataset_bad_duration_removed.map(preprocess_audio, num_proc = 12)


import os
import torch
from speechbrain.inference.speaker import EncoderClassifier

spk_model_name = "speechbrain/spkrec-xvect-voxceleb"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name,
    run_opts={"device": device},
    savedir=os.path.join("/tmp", spk_model_name)
)

def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings


import scipy.io.wavfile as wavfile
def prepare_dataset(example):
    # load the audio data; if necessary, this resamples the audio to 16kHz
    audio = example["audio"]

    input_ipa_text = PersianG2Pconverter.transliterate(example["transcript"],
                                                        tidy = False, secret = True)
    # feature extraction and tokenization
    example = processor(
        text = input_ipa_text,
        audio_target = audio['array'],
        sampling_rate = audio['sampling_rate'],
        return_attention_mask=False,
    )

    # strip off the batch dimension
    example["labels"] = example["labels"][0]

    # use SpeechBrain to obtain x-vector
    example["speaker_embeddings"] = create_speaker_embedding(audio['array'])

    return example





vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")


# spectrogram = torch.tensor(processed_example["labels"])
# with torch.no_grad():
#     speech = vocoder(spectrogram)


# from IPython.display import Audio
# # Audio(speech.cpu().numpy(), rate=16000)

print("prepare data")
preprocessed_dataset = preprocessed_dataset.map(
    prepare_dataset, remove_columns=preprocessed_dataset['train'].column_names)

print('Num Train Examples: ' , len(preprocessed_dataset['train']))
print('Num validation Examples: ' , len(preprocessed_dataset['test']))


from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class TTSDataCollatorWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # collate the inputs and targets into a batch
        batch = processor.pad(
            input_ids=input_ids,
            labels=label_features,
            return_tensors="pt",

        )

        # replace padding with -100 to ignore loss correctly
        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )

        # not used during fine-tuning
        del batch["decoder_attention_mask"]


        # round down target lengths to multiple of reduction factor
        if model.config.reduction_factor > 1:
            target_lengths = torch.tensor([
                len(feature["input_values"]) for feature in label_features
            ])
            target_lengths = target_lengths.new([
                length - length % model.config.reduction_factor for length in target_lengths
            ])
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        # also add in the speaker embeddings
        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        return batch


data_collator = TTSDataCollatorWithPadding(processor=processor)


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'




model.config.use_cache = False




from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer


output_dir = "speecht5_v2_finetuned_derakhshesh"
training_args = Seq2SeqTrainingArguments(
    output_dir = output_dir,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    learning_rate = 1e-5,
    warmup_steps = 1000,
    max_steps = 12_000,
    gradient_checkpointing=True,
    eval_strategy="steps",
    per_device_eval_batch_size = 8,
    save_steps = 2000,
    eval_steps = 500,
    logging_steps = 500,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    greater_is_better=False,
    label_names=["labels"],
    dataloader_num_workers = 5
)



trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=preprocessed_dataset["train"],
    eval_dataset=preprocessed_dataset["test"],
    data_collator=data_collator,
    tokenizer=processor,
)


trainer.train()





processor.save_pretrained("Alidr79/speecht5_v2_finetuned_derakhshesh")





