import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython import display
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from transformers import SpeechT5HifiGan
from datasets import load_from_disk, Audio
from PersianG2p import Persian_g2p_converter
from torch.multiprocessing import freeze_support
import multiprocessing
import warnings
import os

import librosa
import os
import torch
from speechbrain.inference.speaker import EncoderClassifier
import scipy.io.wavfile as wavfile

from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer


@dataclass
class TTSDataCollatorWithPadding:
    processor: Any
    PersianG2Pconverter: Any
    model: Any
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        target_sr = 16000
        input_id_list = []
        speaker_embedding_list = []
        label_list = []
        
        for example in features:
            audio = example['audio']['array']
            original_sr = example['audio']['sampling_rate']
            phonemes = self.PersianG2Pconverter.transliterate(example["transcription"], tidy=False, secret=True)
            if original_sr != target_sr:
                audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
            audio = torch.tensor(audio, dtype=torch.float32)
            out_put = self.processor(
                        text = phonemes,
                        audio_target = audio,
                        sampling_rate = target_sr,
                        return_attention_mask=False,
                    )
            input_id = out_put['input_ids']
            label = out_put["labels"][0]
            speaker_embedding = self.create_speaker_embedding(audio)
            input_id_list.append(input_id)
            label_list.append(label)
            speaker_embedding_list.append(speaker_embedding)
        input_ids = [{"input_ids": feature} for feature in input_id_list]
        label_features = [{"input_values": feature} for feature in label_list]
        speaker_features = [feature for feature in speaker_embedding_list]
        # collate the inputs and targets into a batch
        batch = self.processor.pad(
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
        if self.model.config.reduction_factor > 1:
            target_lengths = torch.tensor([
                len(feature["input_values"]) for feature in label_features
            ])
            target_lengths = target_lengths.new([
                length - length % self.model.config.reduction_factor for length in target_lengths
            ])
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]
        # also add in the speaker embeddings

        speaker_features = np.array(speaker_features)
        batch["speaker_embeddings"] = torch.tensor(speaker_features)
        return batch

    def create_speaker_embedding(self, waveform):
        spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
        device = "cuda"
        speaker_model = EncoderClassifier.from_hparams(
            source=spk_model_name,
            run_opts={"device": device},
            savedir=os.path.join("/tmp", spk_model_name)
        )
        with torch.no_grad():
            speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
            speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
            speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
        return speaker_embeddings

def main():
    warnings.filterwarnings("ignore")


    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")


    dataset = load_from_disk('/media/external_2T/malekahmadi/speechprocess_final_project/youtube_pourmand')


    tokenizer = processor.tokenizer




    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    PersianG2Pconverter = Persian_g2p_converter(use_large = False)


    data_collator = TTSDataCollatorWithPadding(
        processor=processor,
        PersianG2Pconverter=PersianG2Pconverter,
        model = model
    )





    model.config.use_cache = False






    output_dir = "speecht5_v3_finetuned_derakhshesh"
    training_args = Seq2SeqTrainingArguments(
        output_dir = output_dir,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # 4*2 = 8
        learning_rate = 1e-5,
        warmup_steps = 500,
        max_steps = 16_000,
        gradient_checkpointing=True,
        eval_strategy="steps",
        per_device_eval_batch_size = 4,
        save_steps = 2000,
        eval_steps = 2000,
        logging_steps = 2000,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        greater_is_better=False,
        label_names=["labels"],
        remove_unused_columns=False,
        dataloader_num_workers = 8
    )

# Ali Notes on timing with multi GPUs
# 8 , 2
# 16 : 2min : 23
# 4 : 2 : 23
# none : 2 : 8

# 2, 4
# 20 processors : 2 min : 54
# 8 : 2 min : 45

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"].select(range(int(len(dataset['val']) * 0.3))),
        data_collator=data_collator,
        tokenizer=processor,
    )



    trainer.train()





    processor.save_pretrained("Alidr79/speecht5_processor_v3_derakhshesh")





if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    freeze_support()
    main()