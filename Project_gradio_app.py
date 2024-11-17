#This code is mean to deploy a gradio webapp. Please note that you nginx to be abale to this. Please visit here for more information
# https://www.gradio.app/guides/running-gradio-on-your-web-server-with-nginx


from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import torch
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from transformers import SpeechT5HifiGan
from transformers import AutoProcessor, AutoModelForTextToSpectrogram
from speechbrain.inference.classifiers import EncoderClassifier
import os
import noisereduce as nr
from pydub import AudioSegment
from PersianG2p import Persian_g2p_converter
from scipy.io import wavfile
import soundfile as sf
import librosa
from unsloth import FastLanguageModel
import torch
import gradio as gr


def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(1)

##########              ##########
#          Load Models
##########             ##########

########## Whisper
model_directory = "/home/taha/speechprocess_final_project/whisper/model/best_whisper_yet"
processor = WhisperProcessor.from_pretrained(model_directory, language="Persian")
whisper_model = WhisperForConditionalGeneration.from_pretrained(model_directory)
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper_model.to(device)

tts_processor = AutoProcessor.from_pretrained("/home/taha/speechprocess_final_project/Alidr79/speecht5_processor_v2_derakhshesh")
tts_model = AutoModelForTextToSpectrogram.from_pretrained("/home/taha/speechprocess_final_project/speecht5_v2/checkpoint-4000")





##########              ##########
#          TTS Model
##########             ##########
spk_model_name = "speechbrain/spkrec-xvect-voxceleb"

speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name,
    run_opts={"device": device},
    savedir=os.path.join("/tmp", spk_model_name),
)

def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings

vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
PersianG2Pconverter = Persian_g2p_converter(use_large = True)

def denoise_audio(audio, sr):
    # Perform noise reduction
    denoised_audio = nr.reduce_noise(y=audio, sr=sr)
    return denoised_audio


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def tts_function(slider_value, input_text):
    audio_embedding,sample_rate_embedding = librosa.load(f'/home/taha/speechprocess_final_project/dialogue_sample_voices/sample_{str(slider_value)}.wav')
    if sample_rate_embedding!=16_000:
        audio_embedding = librosa.resample(audio_embedding, orig_sr=sample_rate_embedding, target_sr=16_000)
    # assert sample_rate_embedding == 16_000

    len_audio = len(audio_embedding)/16_000
    
    with torch.no_grad():
        speaker_embedding = create_speaker_embedding(audio_embedding)
        speaker_embedding = torch.tensor(speaker_embedding).unsqueeze(0)

    phonemes = PersianG2Pconverter.transliterate(input_text, tidy = False, secret = True)

    # text = "</s>"
    # for i in phonemes.replace(' .', '').split(" "):
    #     text += i + " <pad> "

    # text += "</s>"

    text = phonemes

    print("sentence phonemes:", text)

    with torch.no_grad():
        inputs = tts_processor(text = text, return_tensors="pt")

    with torch.no_grad():
        spectrogram = tts_model.generate_speech(inputs["input_ids"], speaker_embedding, minlenratio = 2, maxlenratio = 4, threshold = 0.3)

    with torch.no_grad():
        speech = vocoder(spectrogram)

    speech = speech.numpy().reshape(-1)
    speech_denoised = denoise_audio(speech, 16000)
    sf.write("in_speech.wav", speech_denoised, 16000)

    sound = AudioSegment.from_wav("in_speech.wav", "wav")
    normalized_sound = match_target_amplitude(sound, -20.0)
    normalized_sound.export("out_sound.wav", format="wav")

    sample_rate_out, audio_out = wavfile.read("out_sound.wav")

    assert sample_rate_out == 16_000

    return 16000, (audio_out.reshape(-1)).astype(np.int16)




##########              ##########
#          LLAMA Model
##########             ##########

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

model.load_state_dict(torch.load("trained_llama3_on_persian_high_train.pt"))


##########              ##########
#        PUT ALL TO GATHER
######### + Gradio APP  ##########


history = []
def master_function(slider_value, user_voice):

    # load model and processor
    
    sr,audio = user_voice

    audio = librosa.resample(audio.astype(np.float32) / 32768.0, orig_sr=sr, target_sr=16000)
    # sf.write("output_path.wav",audio, 16000)
    # print("Audio saved")
    input_features = processor(torch.tensor(audio), sampling_rate=16000, return_tensors="pt").input_features.cuda()

    # generate token ids
    predicted_ids = whisper_model.generate(input_features)
    # decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    print("1. transcription: ", transcription)
    # alpaca_prompt = Copied from above
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    inputs = tokenizer(
    [
        alpaca_prompt.format(
            transcription[0], # instruction
            "", # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 256, use_cache = True)
    llama_response = tokenizer.batch_decode(outputs)
    llama_response = llama_response[0].split('Response')[-1].split('<|end_of_text|>')[0][2:]
    print("2. LLAMA response: ", llama_response)


    all_speech = []
    for sentence in llama_response.split("."):
        sampling_rate_response, audio_chunk_response = tts_function(slider_value, sentence)
        all_speech.append(audio_chunk_response)
        
    audio_response = np.concatenate(all_speech)
    assert sampling_rate_response == 16_000
    return sampling_rate_response, audio_response

slider = gr.Slider(
    minimum=1,
    maximum=102,
    value=8,
    step=1,
    label="Select a speaker"
)

text_input = gr.Textbox(
    label="Enter some text",
    placeholder="Type something here..."
)

demo = gr.Interface(
    fn = master_function,
    inputs=[slider, "audio"],  # List of inputs
    outputs = "audio"
)

demo.queue().launch(root_path="/gradio-demo", share = True)