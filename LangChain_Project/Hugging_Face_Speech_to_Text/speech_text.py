from transformers import pipeline
from pydub import AudioSegment
#audio = AudioSegment.from_file('C:\Users\get_p\OneDrive\Documents\Projects\AiProjects\LangChain_Project\Hugging_Face_Speech_to_Text\sample-1.m4a')
# pipeline allow data flow from source to destination system
import transformers
# print(transformers.__version__)
import librosa
import torch
import IPython.display as display
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import numpy as np
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
audio, sample_rate = librosa.load('sample-1.m4a', sr=16000)
audio, sample_rate
#display.Audio(audio, rate=sample_rate)
display.Audio('sample-1.m4a', autoplay=True)
input_values = tokenizer(audio, return_tensors="pt").input_values
input_values
logits = model(input_values).logits
logits
Predicted_ids = torch.argmax(logits, dim=-1)
trnscriptions = tokenizer.batch_decode(Predicted_ids)[0]
trnscriptions

 