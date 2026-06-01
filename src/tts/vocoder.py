import io
import torch
import torchaudio
import librosa
from vocos import Vocos

# global model instance, call load_model() before use
vocos = None 


def load_model():
    '''Loads model which can be called by below functions'''
    global vocos
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")  # downloads weights on first call
    vocos.eval()


def get_mel_vocos(audio_bytes):
    '''
    Get mel spectrogram in correct form form vocos model

    L1 loss (MAE) should be used when training models: F.l1_loss(pred_mel, target_mel)
    Apparently L2 loss (MSE) produces blurry output
    '''
    if vocos is None:
        raise RuntimeError("Model not loaded — call tts.vocoder.load_model() first")
    y, sr = torchaudio.load(io.BytesIO(audio_bytes))  # decode audio bytes
    if y.size(0) > 1:
        y = y.mean(dim=0, keepdim=True)  # mix to mono
    y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)  # resample to 24kHz
    with torch.no_grad():
        return vocos.feature_extractor(y)  # (1, 100, T) log-mel spectrogram


def mel_to_audio_vocos(mel):
    '''Converts mel to audio using vocos model'''
    if vocos is None:
        raise RuntimeError("Model not loaded — call tts.vocoder.load_model() first")
    with torch.no_grad():
        return vocos.decode(mel).squeeze().numpy()  # (N,) float32 waveform


def mel_to_audio_random_phase(mel, sr=24000, n_fft=1024, hop_length=256):
    '''Convert mel to audio via random phase'''
    if vocos is None:
        raise RuntimeError("Model not loaded — call load_model() first")
    
    # get the exact filterbank matrix torchaudio used
    fb = vocos.feature_extractor.mel_spec.mel_scale.fb.numpy()  # (freq, n_mels)
    
    mel_amp = torch.exp(mel).squeeze(0).numpy()  # undo log, (n_mels, T)
    
    # invert filterbank via pseudoinverse
    fb_pinv = np.linalg.pinv(fb.T)  # (n_mels, freq) -> invert to (freq, n_mels)
    mag = np.maximum(0, fb_pinv @ mel_amp)  # (freq, T)
    
    phase = np.exp(1j * 2 * np.pi * np.random.rand(*mag.shape))  # random phase
    audio = librosa.istft(mag * phase, hop_length=hop_length, win_length=n_fft)
    return audio.astype(np.float32)  # (N,) float32 waveform