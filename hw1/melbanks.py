from typing import Optional

import torch
from torch import nn
from torchaudio import functional as F

class LogMelFilterBanks(nn.Module):
    def __init__(
            self,
            n_fft: int = 400,
            samplerate: int = 16000,
            hop_length: int = 160,
            n_mels: int = 80,
            pad_mode: str = 'reflect',
            power: float = 2.0,
            normalize_stft: bool = False,
            onesided: bool = True,
            center: bool = True,
            return_complex: bool = True,
            f_min_hz: float = 0.0,
            f_max_hz: Optional[float] = None,
            norm_mel: Optional[str] = None,
            mel_scale: str = 'htk'
        ):
        super(LogMelFilterBanks, self).__init__()
        self.n_fft = n_fft
        self.samplerate = samplerate
        self.window_length = n_fft
        self.window = torch.hann_window(self.window_length)
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.center = center
        self.return_complex = return_complex
        self.onesided = onesided
        self.normalize_stft = normalize_stft
        self.pad_mode = pad_mode
        self.power = power
        self.f_min_hz = f_min_hz
        self.f_max_hz = f_max_hz if f_max_hz is not None else samplerate // 2
        self.norm_mel = norm_mel
        self.mel_scale = mel_scale
        self.mel_fbanks = self._init_melscale_fbanks()

    def _init_melscale_fbanks(self):
        return F.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=self.f_min_hz,
            f_max=self.f_max_hz,
            n_mels=self.n_mels,
            sample_rate=self.samplerate,
            norm=self.norm_mel,
            mel_scale=self.mel_scale
        )

    def spectrogram(self, x):
        # Ensure the window tensor is on the same device as the input tensor
        self.window = self.window.to(x.device)
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalize_stft,
            onesided=self.onesided,
            return_complex=self.return_complex
        )

    def forward(self, x):
        """
        Args:
            x (Torch.Tensor): Tensor of audio of dimension (batch, time), audiosignal
        Returns:
            Torch.Tensor: Tensor of log mel filterbanks of dimension (batch, n_mels, n_frames),
                where n_frames is a function of the window_length, hop_length and length of audio
        """
        specs = self.spectrogram(x)
        
        if self.return_complex:
            specs = torch.abs(specs) ** self.power
        else:
            specs = specs.pow(2)
        
        if len(specs.shape) == 4:
            specs = specs[..., 0] + specs[..., 1]
        
        batch_size = specs.shape[0]
        specs_t = specs.transpose(1, 2)
        mel_fbanks_expanded = self.mel_fbanks.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        mel_specs_t = torch.bmm(specs_t, mel_fbanks_expanded)
        mel_specs = mel_specs_t.transpose(1, 2)
        log_mel_specs = torch.log(mel_specs + 1e-6)
        
        return log_mel_specs

    def to(self, *args, **kwargs):
        # Move the module to the specified device
        self = super().to(*args, **kwargs)
        # Ensure the window tensor is moved to the same device
        self.window = self.window.to(*args, **kwargs)
        self.mel_fbanks = self.mel_fbanks.to(*args, **kwargs)
        return self