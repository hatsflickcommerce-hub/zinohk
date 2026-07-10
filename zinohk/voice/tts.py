"""
zinohk.voice.tts
=================
ZINOHK Voice Generation — text → speech from scratch.

Pipeline (all ZINOHK-native, no third-party TTS):
  text
    ↓ phoneme encoder
  phoneme sequence
    ↓ spike encoder (GAP-05)
  spike pattern
    ↓ mel predictor (Hebbian trained)
  mel spectrogram (80 bands, ~100 frames/sec)
    ↓ vocoder (Griffin-Lim, no neural net)
  waveform (.wav)

Training:
  - Feed (text, mel) pairs
  - Hebbian: text spikes that co-occur with mel frames → strengthen
  - No backprop anywhere in the pipeline

Why Griffin-Lim vocoder:
  - No neural net — pure signal processing
  - Converts mel spectrogram → waveform mathematically
  - Quality improves with more iterations

Phase 14 upgrade path:
  - Replace Griffin-Lim with ZINOHK neural vocoder
  - Train on LibriSpeech for natural voice quality
"""

import numpy as np
import re
from typing import List, Optional, Tuple


# ── Phoneme table ─────────────────────────────────────────────
# Simple English text → phoneme approximation
# Full version needs a pronunciation dictionary (CMUdict)

PHONEME_MAP = {
    'a': 'AE', 'b': 'B',  'c': 'K',  'd': 'D',  'e': 'EH',
    'f': 'F',  'g': 'G',  'h': 'HH', 'i': 'IH', 'j': 'JH',
    'k': 'K',  'l': 'L',  'm': 'M',  'n': 'N',  'o': 'OW',
    'p': 'P',  'q': 'K',  'r': 'R',  's': 'S',  't': 'T',
    'u': 'UH', 'v': 'V',  'w': 'W',  'x': 'KS', 'y': 'Y',
    'z': 'Z',
    'th': 'TH', 'sh': 'SH', 'ch': 'CH', 'ph': 'F',
    'wh': 'W',  'ck': 'K',  'ng': 'NG', 'qu': 'KW',
}

ALL_PHONEMES = sorted(set(PHONEME_MAP.values()))
PHONEME_IDX  = {p: i for i, p in enumerate(ALL_PHONEMES)}
N_PHONEMES   = len(ALL_PHONEMES)


def text_to_phonemes(text: str) -> List[str]:
    """
    Convert text to approximate phoneme sequence.

    Example
    -------
    >>> text_to_phonemes("hello")
    ['HH', 'EH', 'L', 'L', 'OW']
    """
    text   = text.lower().strip()
    text   = re.sub(r'[^a-z\s]', '', text)
    phonemes = []

    i = 0
    while i < len(text):
        if text[i] == ' ':
            phonemes.append('SIL')   # silence between words
            i += 1
            continue
        # Try 2-char digraph first
        if i + 1 < len(text) and text[i:i+2] in PHONEME_MAP:
            phonemes.append(PHONEME_MAP[text[i:i+2]])
            i += 2
        elif text[i] in PHONEME_MAP:
            phonemes.append(PHONEME_MAP[text[i]])
            i += 1
        else:
            i += 1

    return phonemes


def phonemes_to_spikes(
    phonemes: List[str],
    n_nodes:  int = 256,
) -> np.ndarray:
    """
    Convert phoneme sequence to spike matrix.

    Returns (n_frames, n_nodes) spike matrix.
    Each phoneme generates ~5 frames.
    """
    frames_per_phoneme = 5
    n_frames = len(phonemes) * frames_per_phoneme
    spikes   = np.zeros((n_frames, n_nodes), dtype=np.float32)

    for i, phoneme in enumerate(phonemes):
        # Each phoneme activates a sparse set of nodes
        if phoneme == 'SIL':
            continue   # silence = all zeros

        idx = PHONEME_IDX.get(phoneme, 0)
        # Primary activation
        start = (idx * 7) % n_nodes
        for j in range(frames_per_phoneme):
            frame_idx = i * frames_per_phoneme + j
            # Activate ~5% of nodes per frame
            active = [(start + k * 13) % n_nodes
                      for k in range(max(1, n_nodes // 20))]
            for node in active:
                # Temporal envelope — louder in middle
                env = 1.0 - abs(j - frames_per_phoneme//2) / frames_per_phoneme
                spikes[frame_idx, node] = env

    return spikes


class MelPredictor:
    """
    Predicts mel spectrogram from spike patterns.

    Hebbian trained: when spike pattern X co-occurs
    with mel frame Y, strengthen X→Y connection.

    Parameters
    ----------
    n_nodes  : number of spike nodes (input dim)
    n_mels   : number of mel frequency bands (output dim)
    lr       : Hebbian learning rate
    """

    def __init__(
        self,
        n_nodes: int = 256,
        n_mels:  int = 80,
        lr:      float = 0.01,
    ):
        self.n_nodes = n_nodes
        self.n_mels  = n_mels
        self.lr      = lr

        # Decoder weight matrix: spikes → mel
        self.W = np.random.uniform(
            0.0, 0.1, (n_nodes, n_mels)).astype(np.float32)

        self.n_updates = 0

    def predict_frame(self, spike_frame: np.ndarray) -> np.ndarray:
        """
        Predict one mel frame from one spike frame.

        Parameters
        ----------
        spike_frame : (n_nodes,) spike activation

        Returns
        -------
        mel_frame : (n_mels,) mel spectrogram frame
        """
        raw = spike_frame @ self.W   # (n_mels,)
        # Normalise to mel range (typically log scale)
        if raw.max() > 0:
            raw = raw / raw.max() * 80.0   # scale to dB range
        return raw

    def predict(self, spikes: np.ndarray) -> np.ndarray:
        """
        Predict full mel spectrogram from spike matrix.

        Parameters
        ----------
        spikes : (n_frames, n_nodes) spike matrix

        Returns
        -------
        mel : (n_mels, n_frames) mel spectrogram
        """
        n_frames = spikes.shape[0]
        mel      = np.zeros((self.n_mels, n_frames),
                             dtype=np.float32)
        for i in range(n_frames):
            mel[:, i] = self.predict_frame(spikes[i])
        return mel

    def learn(
        self,
        spike_frame: np.ndarray,
        mel_frame:   np.ndarray,
    ) -> None:
        """
        Hebbian update: strengthen spike→mel connections.

        When spike node i is active and mel band j is loud:
          W[i, j] += lr * spike[i] * mel[j]
        """
        # Normalise mel frame to [0,1]
        mel_norm = mel_frame / (mel_frame.max() + 1e-8)
        self.W  += self.lr * np.outer(spike_frame, mel_norm)
        self.W   = np.clip(self.W, 0.0, 1.0)
        self.n_updates += 1


class GriffinLimVocoder:
    """
    Griffin-Lim algorithm: mel spectrogram → waveform.

    Pure signal processing, no neural net.
    Quality improves with more iterations.

    Parameters
    ----------
    sample_rate : audio sample rate (default 22050 Hz)
    n_fft       : FFT size
    hop_length  : hop between frames
    n_mels      : number of mel bands
    n_iter      : Griffin-Lim iterations (more = better quality)
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft:       int = 1024,
        hop_length:  int = 256,
        n_mels:      int = 80,
        n_iter:      int = 32,
    ):
        self.sample_rate = sample_rate
        self.n_fft       = n_fft
        self.hop_length  = hop_length
        self.n_mels      = n_mels
        self.n_iter      = n_iter

        # Build mel filterbank
        self.mel_fb = self._build_mel_filterbank()

    def _build_mel_filterbank(self) -> np.ndarray:
        """Build mel filterbank matrix."""
        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        def mel_to_hz(mel):
            return 700 * (10**(mel / 2595) - 1)

        fmin     = 0.0
        fmax     = self.sample_rate / 2
        mel_min  = hz_to_mel(fmin)
        mel_max  = hz_to_mel(fmax)
        mel_pts  = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_pts   = mel_to_hz(mel_pts)
        bin_pts  = np.floor(
            (self.n_fft + 1) * hz_pts / self.sample_rate
        ).astype(int)

        fb = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        for m in range(1, self.n_mels + 1):
            f_left  = bin_pts[m - 1]
            f_center= bin_pts[m]
            f_right = bin_pts[m + 1]

            for k in range(f_left, f_center):
                if f_center != f_left:
                    fb[m-1, k] = (k - f_left) / (f_center - f_left)
            for k in range(f_center, f_right):
                if f_right != f_center:
                    fb[m-1, k] = (f_right - k) / (f_right - f_center)

        return fb.astype(np.float32)

    def mel_to_linear(self, mel_spec: np.ndarray) -> np.ndarray:
        """Convert mel spectrogram to linear spectrogram."""
        # Pseudo-inverse of mel filterbank
        mel_spec = np.maximum(mel_spec, 1e-10)
        fb_T     = self.mel_fb.T
        linear   = fb_T @ mel_spec
        return np.maximum(linear, 1e-10)

    def griffin_lim(self, linear_spec: np.ndarray) -> np.ndarray:
        """
        Griffin-Lim algorithm: magnitude → waveform.

        Iteratively estimates phase to reconstruct signal.
        """
        n_freq, n_frames = linear_spec.shape

        # Start with random phase
        phase        = np.random.uniform(0, 2*np.pi,
                                         (n_freq, n_frames))
        complex_spec = linear_spec * np.exp(1j * phase)

        for _ in range(self.n_iter):
            # Inverse STFT
            audio = self._istft(complex_spec)
            # Forward STFT — crop/pad to match n_frames
            stft  = self._stft(audio)
            # Match frames
            min_frames = min(stft.shape[1], n_frames)
            phase = np.angle(stft[:, :min_frames])
            # Pad if needed
            if min_frames < n_frames:
                pad   = np.zeros((n_freq, n_frames - min_frames))
                phase = np.concatenate([phase, pad], axis=1)
            complex_spec = linear_spec * np.exp(1j * phase)

        return self._istft(complex_spec)

    def _stft(self, audio: np.ndarray) -> np.ndarray:
        """Short-time Fourier transform."""
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
        stft     = np.zeros(
            (self.n_fft//2 + 1, n_frames), dtype=complex)
        window   = np.hanning(self.n_fft)

        for i in range(n_frames):
            start  = i * self.hop_length
            frame  = audio[start:start+self.n_fft]
            if len(frame) < self.n_fft:
                frame = np.pad(frame, (0, self.n_fft - len(frame)))
            stft[:, i] = np.fft.rfft(frame * window)

        return stft

    def _istft(self, stft: np.ndarray) -> np.ndarray:
        """Inverse short-time Fourier transform."""
        n_frames = stft.shape[1]
        length   = n_frames * self.hop_length + self.n_fft
        audio    = np.zeros(length)
        window   = np.hanning(self.n_fft)
        norm     = np.zeros(length)

        for i in range(n_frames):
            frame  = np.fft.irfft(stft[:, i])[:self.n_fft]
            start  = i * self.hop_length
            audio[start:start+self.n_fft] += frame * window
            norm[start:start+self.n_fft]  += window**2

        norm  = np.maximum(norm, 1e-8)
        audio = audio / norm
        return audio.astype(np.float32)

    def synthesize(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        Convert mel spectrogram to audio waveform.

        Parameters
        ----------
        mel_spec : (n_mels, n_frames) mel spectrogram

        Returns
        -------
        audio : (n_samples,) waveform at sample_rate Hz
        """
        linear = self.mel_to_linear(mel_spec)
        audio  = self.griffin_lim(linear)
        # Normalise
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max() * 0.9
        return audio


class ZINOHKVoice:
    """
    Complete ZINOHK TTS pipeline.

    text → phonemes → spikes → mel → waveform

    No third-party TTS. All ZINOHK-native.

    Parameters
    ----------
    n_nodes     : spike nodes
    n_mels      : mel frequency bands
    sample_rate : output audio sample rate

    Example
    -------
    >>> voice = ZINOHKVoice()
    >>> audio = voice.speak("hello world")
    >>> voice.save(audio, "output.wav")
    """

    def __init__(
        self,
        n_nodes:     int = 256,
        n_mels:      int = 80,
        sample_rate: int = 22050,
    ):
        self.n_nodes     = n_nodes
        self.n_mels      = n_mels
        self.sample_rate = sample_rate

        self.mel_predictor = MelPredictor(n_nodes, n_mels)
        self.vocoder       = GriffinLimVocoder(
            sample_rate=sample_rate,
            n_mels=n_mels,
        )

        self.n_spoken = 0

    def speak(self, text: str) -> np.ndarray:
        """
        Convert text to audio waveform.

        Parameters
        ----------
        text : input text

        Returns
        -------
        audio : (n_samples,) numpy array at sample_rate Hz
        """
        self.n_spoken += 1

        # Step 1: text → phonemes
        phonemes = text_to_phonemes(text)

        # Step 2: phonemes → spike matrix
        spikes = phonemes_to_spikes(phonemes, self.n_nodes)

        # Step 3: spikes → mel spectrogram
        mel = self.mel_predictor.predict(spikes)

        # Step 4: mel → waveform
        audio = self.vocoder.synthesize(mel)

        return audio

    def save(self, audio: np.ndarray, path: str) -> None:
        """Save audio as WAV file (no external library needed)."""
        import struct, wave
        audio_int = (audio * 32767).astype(np.int16)
        with wave.open(path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(
                struct.pack(f'<{len(audio_int)}h', *audio_int))
        print(f"✅ Saved: {path} "
              f"({len(audio)/self.sample_rate:.2f}s, "
              f"{self.sample_rate}Hz)")

    def stats(self) -> dict:
        return {
            'n_nodes'    : self.n_nodes,
            'n_mels'     : self.n_mels,
            'sample_rate': self.sample_rate,
            'n_spoken'   : self.n_spoken,
            'mel_updates': self.mel_predictor.n_updates,
        }

    def __repr__(self) -> str:
        return (f"ZINOHKVoice(nodes={self.n_nodes}, "
                f"mels={self.n_mels}, "
                f"sr={self.sample_rate})")
