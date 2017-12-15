# coding: utf-8

import librosa
import numpy as np



def read_audio_spectrum(filename, n_fft = 2048):
	x, sr = librosa.load(filename)
	S = librosa.stft(x, n_fft)
	
	S = np.log1p(np.abs(S[:,:430]))

	return S, sr
