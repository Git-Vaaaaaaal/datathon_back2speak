""" 
Machine Learning utilisant les features du son
Lien : https://github.com/musikalkemist/AudioSignalProcessingForML
But : Extraire toutes les features dans un audio et l'utiliser dans des algorithmes de classification (machine_learning) / petit réseau de neurone
 """
import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import IPython.display as ipd
import math





BASE_FOLDER = "../audio_resources/"
sound = os.path.join(BASE_FOLDER, "debussy.wav")

# load audio files with librosa
sound, sr = librosa.load(sound)

#Variable
FRAME_SIZE = 1024
HOP_LENGTH = 512

#Calculer l'amplitude d'enveloppe
def amplitude_envelope(signal, frame_size, hop_length):
    """Calculate the amplitude envelope of a signal with a given frame size nad hop length."""
    amplitude_envelope = []
    
    # calculate amplitude envelope for each frame
    for i in range(0, len(signal), hop_length): 
        amplitude_envelope_current_frame = max(signal[i:i+frame_size]) 
        amplitude_envelope.append(amplitude_envelope_current_frame)
    
    return np.array(amplitude_envelope)    


def fancy_amplitude_envelope(signal, frame_size, hop_length):
    """Fancier Python code to calculate the amplitude envelope of a signal with a given frame size."""
    return np.array([max(signal[i:i+frame_size]) for i in range(0, len(signal), hop_length)])

ae_sound = amplitude_envelope(sound, FRAME_SIZE, HOP_LENGTH)
len(ae_sound)

frames = range(len(ae_sound))
t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)

#Visualize
plt.figure(figsize=(15, 17))

ax = plt.subplot(3, 1, 1)
librosa.display.waveshow(sound, alpha=0.5)
plt.plot(t, ae_sound, color="r")
plt.ylim((-1, 1))
plt.title("Son")

plt.show()


#Root mean square energy (librosa)
rms_sound = librosa.feature.rms(y=sound, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]

#RMSE from scratch
def rmse(signal, frame_size, hop_length):
    rmse = []
    
    # calculate rmse for each frame
    for i in range(0, len(signal), hop_length): 
        rmse_current_frame = np.sqrt(sum(signal[i:i+frame_size]**2) / frame_size)
        rmse.append(rmse_current_frame)
    return np.array(rmse)  

#Zero-crossing rate (librosa)
zcr_sound = librosa.feature.zero_crossing_rate(sound, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]


#Fourier Transform
FRAME_SIZE = 2048
HOP_SIZE = 512

S_sound = librosa.stft(sound, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
Y_scale = np.abs(S_sound) ** 2 #Calcul le spectrogram

def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y, 
                             sr=sr, 
                             hop_length=hop_length, 
                             x_axis="time", 
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")

plot_spectrogram(Y_scale, sr, HOP_SIZE)

#Log-Amplitude Spectrogram
Y_log_scale = librosa.power_to_db(Y_scale)
plot_spectrogram(Y_log_scale, sr, HOP_SIZE)

#Mel spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=sound, sr=sr, n_fft=2048, hop_length=512, n_mels=10)

log_mel_spectrogram = librosa.power_to_db(mel_spectrogram) #logarithm de Mel spectrogram


#Extracting MFCC
mfccs = librosa.feature.mfcc(y=sound, n_mfcc=13, sr=sr)
delta_mfccs = librosa.feature.delta(mfccs) #First MFCCs derivatives
delta2_mfccs = librosa.feature.delta(mfccs, order=2) #Second MFCCs derivatives
mfccs_features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs)) #concatenation


#Calculate band energy ratio
def calculate_split_frequency_bin(split_frequency, sample_rate, num_frequency_bins):
    """Infer the frequency bin associated to a given split frequency."""
    
    frequency_range = sample_rate / 2
    frequency_delta_per_bin = frequency_range / num_frequency_bins
    split_frequency_bin = math.floor(split_frequency / frequency_delta_per_bin)
    return int(split_frequency_bin)

split_frequency_bin = calculate_split_frequency_bin(2000, 22050, 1025)


def band_energy_ratio(spectrogram, split_frequency, sample_rate):
    """Calculate band energy ratio with a given split frequency."""
    
    split_frequency_bin = calculate_split_frequency_bin(split_frequency, sample_rate, len(spectrogram[0]))
    band_energy_ratio = []
    
    # calculate power spectrogram
    power_spectrogram = np.abs(spectrogram) ** 2
    power_spectrogram = power_spectrogram.T
    
    # calculate BER value for each frame
    for frame in power_spectrogram:
        sum_power_low_frequencies = frame[:split_frequency_bin].sum()
        sum_power_high_frequencies = frame[split_frequency_bin:].sum()
        band_energy_ratio_current_frame = sum_power_low_frequencies / sum_power_high_frequencies
        band_energy_ratio.append(band_energy_ratio_current_frame)
    
    return np.array(band_energy_ratio)

ber_debussy = band_energy_ratio(S_sound, 2000, sr)

#Spectral centroid (librosa)
sc_debussy = librosa.feature.spectral_centroid(y=sound, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]

#Spectral bandwith (librosa)
ban_debussy = librosa.feature.spectral_bandwidth(y=sound, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]