import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import IPython.display as ipd
import math

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

BASE_FOLDER = "F:\\back2speak\\audio_db"
sound_01 = os.path.join(BASE_FOLDER, "P04_S01_W_I05.wav") #37, F
sound_02 = os.path.join(BASE_FOLDER, "P03_S01_W_I05.wav") #38, m
sound_03 = os.path.join(BASE_FOLDER, "P04_S04_W_I05.wav") #5, M
sound_04 = os.path.join(BASE_FOLDER, "P04_S03_W_I05.wav") #8, M
sound_05 = os.path.join(BASE_FOLDER, "P09_S02_W_I05.wav") #13, F

sound_list = [sound_01, sound_02, sound_03, sound_04, sound_05]



#Variable
FRAME_SIZE = 1024
HOP_LENGTH = 512


    
def afficher_spectrogramme(fichier_audio, display=False):
    # Chargement du fichier audio
    y, sr = librosa.load(fichier_audio)

    # Transformation de Fourier (STFT)
    D = librosa.stft(y)

    # Conversion en décibels
    S_db = librosa.amplitude_to_db(abs(D), ref=np.max)

    # Affichage
    if display==True:
        plt.figure(figsize=(10, 5))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Spectrogramme")
        plt.xlabel("Temps (s)")
        plt.ylabel("Fréquence (Hz)")
        plt.tight_layout()
        plt.show()
    
    
afficher_spectrogramme(sound_01)

for i in sound_list : 
    # load audio files with librosa
    sound, sr = librosa.load(i)

    #Extracting MFCC
    mfccs = librosa.feature.mfcc(y=sound, n_mfcc=13, sr=sr)
    delta_mfccs = librosa.feature.delta(mfccs) #First MFCCs derivatives
    delta2_mfccs = librosa.feature.delta(mfccs, order=2) #Second MFCCs derivatives
    mfccs_features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs)) #concatenation

    # Plot MFCC
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mfccs, sr=sr, hop_length=HOP_LENGTH, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

    # Plot Delta MFCC
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(delta_mfccs, sr=sr, hop_length=HOP_LENGTH, x_axis='time')
    plt.colorbar()
    plt.title('Delta MFCC')
    plt.tight_layout()
    plt.show()

    # Plot Delta² MFCC
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(delta2_mfccs, sr=sr, hop_length=HOP_LENGTH, x_axis='time')
    plt.colorbar()
    plt.title('Delta² MFCC')
    plt.tight_layout()
    plt.show()
    
