import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. Charger + spectrogramme
# ===============================
def mel_spectrogram(fichier_audio):
    y, sr = librosa.load(fichier_audio)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db, sr


# ===============================
# 2. Détection phonèmes
# ===============================
def detecter_phonemes(S_db, seuil_ch=-40, seuil_s=-40, seuil_voyelle=-45):
    phonemes = []

    for t in range(S_db.shape[1]):
        colonne = S_db[:, t]

        energy_s = np.mean(colonne[80:120])
        energy_ch = np.mean(colonne[60:100])
        energy_v = np.mean(colonne[10:50])

        # Détection combinée robuste
        if energy_s > seuil_s and energy_ch > seuil_ch:
            if abs(energy_s - energy_ch) < 5:
                phonemes.append("/sʃ/")
            elif energy_s > energy_ch:
                phonemes.append("/s/")
            else:
                phonemes.append("/ʃ/")
        elif energy_s > seuil_s:
            phonemes.append("/s/")
        elif energy_ch > seuil_ch:
            phonemes.append("/ʃ/")
        elif energy_v > seuil_voyelle:
            phonemes.append("/V/")
        else:
            phonemes.append("-")

    return phonemes


# ===============================
# 3. Lissage (anti bruit)
# ===============================
def lisser_phonemes(phonemes, window_size=5):
    phonemes_lisses = []

    for i in range(len(phonemes)):
        start = max(0, i - window_size // 2)
        end = min(len(phonemes), i + window_size // 2 + 1)

        window = phonemes[start:end]
        phoneme_majoritaire = max(set(window), key=window.count)

        phonemes_lisses.append(phoneme_majoritaire)

    return phonemes_lisses


# ===============================
# 4. Regrouper phonèmes
# ===============================
def regrouper_phonemes(phonemes, sr, hop_length=512):
    segments = []
    current = phonemes[0]
    start = 0

    for i in range(1, len(phonemes)):
        if phonemes[i] != current:
            if current != "-":
                t_start = start * hop_length / sr
                t_end = i * hop_length / sr
                segments.append((current, t_start, t_end))
            current = phonemes[i]
            start = i

    if current != "-":
        t_start = start * hop_length / sr
        t_end = len(phonemes) * hop_length / sr
        segments.append((current, t_start, t_end))

    return segments


# ===============================
# 5. Affichage
# ===============================
def afficher_avec_phonemes(S_db, sr, phonemes, hop_length=512):
    plt.figure(figsize=(12,5))

    # Spectrogramme
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')

    # Regroupement simple
    segments = []
    current = phonemes[0]
    start = 0

    for i in range(1, len(phonemes)):
        if phonemes[i] != current:
            segments.append((current, start, i))
            current = phonemes[i]
            start = i
    segments.append((current, start, len(phonemes)))

    # Bandes colorées
    for p, start, end in segments:
        t_start = start * hop_length / sr
        t_end = end * hop_length / sr

        if p == "/ʃ/":
            plt.axvspan(t_start, t_end, color='#00FF00', alpha=0.3)
        elif p == "/V/":
            plt.axvspan(t_start, t_end, color='#FFFFFF', alpha=0.2)
        elif p == "/s/":
            plt.axvspan(t_start, t_end, color='#00FFFF', alpha=0.3)
        elif p == "/sʃ/":
            plt.axvspan(t_start, t_end, color='#FFA500', alpha=0.4)

    plt.title("Spectrogramme + phonèmes")
    plt.tight_layout()
    plt.show()


# ===============================
# 6. Diagnostic
# ===============================
def detecter_erreur(segments):
    if len(segments) == 0:
        return "Aucun son détecté"

    premier = segments[0][0]

    if premier in ["/s/", "/sʃ/"]:
        return "❌ Ajout d’un /s/ avant le /ʃ/"
    
    if premier != "/ʃ/":
        return "❌ Le mot ne commence pas par /ʃ/"

    return "✅ Prononciation correcte"


# ===============================
# 7. MAIN
# ===============================
fichier = r'pre_processing\BAD_elle_colorie_la_vache.WAV'

# Spectrogramme
S_db, sr = mel_spectrogram(fichier)

print("Max spectrogramme :", np.max(S_db))
print("Min spectrogramme :", np.min(S_db))

# Détection
phonemes = detecter_phonemes(S_db)

# Lissage
phonemes = lisser_phonemes(phonemes)

# Affichage
afficher_avec_phonemes(S_db, sr, phonemes)

# Segments
segments = regrouper_phonemes(phonemes, sr)

print("\nPhonèmes détectés :")
for p, t1, t2 in segments:
    print(f"{p} de {t1:.2f}s à {t2:.2f}s")

# Diagnostic
print("\nDiagnostic :")
print(detecter_erreur(segments))


# ===============================
# 8. Fréquences réelles
# ===============================
freqs_hz = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=sr/2)

print("\nPlages fréquentielles :")
print("V:", freqs_hz[10], freqs_hz[50])
print("ʃ:", freqs_hz[60], freqs_hz[100])
print("s:", freqs_hz[80], freqs_hz[120])
print("sʃ:", freqs_hz[60], freqs_hz[120])