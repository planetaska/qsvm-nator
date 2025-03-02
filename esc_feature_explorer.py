import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import librosa
import librosa.display
import sounddevice as sd
from pathlib import Path
import time
import warnings
import pywt  # Add PyWavelets library

warnings.filterwarnings("ignore")

# Set paths - adjusted to work with the proj/ directory structure
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = CURRENT_DIR / "data"
ESC50_DIR = DATA_DIR / "esc50"
AUDIO_DIR = ESC50_DIR / "audio"
META_PATH = ESC50_DIR / "meta" / "esc50.csv"


def load_random_audio():
    """Load a random audio file from the ESC-50 dataset and its metadata."""
    # Read metadata
    metadata = pd.read_csv(META_PATH)

    # Select a random file
    random_row = metadata.sample(1).iloc[0]
    filename = random_row["filename"]
    category = random_row["category"]
    target_class = random_row["target"]

    # Load audio file
    file_path = AUDIO_DIR / filename
    y, sr = librosa.load(file_path, sr=None)  # Keep original sampling rate

    print(f"\n🔊 Selected audio: {filename}")
    print(f"🏷️  Category: {category} (class {target_class})")
    print(f"📊 Duration: {len(y)/sr:.2f} seconds")
    print(f"📈 Sampling rate: {sr} Hz")

    return y, sr, random_row


def play_audio(y, sr):
    """Play the audio file."""
    print("\n▶️  Playing audio... (press Ctrl+C to stop)")
    try:
        sd.play(y, sr)
        time.sleep(len(y) / sr)
        sd.stop()
    except KeyboardInterrupt:
        sd.stop()
        print("⏹️  Playback stopped.")
    except Exception as e:
        print(f"❌ Error playing audio: {e}")
        print("⚠️  Make sure you have audio output configured correctly.")


def extract_features(y, sr):
    """Extract audio features useful for classification."""
    features = {}

    # Time-domain features

    # 1. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features["zcr"] = {"mean": np.mean(zcr), "std": np.std(zcr), "values": zcr[0]}

    # 2. Root Mean Square Energy
    rms = librosa.feature.rms(y=y)
    features["rms"] = {"mean": np.mean(rms), "std": np.std(rms), "values": rms[0]}

    # Spectral features

    # 3. Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spectral_centroid"] = {
        "mean": np.mean(spectral_centroid),
        "std": np.std(spectral_centroid),
        "values": spectral_centroid[0],
    }

    # 4. Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features["spectral_bandwidth"] = {
        "mean": np.mean(spectral_bandwidth),
        "std": np.std(spectral_bandwidth),
        "values": spectral_bandwidth[0],
    }

    # 5. Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features["spectral_rolloff"] = {
        "mean": np.mean(spectral_rolloff),
        "std": np.std(spectral_rolloff),
        "values": spectral_rolloff[0],
    }

    # 6. Mel-frequency cepstral coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features["mfccs"] = {
        "mean": np.mean(mfccs, axis=1),
        "std": np.std(mfccs, axis=1),
        "values": mfccs,
    }

    # 7. Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features["chroma"] = {
        "mean": np.mean(chroma, axis=1),
        "std": np.std(chroma, axis=1),
        "values": chroma,
    }

    # 8. Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features["contrast"] = {
        "mean": np.mean(contrast, axis=1),
        "std": np.std(contrast, axis=1),
        "values": contrast,
    }

    # 9. Tonnetz (tonal centroid features)
    y_harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    features["tonnetz"] = {
        "mean": np.mean(tonnetz, axis=1),
        "std": np.std(tonnetz, axis=1),
        "values": tonnetz,
    }

    # 10. Continuous Wavelet Transform
    # Generate CWT using 'morl' (Morlet) wavelet, a common choice for audio analysis
    widths = np.arange(1, 128)
    cwt_coeffs, cwt_freqs = compute_cwt(y, widths)
    features["cwt"] = {
        "mean": np.mean(np.abs(cwt_coeffs), axis=1),
        "std": np.std(np.abs(cwt_coeffs), axis=1),
        "values": cwt_coeffs,
        "freqs": cwt_freqs,
    }

    # 11. Discrete Wavelet Transform
    coeffs = pywt.wavedec(y, "db4", level=5)
    dwt_means = [np.mean(np.abs(c)) for c in coeffs]
    dwt_stds = [np.std(np.abs(c)) for c in coeffs]
    features["dwt"] = {
        "mean": np.array(dwt_means),
        "std": np.array(dwt_stds),
        "values": coeffs,
    }

    return features


def compute_cwt(y, widths):
    """Compute the Continuous Wavelet Transform."""
    # Use Morlet wavelet which is good for time-frequency analysis
    cwt_coeffs = pywt.cwt(y, widths, "morl")[0]

    # Calculate approximate frequency for each scale
    # This is a rough approximation based on the central frequency of Morlet wavelet
    central_freq = pywt.central_frequency("morl")
    cwt_freqs = central_freq / (widths * (1 / len(y)))

    return cwt_coeffs, cwt_freqs


def print_feature_stats(features):
    """Print feature statistics to the command line."""
    print("\n📊 Feature Statistics:")
    print("-" * 50)

    # Print time-domain features
    print("⏱️  Time-domain features:")
    print(
        f"  Zero Crossing Rate: mean={features['zcr']['mean']:.4f}, std={features['zcr']['std']:.4f}"
    )
    print(
        f"  RMS Energy: mean={features['rms']['mean']:.4f}, std={features['rms']['std']:.4f}"
    )

    # Print spectral features
    print("\n📊 Spectral features:")
    print(
        f"  Spectral Centroid: mean={features['spectral_centroid']['mean']:.4f}, std={features['spectral_centroid']['std']:.4f}"
    )
    print(
        f"  Spectral Bandwidth: mean={features['spectral_bandwidth']['mean']:.4f}, std={features['spectral_bandwidth']['std']:.4f}"
    )
    print(
        f"  Spectral Rolloff: mean={features['spectral_rolloff']['mean']:.4f}, std={features['spectral_rolloff']['std']:.4f}"
    )

    # Print MFCCs summary
    print("\n🔢 MFCCs:")
    for i, (mean, std) in enumerate(
        zip(features["mfccs"]["mean"], features["mfccs"]["std"])
    ):
        print(f"  MFCC {i+1}: mean={mean:.4f}, std={std:.4f}")

    # Print Chroma features summary
    print("\n🎵 Chroma features:")
    for i, (mean, std) in enumerate(
        zip(features["chroma"]["mean"], features["chroma"]["std"])
    ):
        print(f"  Chroma {i+1}: mean={mean:.4f}, std={std:.4f}")

    # Print Spectral Contrast summary
    print("\n🔊 Spectral Contrast:")
    for i, (mean, std) in enumerate(
        zip(features["contrast"]["mean"], features["contrast"]["std"])
    ):
        print(f"  Band {i+1}: mean={mean:.4f}, std={std:.4f}")

    # Print Tonnetz summary
    print("\n🎹 Tonnetz (tonal centroid features):")
    for i, (mean, std) in enumerate(
        zip(features["tonnetz"]["mean"], features["tonnetz"]["std"])
    ):
        print(f"  Component {i+1}: mean={mean:.4f}, std={std:.4f}")

    # Print Wavelet Transform summary
    print("\n🌊 Wavelet Transform:")
    print("  CWT: First 5 scales mean values:")
    for i in range(min(5, len(features["cwt"]["mean"]))):
        print(
            f"  Scale {i+1}: mean={features['cwt']['mean'][i]:.4f}, std={features['cwt']['std'][i]:.4f}"
        )

    print("\n  DWT: Decomposition levels mean values:")
    level_names = ["Approximation"] + [
        f"Detail {i}" for i in range(1, len(features["dwt"]["mean"]))
    ]
    for i, name in enumerate(level_names):
        print(
            f"  {name}: mean={features['dwt']['mean'][i]:.4f}, std={features['dwt']['std'][i]:.4f}"
        )


def visualize_features(y, sr, features, metadata):
    """Create a comprehensive visualization of all features."""
    # Create a taller figure to accommodate vertical stacking
    plt.figure(figsize=(24, 30))  # Made taller to accommodate wavelet plots

    # Create vertical GridSpec with 11 rows (added 2 for wavelets)
    # Use 3 columns: first for descriptions, second for plots, third for colorbars
    gs = GridSpec(11, 3, figure=plt.gcf(), width_ratios=[0.65, 5, 0.15])

    # Calculate the duration of the audio in seconds
    duration = len(y) / sr

    # Title for the entire plot
    plt.suptitle(
        f"Audio Features: {metadata['filename']} - {metadata['category']} (Duration: {duration:.2f}s)",
        fontsize=16,
    )

    # Descriptions for each feature
    descriptions = {
        "waveform": "Amplitude vs. time representation showing audio pressure fluctuations. Useful for identifying temporal patterns, silence, and loudness variations.",
        "spectrogram": "Time-frequency representation showing energy distribution across frequencies. Helps identify dominant frequencies, harmonics, and temporal changes.",
        "mel_spectrogram": "Spectrogram mapped to the Mel scale (approx human hearing perception). Better represents how humans perceive different frequencies.",
        "mfccs": "Mel-frequency cepstral coefficients (MFCCs) represent the spectral shape of the sound. Each coefficient (y-axis) captures different aspects of the audio at each time frame. Lower coefficients represent overall energy and spectral envelope, while higher ones capture more detailed variations.",
        "chroma": "Represents the distribution of energy across 12 pitch classes (musical notes). Useful for analyzing tonal content while ignoring octave differences.",
        "tonnetz": "Tonal centroid features representing harmonic content in a six-dimensional space. Useful for chord and harmony analysis in music.",
        "zcr": "Zero Crossing Rate measures how often the signal changes sign. Higher for noisier sounds (e.g., consonants, percussion) and lower for tonal sounds.",
        "rms": "Root Mean Square Energy tracks amplitude envelope over time. Indicates loudness variations and can identify percussive events.",
        "spectral": "Spectral features describe the\nshape and characteristics of the spectrum:\n- Centroid: spectral center of mass\n- Bandwidth: spread around centroid\n- Rolloff: frequency below which 85%\n  of energy is concentrated",
        "cwt": "Continuous Wavelet Transform provides multi-resolution time-frequency analysis. Reveals both time and frequency information simultaneously, particularly useful for transient features.",
        "dwt": "Discrete Wavelet Transform decomposes the signal into approximation and detail coefficients at different levels. Useful for capturing both time and frequency characteristics efficiently.",
    }

    # 1. Waveform
    # Description on the left
    ax_desc1 = plt.subplot(gs[0, 0])
    ax_desc1.text(
        0.85,
        0.5,
        descriptions["waveform"],
        va="center",
        ha="right",
        wrap=True,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"
        ),
    )
    ax_desc1.axis("off")

    # Plot on the right
    ax1 = plt.subplot(gs[0, 1])
    librosa.display.waveshow(y, sr=sr, ax=ax1)
    ax1.set_title("Waveform")
    ax1.set_xlim(0, duration)
    # Hide x-axis labels and remove x-label for plots except the last one
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_xlabel("")

    # 2. Spectrogram
    # Description on the left
    ax_desc2 = plt.subplot(gs[1, 0])
    ax_desc2.text(
        0.85,
        0.5,
        descriptions["spectrogram"],
        va="center",
        ha="right",
        wrap=True,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"
        ),
    )
    ax_desc2.axis("off")

    # Plot on the right
    ax2 = plt.subplot(gs[1, 1], sharex=ax1)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, y_axis="log", x_axis="time", sr=sr, ax=ax2)
    ax2.set_title("Spectrogram")

    # Get the current y-axis tick positions and labels
    yticks = ax2.get_yticks()
    ytick_labels = [label.get_text() for label in ax2.get_yticklabels()]

    # Create a subset of ticks: first, last, and every other in between
    if len(yticks) > 2:
        # Always include first and last
        selected_indices = [0]

        # Add every other index for the middle values
        if len(yticks) > 3:
            middle_indices = list(range(1, len(yticks) - 1, 2))
            selected_indices.extend(middle_indices)

        # Always add the last index
        if len(yticks) - 1 not in selected_indices:
            selected_indices.append(len(yticks) - 1)

        # Apply the selected ticks and labels
        ax2.set_yticks([yticks[i] for i in selected_indices])
        ax2.set_yticklabels([ytick_labels[i] for i in selected_indices])

    cbar2 = plt.colorbar(img, cax=plt.subplot(gs[1, 2]), format="%+2.0f dB")
    cbar2.set_label("Power (dB)")
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set_xlabel("")

    # 3. Mel Spectrogram
    # Description on the left
    ax_desc3 = plt.subplot(gs[2, 0])
    ax_desc3.text(
        0.85,
        0.5,
        descriptions["mel_spectrogram"],
        va="center",
        ha="right",
        wrap=True,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"
        ),
    )
    ax_desc3.axis("off")

    # Plot on the right
    ax3 = plt.subplot(gs[2, 1], sharex=ax1)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img3 = librosa.display.specshow(
        mel_spec_db, y_axis="mel", x_axis="time", sr=sr, ax=ax3
    )
    ax3.set_title("Mel Spectrogram")
    cbar3 = plt.colorbar(img3, cax=plt.subplot(gs[2, 2]), format="%+2.0f dB")
    cbar3.set_label("Power (dB)")
    plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.set_xlabel("")

    # 4. MFCCs
    # Description on the left
    ax_desc4 = plt.subplot(gs[3, 0])
    ax_desc4.text(
        0.85,
        0.5,
        descriptions["mfccs"],
        va="center",
        ha="right",
        wrap=True,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"
        ),
    )
    ax_desc4.axis("off")

    # Plot on the right
    ax4 = plt.subplot(gs[3, 1], sharex=ax1)
    mfcc_data = features["mfccs"]["values"]
    img4 = librosa.display.specshow(mfcc_data, x_axis="time", sr=sr, ax=ax4)
    ax4.set_title("MFCCs")
    ax4.set_ylabel("Coeff.")

    # Set y-ticks to show every 3rd coefficient to avoid overlapping
    n_mfccs = mfcc_data.shape[0]
    # Create indices for every 3rd coefficient, always including the first and last
    ytick_indices = np.arange(0, n_mfccs, 3)
    if (n_mfccs - 1) not in ytick_indices:
        ytick_indices = np.append(ytick_indices, n_mfccs - 1)

    ax4.set_yticks(ytick_indices)
    ax4.set_yticklabels([f"{i+1}" for i in ytick_indices])

    # Add minor ticks for all coefficients without labels
    ax4.set_yticks(np.arange(n_mfccs), minor=True)
    ax4.grid(axis="y", which="minor", linestyle=":", linewidth=0.5, alpha=0.3)

    cbar4 = plt.colorbar(img4, cax=plt.subplot(gs[3, 2]))
    cbar4.set_label("Magnitude")
    plt.setp(ax4.get_xticklabels(), visible=False)
    ax4.set_xlabel("")

    # 5. Chroma Features
    # Description on the left
    ax_desc5 = plt.subplot(gs[4, 0])
    ax_desc5.text(
        0.85,
        0.5,
        descriptions["chroma"],
        va="center",
        ha="right",
        wrap=True,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"
        ),
    )
    ax_desc5.axis("off")

    # Plot on the right
    ax5 = plt.subplot(gs[4, 1], sharex=ax1)
    img5 = librosa.display.specshow(
        features["chroma"]["values"], y_axis="chroma", x_axis="time", sr=sr, ax=ax5
    )
    ax5.set_title("Chroma Features")
    cbar5 = plt.colorbar(img5, cax=plt.subplot(gs[4, 2]))
    cbar5.set_label("Magnitude")
    plt.setp(ax5.get_xticklabels(), visible=False)
    ax5.set_xlabel("")

    # 6. Tonnetz
    # Description on the left
    ax_desc6 = plt.subplot(gs[5, 0])
    ax_desc6.text(
        0.85,
        0.5,
        descriptions["tonnetz"],
        va="center",
        ha="right",
        wrap=True,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"
        ),
    )
    ax_desc6.axis("off")

    # Plot on the right
    ax6 = plt.subplot(gs[5, 1], sharex=ax1)
    img6 = librosa.display.specshow(
        features["tonnetz"]["values"], y_axis="tonnetz", x_axis="time", sr=sr, ax=ax6
    )
    ax6.set_title("Tonnetz")
    cbar6 = plt.colorbar(img6, cax=plt.subplot(gs[5, 2]))
    cbar6.set_label("Magnitude")
    plt.setp(ax6.get_xticklabels(), visible=False)
    ax6.set_xlabel("")

    # 7. Zero Crossing Rate
    # Description on the left
    ax_desc7 = plt.subplot(gs[6, 0])
    ax_desc7.text(
        0.85,
        0.5,
        descriptions["zcr"],
        va="center",
        ha="right",
        wrap=True,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"
        ),
    )
    ax_desc7.axis("off")

    # Plot on the right
    ax7 = plt.subplot(gs[6, 1], sharex=ax1)
    frames = range(len(features["zcr"]["values"]))
    t = librosa.frames_to_time(frames, sr=sr)
    ax7.plot(t, features["zcr"]["values"])
    ax7.set_title("Zero Crossing Rate")
    ax7.set_ylabel("ZCR")
    plt.setp(ax7.get_xticklabels(), visible=False)
    ax7.set_xlabel("")

    # 8. RMS Energy
    # Description on the left
    ax_desc8 = plt.subplot(gs[7, 0])
    ax_desc8.text(
        0.85,
        0.5,
        descriptions["rms"],
        va="center",
        ha="right",
        wrap=True,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"
        ),
    )
    ax_desc8.axis("off")

    # Plot on the right
    ax8 = plt.subplot(gs[7, 1], sharex=ax1)
    frames = range(len(features["rms"]["values"]))
    t = librosa.frames_to_time(frames, sr=sr)
    ax8.plot(t, features["rms"]["values"])
    ax8.set_title("RMS Energy")
    ax8.set_ylabel("Energy")
    plt.setp(ax8.get_xticklabels(), visible=False)
    ax8.set_xlabel("")

    # 9. Combined Spectral Features - No longer the last plot, so remove x label
    # Description on the left
    ax_desc9 = plt.subplot(gs[8, 0])
    ax_desc9.text(
        0.85,
        0.5,
        descriptions["spectral"],
        va="center",
        ha="right",
        wrap=True,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"
        ),
    )
    ax_desc9.axis("off")

    # Plot on the right
    ax9 = plt.subplot(gs[8, 1], sharex=ax1)
    frames = range(len(features["spectral_centroid"]["values"]))
    t = librosa.frames_to_time(frames, sr=sr)
    ax9.plot(t, features["spectral_centroid"]["values"], label="Centroid")
    ax9.plot(t, features["spectral_bandwidth"]["values"], label="Bandwidth")
    ax9.plot(t, features["spectral_rolloff"]["values"] / sr * 2, label="Rolloff")
    ax9.set_title("Spectral Features")
    ax9.set_xlabel("")  # Remove the x-axis label
    ax9.set_ylabel("Hz")
    plt.setp(ax9.get_xticklabels(), visible=False)  # Hide x-axis tick labels
    ax9.legend(loc="upper right", bbox_to_anchor=(0.98, 1.0))

    # After the 9th plot (spectral features), add the wavelet plots

    # 10. Continuous Wavelet Transform (CWT)
    # Description on the left
    ax_desc10 = plt.subplot(gs[9, 0])
    ax_desc10.text(
        0.85,
        0.5,
        descriptions["cwt"],
        va="center",
        ha="right",
        wrap=True,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"
        ),
    )
    ax_desc10.axis("off")

    # Plot on the right
    ax10 = plt.subplot(gs[9, 1], sharex=ax1)
    cwt_coeffs = features["cwt"]["values"]
    cwt_freqs = features["cwt"]["freqs"]

    # Get a subset of frequencies to display on y-axis (logarithmically spaced)
    num_yticks = 5
    freq_indices = np.round(
        np.logspace(np.log10(1), np.log10(len(cwt_freqs) - 1), num_yticks)
    ).astype(int)

    img10 = ax10.imshow(
        np.abs(cwt_coeffs),
        aspect="auto",
        origin="lower",
        extent=[0, duration, 0, len(cwt_freqs)],
        cmap="viridis",
    )
    ax10.set_title("Continuous Wavelet Transform (Morlet)")
    ax10.set_ylabel("Hz")

    # Add approximate frequency ticks to y-axis
    ax10.set_yticks(freq_indices)
    # Round frequencies to readable values
    y_tick_labels = [f"{cwt_freqs[i]:.0f}" for i in freq_indices]
    ax10.set_yticklabels(y_tick_labels)

    plt.setp(ax10.get_xticklabels(), visible=False)  # Hide x-axis tick labels
    ax10.set_xlabel("")
    cbar10 = plt.colorbar(img10, cax=plt.subplot(gs[9, 2]))
    cbar10.set_label("Magnitude")

    # 11. Discrete Wavelet Transform (DWT) - This is the last plot, so it should show x-axis label
    # Description on the left
    ax_desc11 = plt.subplot(gs[10, 0])
    ax_desc11.text(
        0.85,
        0.5,
        descriptions["dwt"],
        va="center",
        ha="right",
        wrap=True,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"
        ),
    )
    ax_desc11.axis("off")

    # Plot on the right
    ax11 = plt.subplot(gs[10, 1])
    dwt_coeffs = features["dwt"]["values"]
    # Plot approximation and detail coefficients
    level_names = ["Approx."] + [f"Detail {i}" for i in range(1, len(dwt_coeffs))]
    n_levels = len(dwt_coeffs)

    # Calculate proper y positions for each level
    y_positions = np.linspace(0, 1, n_levels)

    # Plot each level of coefficients
    for i, (coef, name) in enumerate(zip(dwt_coeffs, level_names)):
        # Scale to fit in the plot
        scaled_coef = (
            coef / (np.max(np.abs(coef)) * 2) if np.max(np.abs(coef)) > 0 else coef
        )
        # Center vertically at the appropriate y position
        y_pos = y_positions[i]
        x = np.linspace(0, duration, len(coef))
        ax11.plot(x, scaled_coef + y_pos, label=name)

    ax11.set_title("Discrete Wavelet Transform (db4)")
    ax11.set_xlabel("Time (s)")  # Keep this as the only x-axis label
    ax11.set_yticks(y_positions)
    ax11.set_yticklabels(level_names)
    ax11.set_xlim(0, duration)
    ax11.grid(True, axis="y", linestyle="--", alpha=0.6)

    # Adjust layout - modified to reduce gaps
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(
        hspace=0.4, wspace=0.15, left=0.11, right=0.95, bottom=0.05, top=0.92
    )  # Increased wspace to create more space between description and plot

    # No longer saving the figure
    print(f"\n📊 Displaying audio features visualization")
    plt.show()


def main():
    """Main function to run the audio feature explorer."""
    print("=" * 60)
    print("🎵 ESC-50 AUDIO FEATURE EXPLORER 🎵")
    print("=" * 60)

    # Check if the dataset exists
    if not AUDIO_DIR.exists() or not META_PATH.exists():
        print(f"❌ Error: ESC-50 dataset not found at {ESC50_DIR}")
        print("Please make sure you've downloaded the dataset.")
        return

    try:
        # Load a random audio file
        y, sr, metadata = load_random_audio()

        # Ask user if they want to play audio
        play_sound = (
            input("\nWould you like to play the audio? (Y/n): ").strip().lower()
        )
        if play_sound != "n":
            # Play the audio
            play_audio(y, sr)
        else:
            print("\n🔇 Audio playback skipped.")

        # Extract features
        print("\n⚙️  Extracting audio features...")
        features = extract_features(y, sr)

        # Print feature statistics
        print_feature_stats(features)

        # Visualize features
        print("\n📊 Creating visualizations...")
        visualize_features(y, sr, features, metadata)

    except Exception as e:
        print(f"❌ An error occurred: {e}")

    print("\n✅ Done! To analyze another file, run the script again.")


if __name__ == "__main__":
    main()
