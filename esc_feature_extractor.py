import os
import sys
import numpy as np
import pandas as pd
import librosa
import pywt
import time
import pickle
import warnings
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Set paths - adjusted to work with the proj/ directory structure
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = CURRENT_DIR / "data"
ESC50_DIR = DATA_DIR / "esc50"
AUDIO_DIR = ESC50_DIR / "audio"
META_PATH = ESC50_DIR / "meta" / "esc50.csv"
OUTPUT_DIR = CURRENT_DIR / "datasets"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)


# Feature extraction functions
def extract_zero_crossing_rate(y):
    """Extract Zero Crossing Rate feature."""
    zcr = librosa.feature.zero_crossing_rate(y)
    return {"mean": np.mean(zcr), "std": np.std(zcr), "values": zcr[0]}


def extract_rms_energy(y):
    """Extract Root Mean Square Energy feature."""
    rms = librosa.feature.rms(y=y)
    return {"mean": np.mean(rms), "std": np.std(rms), "values": rms[0]}


def extract_spectral_centroid(y, sr):
    """Extract Spectral Centroid feature."""
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return {
        "mean": np.mean(spectral_centroid),
        "std": np.std(spectral_centroid),
        "values": spectral_centroid[0],
    }


def extract_spectral_bandwidth(y, sr):
    """Extract Spectral Bandwidth feature."""
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    return {
        "mean": np.mean(spectral_bandwidth),
        "std": np.std(spectral_bandwidth),
        "values": spectral_bandwidth[0],
    }


def extract_spectral_rolloff(y, sr):
    """Extract Spectral Rolloff feature."""
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    return {
        "mean": np.mean(spectral_rolloff),
        "std": np.std(spectral_rolloff),
        "values": spectral_rolloff[0],
    }


def extract_mfccs(y, sr, n_mfcc=13):
    """Extract MFCCs feature."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return {
        "mean": np.mean(mfccs, axis=1),
        "std": np.std(mfccs, axis=1),
        "values": mfccs,
    }


def extract_chroma(y, sr):
    """Extract Chroma feature."""
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return {
        "mean": np.mean(chroma, axis=1),
        "std": np.std(chroma, axis=1),
        "values": chroma,
    }


def extract_contrast(y, sr):
    """Extract Spectral Contrast feature."""
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    return {
        "mean": np.mean(contrast, axis=1),
        "std": np.std(contrast, axis=1),
        "values": contrast,
    }


def extract_tonnetz(y, sr):
    """Extract Tonnetz feature."""
    y_harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    return {
        "mean": np.mean(tonnetz, axis=1),
        "std": np.std(tonnetz, axis=1),
        "values": tonnetz,
    }


def extract_dwt(y, wavelet="db4", level=5):
    """Extract Discrete Wavelet Transform feature."""
    coeffs = pywt.wavedec(y, wavelet, level=level)
    dwt_means = [np.mean(np.abs(c)) for c in coeffs]
    dwt_stds = [np.std(np.abs(c)) for c in coeffs]
    return {
        "mean": np.array(dwt_means),
        "std": np.array(dwt_stds),
        "values": coeffs,
    }


def extract_features(y, sr, feature_set):
    """Extract all selected features from an audio signal."""
    features = {}

    # Time-domain features
    if "zcr" in feature_set:
        features["zcr"] = extract_zero_crossing_rate(y)

    if "rms" in feature_set:
        features["rms"] = extract_rms_energy(y)

    # Spectral features
    if "centroid" in feature_set:
        features["spectral_centroid"] = extract_spectral_centroid(y, sr)

    if "bandwidth" in feature_set:
        features["spectral_bandwidth"] = extract_spectral_bandwidth(y, sr)

    if "rolloff" in feature_set:
        features["spectral_rolloff"] = extract_spectral_rolloff(y, sr)

    if "mfcc" in feature_set:
        features["mfccs"] = extract_mfccs(y, sr, n_mfcc=13)

    if "chroma" in feature_set:
        features["chroma"] = extract_chroma(y, sr)

    if "contrast" in feature_set:
        features["contrast"] = extract_contrast(y, sr)

    if "tonnetz" in feature_set:
        features["tonnetz"] = extract_tonnetz(y, sr)

    if "dwt" in feature_set:
        features["dwt"] = extract_dwt(y)

    return features


def create_feature_vector(features, use_mean=True, use_std=True):
    """Create a flattened feature vector from the extracted features."""
    feature_vector = []

    # Add each selected feature to the vector
    for feature_name, feature_data in features.items():
        if use_mean and "mean" in feature_data:
            if isinstance(feature_data["mean"], np.ndarray):
                feature_vector.extend(feature_data["mean"])
            else:
                feature_vector.append(feature_data["mean"])

        if use_std and "std" in feature_data:
            if isinstance(feature_data["std"], np.ndarray):
                feature_vector.extend(feature_data["std"])
            else:
                feature_vector.append(feature_data["std"])

    return np.array(feature_vector)


def get_feature_names(features, use_mean=True, use_std=True):
    """Generate names for each component in the feature vector."""
    feature_names = []

    for feature_name, feature_data in features.items():
        if use_mean and "mean" in feature_data:
            if isinstance(feature_data["mean"], np.ndarray):
                feature_names.extend(
                    [
                        f"{feature_name}_mean_{i}"
                        for i in range(len(feature_data["mean"]))
                    ]
                )
            else:
                feature_names.append(f"{feature_name}_mean")

        if use_std and "std" in feature_data:
            if isinstance(feature_data["std"], np.ndarray):
                feature_names.extend(
                    [f"{feature_name}_std_{i}" for i in range(len(feature_data["std"]))]
                )
            else:
                feature_names.append(f"{feature_name}_std")

    return feature_names


def select_specific_features(X, feature_names):
    """Allow user to select specific features by index."""
    # Display all features with indices
    print("\nAvailable feature components:")
    for i, feature in enumerate(feature_names):
        print(f"{i+1}. {feature}")

    # Let user select features
    print(
        "\nEnter the numbers of feature components to use (e.g., '1 3 5-7' or 'all'):"
    )
    feature_input = input("> ").strip().lower()

    if feature_input == "all":
        # Use all features
        selected_indices = list(range(len(feature_names)))
    else:
        # Parse input like "1 3 5-7" into a list of indices
        selected_indices = []
        parts = feature_input.split()

        for part in parts:
            if "-" in part:
                # Range like "5-7"
                start, end = map(int, part.split("-"))
                selected_indices.extend(range(start - 1, end))
            else:
                # Single number
                selected_indices.append(int(part) - 1)

        # Filter to valid indices only
        selected_indices = [i for i in selected_indices if 0 <= i < len(feature_names)]

        if not selected_indices:
            print("No valid features selected. Using all features.")
            selected_indices = list(range(len(feature_names)))

    # Filter X and feature_names
    X_selected = X[:, selected_indices]
    feature_names_selected = [feature_names[i] for i in selected_indices]

    print(f"\nSelected {len(selected_indices)} feature components")

    return X_selected, feature_names_selected, selected_indices


def process_dataset(dataset_type, feature_set, use_mean=True, use_std=True):
    """Process the ESC dataset and extract features for all audio files."""
    # Read metadata
    metadata = pd.read_csv(META_PATH)

    # Filter for ESC-10 if selected
    if dataset_type == "esc10":
        metadata = metadata[metadata["esc10"] == True]

    print(
        f"\n🔍 Processing {len(metadata)} files from the ESC-{dataset_type.replace('esc', '')} dataset"
    )
    print(f"🧰 Selected features: {', '.join(feature_set)}")

    # Initialize arrays for features and labels
    X = []
    y = []
    file_paths = []

    # Sample file to get feature dimensionality
    first_file = metadata.iloc[0]["filename"]
    first_path = AUDIO_DIR / first_file
    first_audio, sr = librosa.load(first_path, sr=None)
    first_features = extract_features(first_audio, sr, feature_set)
    feature_names = get_feature_names(first_features, use_mean, use_std)

    print(f"📊 Feature vector will have {len(feature_names)} dimensions")

    # Process each file
    start_time = time.time()
    for _, row in tqdm(
        metadata.iterrows(), total=len(metadata), desc="Extracting features"
    ):
        filename = row["filename"]
        file_path = AUDIO_DIR / filename

        try:
            # Load audio
            y_audio, sr = librosa.load(file_path, sr=None)

            # Extract features
            features = extract_features(y_audio, sr, feature_set)

            # Create feature vector
            feature_vector = create_feature_vector(features, use_mean, use_std)

            # Add to dataset
            X.append(feature_vector)
            y.append(row["target"])
            file_paths.append(str(file_path))

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    processing_time = time.time() - start_time
    print(f"✅ Processing completed in {processing_time:.2f} seconds")
    print(f"📊 Feature matrix shape: {X.shape}")

    # Let user select specific feature components
    print("\n🔍 Now you can select specific feature components:")
    X_selected, feature_names_selected, selected_indices = select_specific_features(
        X, feature_names
    )

    print(f"📊 Selected feature matrix shape: {X_selected.shape}")

    # Create dataset info
    dataset_info = {
        "dataset_type": dataset_type,
        "feature_set": feature_set,
        "use_mean": use_mean,
        "use_std": use_std,
        "feature_names": feature_names_selected,
        "original_feature_names": feature_names,
        "selected_indices": selected_indices,
        "shape": X_selected.shape,
        "processing_time": processing_time,
        "file_paths": file_paths,
    }

    return X_selected, y, dataset_info


def save_dataset(X, y, dataset_info):
    """Save the processed dataset to disk."""
    # Create a directory name based on features
    features_str = "_".join(dataset_info["feature_set"])
    stats_str = ""
    if dataset_info["use_mean"]:
        stats_str += "mean"
    if dataset_info["use_std"]:
        stats_str += "std"

    # Add information about selected features
    selection_str = f"{len(dataset_info['feature_names'])}_components"

    dir_name = (
        f"{dataset_info['dataset_type']}_{features_str}_{stats_str}_{selection_str}"
    )
    output_path = OUTPUT_DIR / dir_name
    output_path.mkdir(exist_ok=True)

    # Save X and y
    np.save(output_path / "X.npy", X)
    np.save(output_path / "y.npy", y)

    # Save metadata
    with open(output_path / "metadata.pkl", "wb") as f:
        pickle.dump(dataset_info, f)

    # Also save as CSV for easy inspection
    df = pd.DataFrame(X, columns=dataset_info["feature_names"])
    df["target"] = y
    df.to_csv(output_path / "features.csv", index=False)

    print(f"\n💾 Dataset saved to {output_path}")
    print(f"Files saved:")
    print(f"  - X.npy: Feature matrix ({X.shape})")
    print(f"  - y.npy: Labels ({y.shape})")
    print(f"  - metadata.pkl: Dataset information")
    print(f"  - features.csv: CSV format for easy inspection")

    return str(output_path)


def main():
    """Main function to run the audio feature extractor."""
    print("=" * 60)
    print("🎵 ESC DATASET AUDIO FEATURE EXTRACTOR 🎵")
    print("=" * 60)

    # Check if the dataset exists
    if not AUDIO_DIR.exists() or not META_PATH.exists():
        print(f"❌ Error: ESC-50 dataset not found at {ESC50_DIR}")
        print("Please make sure you've downloaded the dataset.")
        return

    try:
        # Select dataset type
        print("\n📂 Select dataset:")
        print("1. ESC-10 (10 classes)")
        print("2. ESC-50 (50 classes)")
        dataset_choice = input("Enter your choice (1/2): ").strip()

        if dataset_choice == "1":
            dataset_type = "esc10"
        elif dataset_choice == "2":
            dataset_type = "esc50"
        else:
            print("❌ Invalid choice. Defaulting to ESC-50.")
            dataset_type = "esc50"

        # Select features
        print("\n🔍 Select features to extract (enter numbers separated by space):")
        feature_options = [
            ("zcr", "Zero Crossing Rate"),
            ("rms", "Root Mean Square Energy"),
            ("centroid", "Spectral Centroid"),
            ("bandwidth", "Spectral Bandwidth"),
            ("rolloff", "Spectral Rolloff"),
            ("mfcc", "MFCCs (13 coefficients)"),
            ("chroma", "Chroma Features"),
            ("contrast", "Spectral Contrast"),
            ("tonnetz", "Tonnetz"),
            ("dwt", "Discrete Wavelet Transform"),
        ]

        for i, (code, desc) in enumerate(feature_options, 1):
            print(f"{i}. {desc} ({code})")

        feature_choices = input("Enter your choices (e.g., '1 2 6' or 'all'): ").strip()

        if feature_choices.lower() == "all":
            selected_features = [code for code, _ in feature_options]
        else:
            try:
                indices = [int(x) - 1 for x in feature_choices.split()]
                selected_features = [
                    feature_options[i][0]
                    for i in indices
                    if 0 <= i < len(feature_options)
                ]

                if not selected_features:
                    print("❌ No valid features selected. Defaulting to MFCCs.")
                    selected_features = ["mfcc"]
            except:
                print("❌ Invalid input. Defaulting to MFCCs.")
                selected_features = ["mfcc"]

        # Select statistics
        print("\n📊 Select statistics to include:")
        print("1. Mean only")
        print("2. Standard deviation only")
        print("3. Both mean and standard deviation")
        stats_choice = input("Enter your choice (1/2/3): ").strip()

        use_mean = True
        use_std = True

        if stats_choice == "1":
            use_std = False
        elif stats_choice == "2":
            use_mean = False
        elif stats_choice != "3":
            print("❌ Invalid choice. Using both mean and standard deviation.")

        # Process the dataset
        X, y, dataset_info = process_dataset(
            dataset_type, selected_features, use_mean, use_std
        )

        # Save the dataset
        save_path = save_dataset(X, y, dataset_info)

        print(f"\n✅ Feature extraction complete!")
        print(
            f"The extracted features can now be used for machine learning models like SVM."
        )
        print(f"To load this dataset in your machine learning code, use:")
        print(f"\n```python")
        print(f"import numpy as np")
        print(f"import pickle")
        print(f"")
        print(f"# Load feature matrix and labels")
        print(f"X = np.load('{save_path}/X.npy')")
        print(f"y = np.load('{save_path}/y.npy')")
        print(f"")
        print(f"# Load metadata (contains feature names and other info)")
        print(f"with open('{save_path}/metadata.pkl', 'rb') as f:")
        print(f"    metadata = pickle.load(f)")
        print(f"```")

    except Exception as e:
        print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    main()
