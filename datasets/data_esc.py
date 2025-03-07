import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import zipfile
import pandas as pd
import shutil
import librosa
import pywt
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import pickle
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Directory structure
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ESC50_DIR = DATA_DIR / "esc50"
ESC10_DIR = DATA_DIR / "esc10"
AUDIO_DIR = ESC50_DIR / "audio"
META_PATH = ESC50_DIR / "meta" / "esc50.csv"


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
    return {"mean": np.array(dwt_means), "std": np.array(dwt_stds), "values": coeffs}


def _extract_features(y, sr, feature_set):
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


def _create_feature_vector(features, use_mean=True, use_std=True):
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


def _get_feature_names(features, use_mean=True, use_std=True):
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


def _extract_dataset_features(dataset_type, feature_set, use_mean=True, use_std=True):
    """Extract features from the dataset."""
    # Determine which dataset to use
    if dataset_type == "esc10":
        dataset_dir = ESC10_DIR
        audio_dir = ESC10_DIR / "audio"
        meta_path = ESC10_DIR / "meta" / "esc10.csv"
    else:
        dataset_dir = ESC50_DIR
        audio_dir = ESC50_DIR / "audio"
        meta_path = ESC50_DIR / "meta" / "esc50.csv"

    # Create features directory if it doesn't exist
    features_dir = dataset_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata = pd.read_csv(meta_path)

    # Get audio files
    audio_files = list(audio_dir.glob("*.wav"))
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {audio_dir}")

    print(
        f"\n🔍 Processing {len(audio_files)} files from the {dataset_type.upper()} dataset"
    )
    print(f"🧰 Selected features: {', '.join(feature_set)}")

    # Extract features for each file
    features_list = []
    labels = []

    for file_path in tqdm(audio_files, desc="Extracting features"):
        # Get metadata for file
        filename = file_path.name
        file_metadata = metadata[metadata["filename"] == filename]
        if file_metadata.empty:
            continue

        label = file_metadata["target"].values[0]

        # Load audio file
        y, sr = librosa.load(file_path, sr=None)

        # Extract features
        features = _extract_features(y, sr, feature_set)

        # Create feature vector
        feature_vector = _create_feature_vector(features, use_mean, use_std)

        features_list.append(feature_vector)
        labels.append(label)

    # Convert to numpy arrays
    X = np.array(features_list)
    y = np.array(labels)

    # Get feature names
    feature_names = _get_feature_names(features, use_mean, use_std)

    # Create dataset info
    dataset_info = {
        "feature_set": feature_set,
        "feature_names": feature_names,
        "use_mean": use_mean,
        "use_std": use_std,
        "extraction_time": time.time(),
    }

    # Save CSV of features
    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = y
    df.to_csv(features_dir / f"{dataset_type}_features.csv", index=False)

    # Log feature extraction
    start_time = time.time()
    print(f"📊 Feature vector will have {X.shape[1]} dimensions")

    print(f"✅ Processing completed in {time.time() - start_time:.2f} seconds")
    print(f"📊 Feature matrix shape: {X.shape}")

    return X, y, dataset_info


def _download_dataset():
    """Download and organize the ESC-50 dataset."""
    print("This script will:")
    print("1. Create a 'data' directory in the current location (if it doesn't exist)")
    print("2. Download the ESC-50 dataset (~600 MB)")
    print("3. Extract and organize the dataset into ESC-10 and ESC-50 subsets")

    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(exist_ok=True)

    # URL for the ESC-50 dataset
    esc50_url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    zip_path = DATA_DIR / "ESC-50-master.zip"

    # Check if the zip file already exists and is valid
    if zip_path.exists() and _is_valid_zip(zip_path):
        print(f"Valid zip file already exists at {zip_path}. Skipping download.")
    else:
        # Remove the file if it exists but is invalid
        if zip_path.exists():
            print(
                f"Existing zip file at {zip_path} is invalid. Removing and downloading again."
            )
            zip_path.unlink()

        # Download the dataset
        print(f"Downloading ESC-50 dataset from {esc50_url}...")
        urllib.request.urlretrieve(esc50_url, zip_path)
        print(f"Download complete. Saved to {zip_path}")

    # Extract the zip file
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("Extraction complete.")

    # Path to the extracted dataset
    extracted_dir = DATA_DIR / "ESC-50-master"

    # Read the metadata
    metadata_path = extracted_dir / "meta" / "esc50.csv"
    metadata = pd.read_csv(metadata_path)

    # Create directories for ESC-10 and ESC-50 with audio and meta subdirectories
    esc10_audio_dir = ESC10_DIR / "audio"
    esc10_meta_dir = ESC10_DIR / "meta"
    esc50_audio_dir = ESC50_DIR / "audio"
    esc50_meta_dir = ESC50_DIR / "meta"

    # Create all directories
    for directory in [
        ESC10_DIR,
        ESC50_DIR,
        esc10_audio_dir,
        esc10_meta_dir,
        esc50_audio_dir,
        esc50_meta_dir,
    ]:
        directory.mkdir(exist_ok=True)

    # Copy audio files to respective directories
    audio_dir = extracted_dir / "audio"

    # Process all files
    print("Organizing files into ESC-10 and ESC-50 subsets...")
    for _, row in metadata.iterrows():
        filename = row["filename"]
        source_path = audio_dir / filename

        # Copy to ESC-50 directory (all files)
        dest_path_esc50 = esc50_audio_dir / filename
        shutil.copy2(source_path, dest_path_esc50)

        # Copy to ESC-10 directory if it belongs to ESC-10 subset
        if row["esc10"] == 1:
            dest_path_esc10 = esc10_audio_dir / filename
            shutil.copy2(source_path, dest_path_esc10)

    # Copy metadata files
    shutil.copy2(metadata_path, esc50_meta_dir / "esc50.csv")
    shutil.copy2(
        extracted_dir / "meta" / "esc50-human.xlsx", esc50_meta_dir / "esc50-human.xlsx"
    )

    # Create ESC-10 specific metadata
    esc10_metadata = metadata[metadata["esc10"] == 1]
    esc10_metadata.to_csv(esc10_meta_dir / "esc10.csv", index=False)
    shutil.copy2(
        extracted_dir / "meta" / "esc50-human.xlsx", esc10_meta_dir / "esc10-human.xlsx"
    )

    # Copy README to both directories
    shutil.copy2(extracted_dir / "README.md", ESC50_DIR / "README.md")
    shutil.copy2(extracted_dir / "README.md", ESC10_DIR / "README.md")

    print(f"ESC-50 dataset organized successfully:")
    print(f"- Full dataset: {ESC50_DIR} ({len(metadata)} files)")
    print(f"- ESC-10 subset: {ESC10_DIR} ({len(esc10_metadata)} files)")
    print(
        f"- Metadata saved to {esc50_meta_dir/'esc50.csv'} and {esc10_meta_dir/'esc10.csv'}"
    )

    # Remove the extracted folder but keep a copy of the README
    shutil.copy2(extracted_dir / "README.md", DATA_DIR / "ESC-50-README.md")
    print(f"Removing extracted directory {extracted_dir}...")
    shutil.rmtree(extracted_dir)
    print("Cleanup complete.")


def _is_valid_zip(file_path):
    """Check if a file is a valid zip file."""
    try:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            return True
    except zipfile.BadZipFile:
        return False


def _load_features(dataset_type):
    """Load pre-extracted features from cache."""
    # Check if dataset exists, if not download it
    if not (ESC10_DIR.exists() and ESC50_DIR.exists()):
        print("Dataset not found. Downloading...")
        _download_dataset()

    # Determine which dataset to use
    if dataset_type == "esc10":
        meta_path = ESC10_DIR / "meta" / "esc10.csv"
        feature_dir = ESC10_DIR / "features"
    else:
        meta_path = ESC50_DIR / "meta" / "esc50.csv"
        feature_dir = ESC50_DIR / "features"

    # Debug info
    print(f"Looking for features in: {feature_dir}")

    # Check if features directory exists
    if not feature_dir.exists():
        print(f"Feature directory not found at: {feature_dir}")
        feature_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created feature directory at: {feature_dir}")

        print("No feature cache found. Extracting features...")
        feature_set = [
            "zcr",
            "rms",
            "centroid",
            "bandwidth",
            "rolloff",
            "mfcc",
            "chroma",
            "contrast",
            "tonnetz",
            "dwt",
        ]
        X, y, dataset_info = _extract_dataset_features(
            dataset_type, feature_set, use_mean=True, use_std=True
        )

        # Save features to cache
        print(f"Saving features to: {feature_dir}")
        np.save(feature_dir / "X.npy", X)
        np.save(feature_dir / "y.npy", y)
        with open(feature_dir / "metadata.pkl", "wb") as f:
            pickle.dump(dataset_info, f)
    else:
        # Check if feature files exist
        x_path = feature_dir / "X.npy"
        y_path = feature_dir / "y.npy"
        metadata_path = feature_dir / "metadata.pkl"

        if x_path.exists() and y_path.exists() and metadata_path.exists():
            # Load features and metadata from NumPy files
            print(f"Loading cached features from NumPy files: {feature_dir}")
            try:
                X = np.load(x_path)
                y = np.load(y_path)
                with open(metadata_path, "rb") as f:
                    dataset_info = pickle.load(f)
                print(
                    f"✅ Successfully loaded {X.shape[0]} samples with {X.shape[1]} features from NumPy files."
                )
                print(f"✅ Label shape: {y.shape}, Unique labels: {np.unique(y)}")
            except Exception as e:
                print(f"❌ Error loading NumPy files: {str(e)}")
                raise
        else:
            print("Feature files incomplete or not found. Extracting new features...")
            feature_set = [
                "zcr",
                "rms",
                "centroid",
                "bandwidth",
                "rolloff",
                "mfcc",
                "chroma",
                "contrast",
                "tonnetz",
                "dwt",
            ]
            X, y, dataset_info = _extract_dataset_features(
                dataset_type, feature_set, use_mean=True, use_std=True
            )

            # Save features to cache
            print(f"Saving features to: {feature_dir}")
            np.save(feature_dir / "X.npy", X)
            np.save(feature_dir / "y.npy", y)
            with open(feature_dir / "metadata.pkl", "wb") as f:
                pickle.dump(dataset_info, f)

    # Load actual label names from CSV
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found at: {meta_path}")

    metadata = pd.read_csv(meta_path)
    label_names = metadata.groupby("target")["category"].first().to_dict()

    # Add label names to dataset info
    dataset_info["label_names"] = label_names
    print(f"Label names: {label_names}")

    return X, y, dataset_info


def _process_features(X, y, n_components, plot_data=True, label_names=None):
    """
    Process features for quantum classification.

    Args:
        X: Feature matrix
        y: Labels
        n_components: Number of PCA components to keep
        plot_data: Whether to create visualization plots
        label_names: Dictionary mapping label indices to names

    Returns:
        Processed data for quantum algorithm
    """
    try:
        print(f"Starting with X shape {X.shape}, y shape {y.shape}")
        print(f"Unique labels in y: {np.unique(y)}")
        print(f"n_components: {n_components}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=109
        )
        print(
            f"After split - X_train shape {X_train.shape}, X_test shape {X_test.shape}"
        )
        print(
            f"After split - y_train shape {y_train.shape}, y_test shape {y_test.shape}"
        )

        # Standardize features
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        print(
            f"After standardization - X_train shape {X_train.shape}, X_test shape {X_test.shape}"
        )

        # Apply PCA to reduce dimensions
        pca = PCA(n_components=n_components).fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        print(f"After PCA - X_train shape {X_train.shape}, X_test shape {X_test.shape}")

        # Scale to [-1, 1]
        samples = np.append(X_train, X_test, axis=0)
        minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
        X_train = minmax_scale.transform(X_train)
        X_test = minmax_scale.transform(X_test)
        print(
            f"After scaling - X_train shape {X_train.shape}, X_test shape {X_test.shape}"
        )

        # Create training/test input dictionaries using actual label names
        if label_names is None:
            label_names = {i: f"Class_{i}" for i in range(len(np.unique(y)))}

        print(f"Label names: {label_names}")
        print(f"Unique labels in y_train: {np.unique(y_train)}")
        print(f"Unique labels in y_test: {np.unique(y_test)}")

        # Create dictionary with class-specific data
        training_input = {
            label_names[k]: X_train[y_train == k, :] for k in np.unique(y)
        }
        test_input = {label_names[k]: X_test[y_test == k, :] for k in np.unique(y)}

        print(f"Created training_input with keys {list(training_input.keys())}")
        print(f"Created test_input with keys {list(test_input.keys())}")

        if plot_data:
            plt.figure(figsize=(10, 6))
            for k in np.unique(y):
                try:
                    x_axis_data = X_train[y_train == k, 0]
                    y_axis_data = X_train[y_train == k, 1]
                    plt.scatter(x_axis_data, y_axis_data, label=label_names[k])
                except Exception as e:
                    print(f"Warning: Could not plot data for class {k}: {str(e)}")

            plt.title(f"ESC Dataset (Dimensionality Reduced With PCA)")

            # Place legend outside of the plot area
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

            # Add more space on the right for the legend
            plt.tight_layout(rect=[0, 0, 0.85, 1])

            # Save the plot
            plot_path = os.path.join("plots", f"esc_{n_components}d_pca_plot.png")
            plt.savefig(plot_path)
            plt.close()

            print(f"📊 Plot generated and saved to: {os.path.abspath(plot_path)}")

        return X_train, training_input, test_input, list(label_names.values())

    except Exception as e:
        print(f"Error in processing features: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


def esc10(n=2, PLOT_DATA=True):
    """
    Load and process ESC-10 dataset.

    Args:
        n: Number of PCA components to keep (default=2)
        PLOT_DATA: Whether to create visualization plots

    Returns:
        X_train: Processed feature matrix
        training_input: Dictionary with training data by class
        test_input: Dictionary with test data by class
        class_labels: List of class names
    """
    try:
        # Load pre-processed features
        X, y, metadata = _load_features("esc10")

        # Process features
        return _process_features(X, y, n, PLOT_DATA, metadata["label_names"])
    except Exception as e:
        print(f"Error loading ESC-10 dataset: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


def esc50(n=2, PLOT_DATA=True):
    """
    Load and process ESC-50 dataset.

    Args:
        n: Number of PCA components to keep (default=2)
        PLOT_DATA: Whether to create visualization plots

    Returns:
        X_train: Processed feature matrix
        training_input: Dictionary with training data by class
        test_input: Dictionary with test data by class
        class_labels: List of class names
    """
    try:
        # Load pre-processed features
        X, y, metadata = _load_features("esc50")

        # Process features
        return _process_features(X, y, n, PLOT_DATA, metadata["label_names"])
    except Exception as e:
        print(f"Error loading ESC-50 dataset: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


def main():
    """Test the dataset processing functions."""
    print("=" * 60)
    print("🎵 ESC DATASET PROCESSING TEST 🎵")
    print("=" * 60)
    print()

    try:
        print("🔍 Loading ESC10 dataset from pre-processed files...")
        X, y, metadata = _load_features("esc10")
        print(f"✅ Successfully loaded features: X shape {X.shape}, y shape {y.shape}")

        print("\n🔄 Processing features...")
        X_train, training_input, test_input, class_labels = _process_features(
            X, y, n_components=2, plot_data=True, label_names=metadata["label_names"]
        )
        print(f"✅ Successfully processed features")
        print(f"Class labels: {class_labels}")
        print(f"X_train shape: {X_train.shape}")

        # Print example of how to use the dataset
        print("\nUse this file like this:\n")
        print("X_train, training_input, test_input, class_labels = esc10(2)")
        print("X_train, training_input, test_input, class_labels = esc50(2)")

    except Exception as e:
        print(f"❌ An error occurred: {repr(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
