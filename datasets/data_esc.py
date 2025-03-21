import numpy as np
import urllib.request
import zipfile
import pandas as pd
import shutil
import librosa
# import pywt
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# Directory structure
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ESC50_DIR = DATA_DIR / "esc50"
ESC10_DIR = DATA_DIR / "esc10"
AUDIO_DIR = ESC50_DIR / "audio"
META_PATH = ESC50_DIR / "meta" / "esc50.csv"


def extract_features_vector(y, sr):
    """Extract all audio features and return directly as a feature vector."""
    feature_vector = []
    feature_names = []
    
    # Zero Crossing Rate
    # zcr = librosa.feature.zero_crossing_rate(y)
    # feature_vector.append(np.mean(zcr))
    # feature_vector.append(np.std(zcr))
    # feature_names.extend(["zcr_mean", "zcr_std"])
    
    # # Root Mean Square Energy
    # rms = librosa.feature.rms(y=y)
    # feature_vector.append(np.mean(rms))
    # feature_vector.append(np.std(rms))
    # feature_names.extend(["rms_mean", "rms_std"])
    
    # # Spectral Centroid
    # spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    # feature_vector.append(np.mean(spectral_centroid))
    # feature_vector.append(np.std(spectral_centroid))
    # feature_names.extend(["spectral_centroid_mean", "spectral_centroid_std"])
    
    # # Spectral Bandwidth
    # spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    # feature_vector.append(np.mean(spectral_bandwidth))
    # feature_vector.append(np.std(spectral_bandwidth))
    # feature_names.extend(["spectral_bandwidth_mean", "spectral_bandwidth_std"])
    
    # # Spectral Rolloff
    # spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    # feature_vector.append(np.mean(spectral_rolloff))
    # feature_vector.append(np.std(spectral_rolloff))
    # feature_names.extend(["spectral_rolloff_mean", "spectral_rolloff_std"])
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_stds = np.std(mfccs, axis=1)
    feature_vector.extend(mfcc_means)
    feature_vector.extend(mfcc_stds)
    feature_names.extend([f"mfccs_mean_{i}" for i in range(len(mfcc_means))])
    feature_names.extend([f"mfccs_std_{i}" for i in range(len(mfcc_stds))])
    
    # # Chroma
    # chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    # chroma_means = np.mean(chroma, axis=1)
    # chroma_stds = np.std(chroma, axis=1)
    # feature_vector.extend(chroma_means)
    # feature_vector.extend(chroma_stds)
    # feature_names.extend([f"chroma_mean_{i}" for i in range(len(chroma_means))])
    # feature_names.extend([f"chroma_std_{i}" for i in range(len(chroma_stds))])
    
    # # Spectral Contrast
    # contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    # contrast_means = np.mean(contrast, axis=1)
    # contrast_stds = np.std(contrast, axis=1)
    # feature_vector.extend(contrast_means)
    # feature_vector.extend(contrast_stds)
    # feature_names.extend([f"contrast_mean_{i}" for i in range(len(contrast_means))])
    # feature_names.extend([f"contrast_std_{i}" for i in range(len(contrast_stds))])
    
    # # Tonnetz
    # y_harmonic = librosa.effects.harmonic(y)
    # tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    # tonnetz_means = np.mean(tonnetz, axis=1)
    # tonnetz_stds = np.std(tonnetz, axis=1)
    # feature_vector.extend(tonnetz_means)
    # feature_vector.extend(tonnetz_stds)
    # feature_names.extend([f"tonnetz_mean_{i}" for i in range(len(tonnetz_means))])
    # feature_names.extend([f"tonnetz_std_{i}" for i in range(len(tonnetz_stds))])
    
    # # Discrete Wavelet Transform
    # coeffs = pywt.wavedec(y, "db4", level=5)
    # dwt_means = [np.mean(np.abs(c)) for c in coeffs]
    # dwt_stds = [np.std(np.abs(c)) for c in coeffs]
    # feature_vector.extend(dwt_means)
    # feature_vector.extend(dwt_stds)
    # feature_names.extend([f"dwt_mean_{i}" for i in range(len(dwt_means))])
    # feature_names.extend([f"dwt_std_{i}" for i in range(len(dwt_stds))])
    
    return np.array(feature_vector), feature_names


def _extract_dataset_features(dataset_type):
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

    print(f"Processing {len(audio_files)} files from the {dataset_type.upper()} dataset")

    # Extract features for each file
    features_list = []
    labels = []
    feature_names = None

    total_files = len(audio_files)
    for i, file_path in enumerate(audio_files):
        # Simple progress indicator
        if (i + 1) % 10 == 0 or (i + 1) == total_files:
            print(f"Extracting features: {i + 1}/{total_files} files processed", end="\r")
            
        # Get metadata for file
        filename = file_path.name
        file_metadata = metadata[metadata["filename"] == filename]
        if file_metadata.empty:
            continue

        label = file_metadata["target"].values[0]

        # Load audio file
        y, sr = librosa.load(file_path, sr=None)

        # Extract features directly as vector
        feature_vector, names = extract_features_vector(y, sr)
        
        # Store feature names from first file
        if feature_names is None:
            feature_names = names

        features_list.append(feature_vector)
        labels.append(label)
    
    print()  # New line after progress indicator

    # Convert to numpy arrays
    X = np.array(features_list)
    y = np.array(labels)

    # Create dataset info
    dataset_info = {
        "feature_names": feature_names,
    }

    # Save features
    np.save(features_dir / "X.npy", X)
    np.save(features_dir / "y.npy", y)
    with open(features_dir / "metadata.pkl", "wb") as f:
        pickle.dump(dataset_info, f)

    print(f"Feature extraction completed. Feature matrix shape: {X.shape}")

    return X, y, dataset_info


def _download_dataset_and_split_into_10_and_50():
    """Download and organize the ESC-50 dataset."""
    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(exist_ok=True)

    # URL for the ESC-50 dataset
    esc50_url = "https://github.com/karoldvl/ESC-50/archive/master.zip"

    # Where to save the zip file
    zip_path = DATA_DIR / "ESC-50-master.zip"

    # Check if the zip file already exists and is valid
    if zip_path.exists():
        print(f"Valid zip file already exists at {zip_path}. Skipping download.")
    else:
        # Download the dataset
        print(f"Downloading ESC-50 dataset...")
        urllib.request.urlretrieve(esc50_url, zip_path)
        print(f"Download complete.")

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

    # Remove the extracted folder but keep a copy of the README
    shutil.copy2(extracted_dir / "README.md", DATA_DIR / "ESC-50-README.md")
    shutil.rmtree(extracted_dir)


def _check_for_extracted_features(dataset_type):
    """Load pre-extracted features from cache."""
    # Check if dataset exists, if not download it
    if not (ESC10_DIR.exists() and ESC50_DIR.exists()):
        print("Dataset not found. Downloading...")
        _download_dataset_and_split_into_10_and_50()

    # Determine which dataset to use
    if dataset_type == "esc10":
        meta_path = ESC10_DIR / "meta" / "esc10.csv"
        feature_dir = ESC10_DIR / "features"
    else:
        meta_path = ESC50_DIR / "meta" / "esc50.csv"
        feature_dir = ESC50_DIR / "features"

    # Check if features directory exists
    if not feature_dir.exists():
        feature_dir.mkdir(parents=True, exist_ok=True)
        print("No feature cache found. Extracting features...")
        X, y, dataset_info = _extract_dataset_features(dataset_type)
    else:
        # Check if feature files exist
        x_path = feature_dir / "X.npy"
        y_path = feature_dir / "y.npy"
        metadata_path = feature_dir / "metadata.pkl"

        if x_path.exists() and y_path.exists() and metadata_path.exists():
            # Load features and metadata from NumPy files
            print(f"Loading cached features from: {feature_dir}")
            try:
                X = np.load(x_path)
                y = np.load(y_path)
                with open(metadata_path, "rb") as f:
                    dataset_info = pickle.load(f)
                print(f"Successfully loaded features with shape {X.shape}")
            except Exception as e:
                print(f"Error loading cached features: {str(e)}")
                print("Extracting new features...")
                X, y, dataset_info = _extract_dataset_features(dataset_type)
        else:
            print("Feature files incomplete or not found. Extracting new features...")
            X, y, dataset_info = _extract_dataset_features(dataset_type)

    # Load actual label names from CSV
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found at: {meta_path}")

    metadata = pd.read_csv(meta_path)
    label_names = metadata.groupby("target")["category"].first().to_dict()

    # Add label names to dataset info
    dataset_info["label_names"] = label_names

    return X, y, dataset_info


def _prepare_features(X, y, n_components, label_names=None):
    """
    Process features for classification.

    Args:
        X: Feature matrix
        y: Labels
        n_components: Number of PCA components to keep
        label_names: Dictionary mapping label indices to names

    Returns:
        Processed data for algorithm
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=109, stratify=y
    )

    # Standardize features
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Apply PCA to reduce dimensions
    pca = PCA(n_components=n_components).fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    # Scale to [-1, 1]
    samples = np.append(X_train, X_test, axis=0)
    minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
    X_train = minmax_scale.transform(X_train)
    X_test = minmax_scale.transform(X_test)

    # Create training/test input dictionaries using actual label names
    if label_names is None:
        label_names = {i: f"Class_{i}" for i in range(len(np.unique(y)))}

    # Create dictionary with class-specific data
    training_input = {
        label_names[k]: X_train[y_train == k, :] for k in np.unique(y)
    }
    test_input = {label_names[k]: X_test[y_test == k, :] for k in np.unique(y)}

    return training_input, test_input, list(label_names.values())


def esc(dataset="esc10", n=2):
    """
    Load and process ESC dataset.

    Args:
        dataset: Which dataset to use: "esc10" or "esc50"
        n: Number of PCA components to keep (default=2)

    Returns:
        X_train: Processed feature matrix
        training_input: Dictionary with training data by class
        test_input: Dictionary with test data by class
        class_labels: List of class names
    """
    if dataset not in ["esc10", "esc50"]:
        raise ValueError("Dataset must be either 'esc10' or 'esc50'")
        
    try:
        # Load pre-extracted features
        X, y, metadata = _check_for_extracted_features(dataset)

        # Standardize, PCA, and scale to [-1, 1]
        return _prepare_features(X, y, n, metadata["label_names"])
    except Exception as e:
        print(f"Error loading {dataset.upper()} dataset: {str(e)}")
        raise
    finally:
        print(f"{dataset.upper()} dataset processing complete")



if __name__ == "__main__":
    # Test on the smaller ESC-10 dataset
    training_input, test_input, class_labels = esc(dataset="esc10", n=2)

    # Print length of each class in training and test data
    print(f"\nClass labels: {class_labels}")
    print("\nTraining data samples per class:")
    for key in training_input:
        print(f"{key}: {len(training_input[key])}")
        
    print("\nTest data samples per class:")
    for key in test_input:
        print(f"{key}: {len(test_input[key])}")