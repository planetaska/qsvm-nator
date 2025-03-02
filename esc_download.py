import os
import sys
import urllib.request
import zipfile
import pandas as pd
import shutil
from pathlib import Path


def main():
    # Inform the user about what the script will do
    print("This script will:")
    print("1. Create a 'data' directory in the current location (if it doesn't exist)")
    print("2. Download the ESC-50 dataset (~600 MB)")
    print("3. Extract and organize the dataset into ESC-10 and ESC-50 subsets")

    # Ask for confirmation
    while True:
        response = input("\nDo you want to proceed? (y/n): ").strip().lower()
        if response in ["y", "yes"]:
            break
        elif response in ["n", "no"]:
            print("Operation cancelled by user.")
            sys.exit(0)
        else:
            print("Please enter 'y' or 'n'.")

    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # URL for the ESC-50 dataset
    esc50_url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    zip_path = data_dir / "ESC-50-master.zip"

    # Check if the zip file already exists and is valid
    if zip_path.exists() and is_valid_zip(zip_path):
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
        zip_ref.extractall(data_dir)
    print("Extraction complete.")

    # Path to the extracted dataset
    extracted_dir = data_dir / "ESC-50-master"

    # Read the metadata
    metadata_path = extracted_dir / "meta" / "esc50.csv"
    metadata = pd.read_csv(metadata_path)

    # Create directories for ESC-10 and ESC-50 with audio and meta subdirectories
    esc10_dir = data_dir / "esc10"
    esc50_dir = data_dir / "esc50"

    esc10_audio_dir = esc10_dir / "audio"
    esc10_meta_dir = esc10_dir / "meta"
    esc50_audio_dir = esc50_dir / "audio"
    esc50_meta_dir = esc50_dir / "meta"

    # Create all directories
    for directory in [
        esc10_dir,
        esc50_dir,
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
    shutil.copy2(extracted_dir / "README.md", esc50_dir / "README.md")
    shutil.copy2(extracted_dir / "README.md", esc10_dir / "README.md")

    print(f"ESC-50 dataset organized successfully:")
    print(f"- Full dataset: {esc50_dir} ({len(metadata)} files)")
    print(f"- ESC-10 subset: {esc10_dir} ({len(esc10_metadata)} files)")
    print(
        f"- Metadata saved to {esc50_meta_dir/'esc50.csv'} and {esc10_meta_dir/'esc10.csv'}"
    )

    # Remove the extracted folder but keep a copy of the README
    shutil.copy2(extracted_dir / "README.md", data_dir / "ESC-50-README.md")
    print(f"Removing extracted directory {extracted_dir}...")
    shutil.rmtree(extracted_dir)
    print("Cleanup complete.")


# Function to check if a file is a valid zip file
def is_valid_zip(file_path):
    try:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            return True
    except zipfile.BadZipFile:
        return False


if __name__ == "__main__":
    main()
