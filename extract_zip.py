import zipfile
from pathlib import Path
import sys

def extract_all_zips(folder_path: Path, output_folder: Path = None):
    """
    Extracts all .zip files in the given folder.

    Args:
        folder_path (Path): Path to the directory containing .zip files.
        output_folder (Path, optional): Base directory where extracted folders will be placed.
            If None, extracted files will go into subdirectories within folder_path.
    """
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Provided path is not a directory: {folder_path}")

    # Use folder_path as base if no output_folder provided
    base_output = output_folder or folder_path
    base_output.mkdir(parents=True, exist_ok=True)

    # Iterate over all .zip files in the folder
    for zip_path in folder_path.glob('*.zip'):
        try:
            # Create a subdirectory named after the zip (without extension)
            extract_dir = base_output / zip_path.stem
            extract_dir.mkdir(exist_ok=True)

            # Open and extract
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(path=extract_dir)
            print(f"Extracted '{zip_path.name}' to '{extract_dir}'")

        except zipfile.BadZipFile:
            print(f"Skipping '{zip_path.name}': not a valid zip file.")
        except Exception as e:
            print(f"Error extracting '{zip_path.name}': {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract all .zip files in a folder into subdirectories named after each zip file."
    )
    parser.add_argument(
        'folder',
        type=Path,
        help='Path to the folder containing zip files'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=None,
        help='Optional output base directory for extracted files'
    )
    args = parser.parse_args()

    extract_all_zips(args.folder, args.output)


if __name__ == '__main__':
    main()
