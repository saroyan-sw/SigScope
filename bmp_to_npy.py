from pathlib import Path

import numpy as np
from PIL import Image


def convert_single_bmp_to_npy(input_file: Path, output_file: Path, mode: str = "rgb"):
    """
    Convert one BMP image to NPY.

    mode:
        "rgb"       -> shape: (H, W, 3)
        "grayscale" -> shape: (H, W)
        "unchanged" -> keep original BMP mode
    """

    if input_file.suffix.lower() != ".bmp":
        raise ValueError(f"Input file must be a BMP file: {input_file}")

    image = Image.open(input_file)

    if mode == "rgb":
        image = image.convert("RGB")
    elif mode == "grayscale":
        image = image.convert("L")
    elif mode == "unchanged":
        pass
    else:
        raise ValueError(f"Unknown mode: {mode}")

    array = np.array(image)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.suffix.lower() != ".npy":
        output_file = output_file.with_suffix(".npy")

    np.save(output_file, array)

    print(f"Saved: {output_file}")
    print(f"Shape: {array.shape}")
    print(f"Dtype: {array.dtype}")


def convert_folder_bmp_to_npy(
    input_folder: Path,
    output_folder: Path,
    mode: str = "rgb",
    recursive: bool = False,
):
    """
    Convert all BMP images inside a folder to NPY.
    """

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)

    pattern = "**/*.bmp" if recursive else "*.bmp"
    bmp_files = sorted(input_folder.glob(pattern))

    if len(bmp_files) == 0:
        print(f"No BMP files found in: {input_folder}")
        return

    for bmp_file in bmp_files:
        relative_path = bmp_file.relative_to(input_folder)
        output_file = output_folder / relative_path.with_suffix(".npy")

        convert_single_bmp_to_npy(
            input_file=bmp_file,
            output_file=output_file,
            mode=mode,
        )


def main():
    # =========================
    # Write your paths here
    # =========================

    input_path = Path(r"bmp_folder")
    output_path = Path(r"npy_folder")

    mode = "rgb"
    recursive = False

    # =========================
    # Conversion logic
    # =========================

    if input_path.is_file():
        convert_single_bmp_to_npy(
            input_file=input_path,
            output_file=output_path,
            mode=mode,
        )

    elif input_path.is_dir():
        convert_folder_bmp_to_npy(
            input_folder=input_path,
            output_folder=output_path,
            mode=mode,
            recursive=recursive,
        )

    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    main()