import os
import re
import PIL.Image
import numpy as np

# Re-expose transfer_luminance_from_source from the Backend directory
# Note: when running from app.py at root, Backend is in sys.path or imported directly.
# Since app.py added the Backend folder or the folder root is in path, we import transfer_luminance_from_source.
try:
    from transfer_quality import transfer_luminance_from_source
except ImportError:
    from Backend.transfer_quality import transfer_luminance_from_source


def _natural_sort_key(filename):
    """Natural sort key for fuzzy file matching.

    Splits filename into alternating (text, number) parts so that:
    - Text parts sort alphabetically (case-insensitive)
    - Embedded numbers sort by numeric value (not lexicographic)
    - Different prefixes from paused/resumed batch runs stay grouped correctly
    """
    stem = os.path.splitext(filename)[0]
    parts = re.split(r'(\d+)', stem)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(('', int(part)))
        elif part:
            key.append((part.lower(), 0))
    return key


def build_external_color_map(grouped_by_folder, external_dir, supported_formats=None):
    """Build a mapping from (rel_dir, image_name) -> external_color_filepath.

    Uses natural sort (alphanumeric) to order both file lists, then matches
    them positionally (1st to 1st, 2nd to 2nd, etc.).
    """
    if not external_dir or not os.path.isdir(external_dir):
        return {}, [f"External color source directory does not exist: {external_dir}"]

    if supported_formats is None:
        supported_formats = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')

    color_map = {}
    warnings = []

    for rel_dir, folder_images in grouped_by_folder.items():
        # Try matching subdirectory first, then flat
        ext_subdir = os.path.join(external_dir, rel_dir) if rel_dir else external_dir
        if not os.path.isdir(ext_subdir):
            ext_subdir = external_dir  # Fall back to flat directory

        # Get external files in this directory
        external_files = []
        try:
            for f in os.listdir(ext_subdir):
                if f.lower().endswith(supported_formats):
                    external_files.append(os.path.join(ext_subdir, f))
        except OSError:
            warnings.append(f"Cannot read external color directory: {ext_subdir}")
            continue

        if not external_files:
            display_dir = rel_dir or '(root)'
            warnings.append(f"No external color images found for folder '{display_dir}' in {ext_subdir}")
            continue

        # Sort both lists by natural sort key (handles prefix groups + numbers)
        pipeline_names = [os.path.basename(fp) for fp, _ in folder_images]
        pipeline_sorted = sorted(
            enumerate(pipeline_names),
            key=lambda x: _natural_sort_key(x[1]),
        )

        external_sorted = sorted(
            external_files,
            key=lambda x: _natural_sort_key(os.path.basename(x)),
        )

        # Match by position
        match_count = min(len(pipeline_sorted), len(external_sorted))
        for i in range(match_count):
            _orig_idx, pipeline_name = pipeline_sorted[i]
            external_path = external_sorted[i]
            color_map[(rel_dir, pipeline_name)] = external_path

        # Warn about unmatched files
        display_dir = rel_dir or '(root)'
        if len(pipeline_sorted) > len(external_sorted):
            unmatched = len(pipeline_sorted) - len(external_sorted)
            warnings.append(
                f"Folder '{display_dir}': {unmatched} pipeline images have no matching external color source"
            )
        elif len(external_sorted) > len(pipeline_sorted):
            extra = len(external_sorted) - len(pipeline_sorted)
            warnings.append(
                f"Folder '{display_dir}': {extra} extra external color images (unused)"
            )

    return color_map, warnings
