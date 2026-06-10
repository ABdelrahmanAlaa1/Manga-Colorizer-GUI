import base64
import io
import math
import random
import re
import string

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import PIL.ImageChops, PIL.ImageOps, PIL.Image


def resize_pad(img, size = 256):
            
    if len(img.shape) == 2:
        img = np.expand_dims(img, 2)
        
    if img.shape[2] == 1:
        img = np.repeat(img, 3, 2)
        
    if img.shape[2] == 4:
        img = img[:, :, :3]

    pad = None        
            
    if (img.shape[0] < img.shape[1]):
        height = img.shape[0]
        ratio = height / (size * 1.5)
        width = int(np.ceil(img.shape[1] / ratio))
        img = cv2.resize(img, (width, int(size * 1.5)), interpolation = cv2.INTER_AREA)

        
        new_width = width + (32 - width % 32)
            
        pad = (0, new_width - width)
        
        img = np.pad(img, ((0, 0), (0, pad[1]), (0, 0)), 'maximum')
    else:
        width = img.shape[1]
        ratio = width / size
        height = int(np.ceil(img.shape[0] / ratio))
        img = cv2.resize(img, (size, height), interpolation = cv2.INTER_AREA)

        new_height = height + (32 - height % 32)
            
        pad = (new_height - height, 0)
        
        img = np.pad(img, ((0, pad[0]), (0, 0), (0, 0)), 'maximum')
        
    if (img.dtype == 'float32'):
        np.clip(img, 0, 1, out = img)

    return img[:, :, :1], pad

def image_to_base64(img, format="WEBP"):
    buffered = io.BytesIO()
    img = PIL.Image.fromarray(img)
    img.save(buffered, format=format)
    buffered.seek(0)
    img_byte = buffered.getvalue()
    return f"data:image/{format.lower()};base64," + base64.b64encode(img_byte).decode('utf-8')

def load_image_as_base64(filepath, format="WEBP"):
    with open(filepath, "rb") as img_file:
        img_byte = img_file.read()
    return f"data:image/{format.lower()};base64," + base64.b64encode(img_byte).decode('utf-8')

def save_image(image, filename, format="WEBP"):
    image_pil = PIL.Image.fromarray(image)
    image_pil.save(filename, format=format)

def sanitize_string(input_string):
    sanitized_string = re.sub(r'[^\w]', '_', input_string)
    return sanitized_string

def distance_from_grayscale(image):
    try:
        img_diff = PIL.ImageChops.difference(image, PIL.ImageOps.grayscale(image).convert('RGB'))
        dist = np.array(img_diff.getdata()).mean()
        return dist
    except:
        return 0

def generate_random_id(length=8):
    characters = string.ascii_uppercase + string.digits
    random_id = ''.join(random.choices(characters, k=length))
    return random_id

def clear_torch_cache():
    torch.cuda.empty_cache()

def _estimate_tile_batch_size(tile_h, tile_w, channels, scale, device, arch_name=None,
                              dtype=None, reserved_mb=0.0):
    """Dynamically compute max tile batch size from free VRAM.

    Parameters
    ----------
    dtype : torch.dtype, optional
        Actual model/input dtype. BF16/FP16 halves per-tile memory vs FP32.
    reserved_mb : float
        Memory (MB) the caller has already committed and that should be
        subtracted from the "free" budget — typically the full-resolution
        output canvas, since it lives for the entire tile loop.
    """
    try:
        if not torch.cuda.is_available():
            return 1
        # Robust device-index extraction: torch.device → .index; str → parse;
        # everything else → default GPU 0.
        dev_idx = 0
        if isinstance(device, torch.device):
            if device.index is not None:
                dev_idx = device.index
        elif isinstance(device, str) and ':' in device:
            try:
                dev_idx = int(device.split(':', 1)[1])
            except ValueError:
                dev_idx = 0
        free_bytes, _ = torch.cuda.mem_get_info(dev_idx)
        free_mb = free_bytes / (1024 * 1024)
        # Caller-reserved memory (e.g. the output canvas) is already part of
        # the live allocation but mem_get_info may not yet reflect it if it
        # was just allocated. Subtracting keeps us honest either way.
        budget_mb = max(0.0, free_mb - reserved_mb)

        # Detect transformer architectures (DAT, FDAT, HAT) vs CNNs (SPAN, ESRGAN)
        is_transformer = False
        if arch_name:
            arch_lower = str(arch_name).lower()
            is_transformer = any(x in arch_lower for x in ('dat', 'hat'))

        # bytes_per_elem from the *actual* runtime dtype. BF16/FP16 = 2 bytes,
        # FP32 = 4. The activation maps follow the compute dtype, so this also
        # scales the multiplier-weighted term correctly.
        if dtype in (torch.float16, torch.bfloat16):
            bytes_per_elem = 2
        else:
            bytes_per_elem = 4

        input_mb = (tile_h * tile_w * channels * bytes_per_elem) / (1024 * 1024)
        output_mb = (tile_h * scale * tile_w * scale * 3 * bytes_per_elem) / (1024 * 1024)

        # Transformers generate massive intermediate activation maps
        activations_multiplier = 40.0 if is_transformer else 4.0
        per_tile_mb = input_mb + output_mb * activations_multiplier

        if per_tile_mb <= 0:
            return 1

        # Use 50% of remaining VRAM (conservative for 8GB cards)
        max_batch = max(1, int((budget_mb * 0.5) / per_tile_mb))
        return min(max_batch, 16)
    except Exception:
        return 1


def tile_process(model, img, scale, tile_size, tile_pad, arch_name=None):
    if scale == 2: print('[-] ScaleFactor=2 is broken, please do not use it yet')
    batch_dim, channel, height, width = img.shape
    output_height = height * scale
    output_width = width * scale

    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

    # Pad input so tile grid is exact AND tile_pad is available on all borders
    grid_h = tiles_y * tile_size
    grid_w = tiles_x * tile_size
    pad_right = grid_w - width
    pad_bottom = grid_h - height

    # Reflect-pad: grid alignment + extraction margin (fallback to replicate if too large)
    pad_spec = (tile_pad, tile_pad + pad_right, tile_pad, tile_pad + pad_bottom)
    try:
        img_padded = F.pad(img, pad_spec, mode='reflect')
    except RuntimeError:
        img_padded = F.pad(img, pad_spec, mode='replicate')

    # Uniform tile dimensions — all tiles are identical size now
    tile_h = tile_size + 2 * tile_pad
    tile_w = tile_size + 2 * tile_pad

    # Build coordinate list instead of holding slice views — allows img_padded
    # to be freed as soon as the last tile in each batch is extracted, rather
    # than being kept alive by view references for the entire loop.
    tile_coords = []
    for y in range(tiles_y):
        for x in range(tiles_x):
            tile_coords.append((y * tile_size, x * tile_size))

    total_tiles = len(tile_coords)

    # channels_last canvas for consistent memory layout with Tensor Cores.
    # Allocate BEFORE estimating batch size so the canvas is included in the
    # live-memory budget — otherwise estimate_tile_batch_size sees free VRAM
    # that the canvas is about to consume and over-batches.
    output = img.new_zeros((batch_dim, 3, output_height, output_width))
    try:
        output = output.contiguous(memory_format=torch.channels_last)
    except Exception:
        pass

    # Account for the canvas explicitly (defensive in case mem_get_info lags).
    bytes_per_elem = 2 if img.dtype in (torch.float16, torch.bfloat16) else 4
    canvas_mb = (batch_dim * 3 * output_height * output_width * bytes_per_elem) / (1024 * 1024)

    # Dynamic batch size based on free VRAM, parameterized by actual dtype.
    max_batch = _estimate_tile_batch_size(
        tile_h, tile_w, channel, scale, img.device, arch_name,
        dtype=img.dtype, reserved_mb=canvas_mb,
    )

    # Scaled tile dimensions for output placement
    out_tile_pad = tile_pad * scale
    out_tile_size = tile_size * scale

    # Process tiles in VRAM-safe batches
    for batch_start in range(0, total_tiles, max_batch):
        batch_end = min(batch_start + max_batch, total_tiles)

        # Slice tiles on-the-fly from img_padded (no persistent views)
        batch_tiles = [
            img_padded[:, :, ys:ys + tile_h, xs:xs + tile_w]
            for ys, xs in tile_coords[batch_start:batch_end]
        ]

        # Stack into batch: (B, C, tile_h, tile_w)
        tile_batch = torch.cat(batch_tiles, dim=0)
        del batch_tiles  # free view references immediately

        try:
            raw_output = model(tile_batch)
            output_batch = raw_output[0] if isinstance(raw_output, tuple) else raw_output
        except RuntimeError as error:
            # OOM or shape error — free the failed batch allocation BEFORE
            # retrying. Without empty_cache(), the fragmented reservation often
            # makes the per-tile retry OOM as well even though a single tile
            # would otherwise fit easily.
            is_oom = 'out of memory' in str(error).lower()
            del tile_batch
            if is_oom and img.is_cuda:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            output_parts = []
            tile_failures = []
            # Re-slice tiles on-the-fly (batch_tiles was freed above)
            fallback_tiles = [
                img_padded[:, :, ys:ys + tile_h, xs:xs + tile_w]
                for ys, xs in tile_coords[batch_start:batch_end]
            ]
            for tile_idx, single_tile in enumerate(fallback_tiles):
                try:
                    raw_out = model(single_tile)
                    out = raw_out[0] if isinstance(raw_out, tuple) else raw_out
                    output_parts.append(out)
                except RuntimeError as single_err:
                    # Final retry: clear cache and try once more in isolation.
                    if 'out of memory' in str(single_err).lower() and single_tile.is_cuda:
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        try:
                            raw_out = model(single_tile)
                            out = raw_out[0] if isinstance(raw_out, tuple) else raw_out
                            output_parts.append(out)
                            continue
                        except RuntimeError as retry_err:
                            single_err = retry_err
                    # Truly unrecoverable for this tile. Record it so the
                    # caller can decide to fail loudly, then place a black
                    # patch so we can at least continue the batch.
                    import traceback
                    print(f'[!] Tile {batch_start + tile_idx} failed: {single_err}')
                    traceback.print_exc()
                    tile_failures.append((batch_start + tile_idx, repr(single_err)))
                    output_parts.append(single_tile.new_zeros(
                        1, 3, tile_h * scale, tile_w * scale
                    ))
            # Surface tile failures upstream. _run_with_precision can then
            # trigger an FP32 retry of the whole image instead of returning
            # a silently-corrupted result with black tile holes.
            if tile_failures:
                raise RuntimeError(
                    f"tile_process: {len(tile_failures)} of {total_tiles} tiles "
                    f"failed after per-tile retry; first failure: {tile_failures[0][1]}"
                )
            output_batch = torch.cat(output_parts, dim=0)

        # Write batch results to output canvas (no CPU round-trip)
        for i in range(output_batch.shape[0]):
            flat_idx = batch_start + i
            y = flat_idx // tiles_x
            x = flat_idx % tiles_x

            out_y = y * out_tile_size
            out_x = x * out_tile_size

            # Crop padding from the output tile
            cropped = output_batch[i:i+1, :,
                                   out_tile_pad:out_tile_pad + out_tile_size,
                                   out_tile_pad:out_tile_pad + out_tile_size]

            # Handle edge tiles that extend beyond original output dimensions
            actual_h = min(out_tile_size, output_height - out_y)
            actual_w = min(out_tile_size, output_width - out_x)

            output[:, :, out_y:out_y + actual_h, out_x:out_x + actual_w] = \
                cropped[:, :, :actual_h, :actual_w]

        del output_batch  # free batch intermediates before next iteration

    del img_padded  # free padded input before returning canvas
    return output