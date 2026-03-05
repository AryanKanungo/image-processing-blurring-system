"""
============================================================================================
 REGION BLUR TOOL — blur_region.py
============================================================================================

 REQUIREMENTS:
   pip install opencv-python numpy

 USAGE:
   python blur_region.py <path_to_image_or_video> [--kernel <size>]

 EXAMPLES:
   python blur_region.py photo.jpg
   python blur_region.py photo.jpg --kernel 15
   python blur_region.py clip.mp4
   python blur_region.py clip.mp4 --kernel 21

 INTERACTION:
   • Click and drag to draw a rectangle over the region you want blurred.
   • Press ENTER  → confirm selection and process the file.
   • Press ESC    → cancel / redraw selection.

 OUTPUT:
   Image → blurred_output.jpg  (same directory as input)
   Video → blurred_output.mp4  (same directory as input)

 NOTES:
   • The blur is a manual mean (box) blur — no cv2 blur functions are used.
   • Kernel size must be an odd integer ≥ 3.  An even value is bumped up by 1.
   • For video, the selected region is blurred on every frame at the same coordinates.
============================================================================================
"""

import sys
import os
import argparse
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# 1. MANUAL MEAN-BLUR IMPLEMENTATION
# ---------------------------------------------------------------------------

def manual_mean_blur(image: np.ndarray, roi: tuple, kernel_size: int = 5) -> np.ndarray:
    """
    Apply a mean (box) blur to a rectangular region of interest (ROI) inside
    *image* without using any high-level OpenCV blur function.

    The blur is computed via 2-D prefix sums (summed-area table), which gives
    O(1) per-pixel cost regardless of kernel size — fully vectorised with NumPy.

    Parameters
    ----------
    image       : H×W×C uint8 array (BGR or grayscale).
    roi         : (x1, y1, x2, y2) pixel coordinates (inclusive, top-left origin).
    kernel_size : Side length of the square averaging kernel (will be forced odd ≥ 3).

    Returns
    -------
    A copy of *image* with the ROI replaced by the blurred version.
    """
    # ---- sanitise kernel size ------------------------------------------------
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    half = kernel_size // 2

    # ---- unpack & clamp ROI --------------------------------------------------
    x1, y1, x2, y2 = roi
    h, w = image.shape[:2]
    x1, x2 = sorted([max(0, x1), min(w - 1, x2)])
    y1, y2 = sorted([max(0, y1), min(h - 1, y2)])

    if x1 >= x2 or y1 >= y2:
        return image.copy()   # degenerate ROI — nothing to do

    result = image.copy()

    # Work on float64 for accurate accumulation, then cast back.
    # If image is grayscale (2-D), add a channel dim so the logic is uniform.
    squeezed = image.ndim == 2
    src = image.astype(np.float64)
    if squeezed:
        src = src[:, :, np.newaxis]

    C = src.shape[2]

    # ---- build padded summed-area table (SAT) for the whole image -----------
    # Pad by *half* on all sides so every ROI pixel has a full kernel window.
    pad = half
    padded = np.pad(src, ((pad, pad), (pad, pad), (0, 0)), mode='edge')

    # SAT shape: (H+1+2*pad) × (W+1+2*pad) × C  (extra +1 for the prefix trick)
    sat = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1, C), dtype=np.float64)
    sat[1:, 1:, :] = np.cumsum(np.cumsum(padded, axis=0), axis=1)

    # ---- vectorised box-filter query over the ROI ---------------------------
    # For each pixel (r, c) in the original image the kernel covers
    #   rows  [r-half .. r+half]  →  in padded coords [r .. r+kernel_size-1]
    #   (after the +1 offset of the SAT)
    #
    # SAT query for box sum from (r0,c0) to (r1,c1) inclusive:
    #   S = sat[r1+1, c1+1] - sat[r0, c1+1] - sat[r1+1, c0] + sat[r0, c0]

    roi_rows = np.arange(y1, y2 + 1)   # shape (Ry,)
    roi_cols = np.arange(x1, x2 + 1)   # shape (Rx,)

    # In padded+SAT index space the kernel top-left for row r is just r
    # (because padding shifts by *half*, and SAT has an extra leading 0-row/col).
    # SAT row for top of kernel    = r  (0-indexed in SAT)
    # SAT row for bottom of kernel = r + kernel_size
    r0 = roi_rows[:, np.newaxis]           # (Ry, 1)
    r1 = r0 + kernel_size                  # (Ry, 1)
    c0 = roi_cols[np.newaxis, :]           # (1, Rx)
    c1 = c0 + kernel_size                  # (1, Rx)

    box_sum = (
        sat[r1, c1]          # shape (Ry, Rx, C)
        - sat[r0, c1]
        - sat[r1, c0]
        + sat[r0, c0]
    )

    box_mean = box_sum / (kernel_size * kernel_size)

    blurred_roi = np.clip(box_mean, 0, 255).astype(np.uint8)

    if squeezed:
        blurred_roi = blurred_roi[:, :, 0]

    result[y1:y2 + 1, x1:x2 + 1] = blurred_roi
    return result


# ---------------------------------------------------------------------------
# 2. REGION SELECTION (interactive)
# ---------------------------------------------------------------------------

def select_region(frame: np.ndarray, window_title: str = "Select Region") -> tuple | None:
    """
    Display *frame* in an OpenCV window and let the user draw a rectangle.

    Controls
    --------
    Left-click + drag : draw rectangle
    ENTER             : confirm
    ESC               : cancel (returns None)

    Returns
    -------
    (x1, y1, x2, y2) on confirmation, or None on cancellation.
    """
    state = {
        "drawing": False,
        "start":   None,
        "end":     None,
        "done":    False,
        "cancel":  False,
    }
    display = frame.copy()
    original = frame.copy()

    def mouse_cb(event, x, y, flags, param):
        nonlocal display
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["start"]   = (x, y)
            state["end"]     = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["end"] = (x, y)
            display = original.copy()
            cv2.rectangle(display, state["start"], state["end"], (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            state["drawing"] = False
            state["end"] = (x, y)
            display = original.copy()
            cv2.rectangle(display, state["start"], state["end"], (0, 255, 0), 2)

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_title, mouse_cb)

    instructions = "Drag to select region  |  ENTER = confirm  |  ESC = cancel"
    print(f"\n  {instructions}\n")

    while True:
        overlay = display.copy()
        cv2.putText(overlay, instructions, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window_title, overlay)

        key = cv2.waitKey(20) & 0xFF

        if key == 13:  # ENTER
            if state["start"] and state["end"] and state["start"] != state["end"]:
                state["done"] = True
                break
            else:
                print("  ⚠  No region selected yet — draw a rectangle first.")

        elif key == 27:  # ESC
            state["cancel"] = True
            break

    cv2.destroyAllWindows()

    if state["cancel"] or not state["done"]:
        return None

    x1, y1 = state["start"]
    x2, y2 = state["end"]
    # Normalise so x1 < x2, y1 < y2
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


# ---------------------------------------------------------------------------
# 3. IMAGE PROCESSING
# ---------------------------------------------------------------------------

def process_image(input_path: str, output_path: str, kernel_size: int) -> None:
    """Load an image, let the user select a ROI, blur it, and save the result."""
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Cannot read image: '{input_path}'")

    print(f"  Image loaded: {image.shape[1]}×{image.shape[0]} px")

    roi = select_region(image, window_title="Select Region — Image")
    if roi is None:
        print("  Selection cancelled. No output written.")
        return

    print(f"  ROI selected: x1={roi[0]}, y1={roi[1]}, x2={roi[2]}, y2={roi[3]}")
    print(f"  Applying manual mean blur (kernel {kernel_size}×{kernel_size}) …")

    blurred = manual_mean_blur(image, roi, kernel_size)

    cv2.imwrite(output_path, blurred)
    print(f"  ✓ Saved → {output_path}")

    # Preview
    cv2.imshow("Result (press any key to close)", blurred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# 4. VIDEO PROCESSING
# ---------------------------------------------------------------------------

def process_video(input_path: str, output_path: str, kernel_size: int) -> None:
    """
    Open a video, display the first frame for ROI selection, then re-encode
    every frame with the selected region blurred.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: '{input_path}'")

    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Video: {width}×{height} @ {fps:.2f} fps  ({total} frames)")

    # --- read first frame for selection ---
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Could not read the first frame of the video.")

    roi = select_region(first_frame, window_title="Select Region — Video (first frame)")
    if roi is None:
        cap.release()
        print("  Selection cancelled. No output written.")
        return

    print(f"  ROI selected: x1={roi[0]}, y1={roi[1]}, x2={roi[2]}, y2={roi[3]}")
    print(f"  Applying manual mean blur (kernel {kernel_size}×{kernel_size}) to all frames …")

    # --- set up writer ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process frame 0 (already read)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    processed = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        blurred_frame = manual_mean_blur(frame, roi, kernel_size)
        writer.write(blurred_frame)
        processed += 1

        # Progress indicator every 30 frames
        if processed % 30 == 0 or processed == total:
            pct = (processed / total * 100) if total > 0 else 0
            print(f"    … {processed}/{total} frames ({pct:.1f}%)", end="\r")

    cap.release()
    writer.release()
    print(f"\n  ✓ Saved → {output_path}  ({processed} frames)")


# ---------------------------------------------------------------------------
# 5. FILE-TYPE DETECTION & ENTRY POINT
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".m4v"}


def detect_media_type(path: str) -> str:
    """Return 'image', 'video', or raise ValueError for unsupported types."""
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    raise ValueError(
        f"Unsupported file extension '{ext}'.\n"
        f"  Supported images: {', '.join(sorted(IMAGE_EXTENSIONS))}\n"
        f"  Supported videos: {', '.join(sorted(VIDEO_EXTENSIONS))}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively blur a selected rectangular region in an image or video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Path to input image or video file.")
    parser.add_argument(
        "--kernel", type=int, default=5,
        help="Blur kernel size (odd integer, default: 5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input
    kernel_size = args.kernel

    # ---- basic validation ---------------------------------------------------
    if not os.path.isfile(input_path):
        print(f"ERROR: File not found: '{input_path}'", file=sys.stderr)
        sys.exit(1)

    if kernel_size < 3:
        print("WARNING: kernel_size < 3; clamping to 3.")
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1
        print(f"WARNING: kernel_size must be odd; bumped to {kernel_size}.")

    try:
        media_type = detect_media_type(input_path)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # ---- build output path --------------------------------------------------
    base_dir = os.path.dirname(os.path.abspath(input_path))
    if media_type == "image":
        output_path = os.path.join(base_dir, "blurred_output.jpg")
    else:
        output_path = os.path.join(base_dir, "blurred_output.mp4")

    print(f"\n  Input  : {input_path}")
    print(f"  Type   : {media_type}")
    print(f"  Output : {output_path}")
    print(f"  Kernel : {kernel_size}×{kernel_size}")

    # ---- dispatch -----------------------------------------------------------
    try:
        if media_type == "image":
            process_image(input_path, output_path, kernel_size)
        else:
            process_video(input_path, output_path, kernel_size)
    except (ValueError, RuntimeError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()