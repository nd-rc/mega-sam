import subprocess
import json
import argparse
import sys
import math

def get_metadata(file_path):
    """Run ffprobe to extract metadata."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        file_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running ffprobe: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def estimate_fov(metadata):
    """
    Estimate Horizontal FOV based on metadata.
    Defaults to 73.0 (approx iPhone 14 Pro 24mm equivalent) if unknown.
    """
    # 1. Try to find Make/Model in format tags
    format_tags = metadata.get("format", {}).get("tags", {})
    make = format_tags.get("com.apple.quicktime.make", "").lower()
    model = format_tags.get("com.apple.quicktime.model", "").lower()
    
    # Fallback to standard make/model if apple tags aren't present
    if not make:
        make = format_tags.get("make", "").lower()
    if not model:
        model = format_tags.get("model", "").lower()

    print(f"Detected Camera: {make} {model}", file=sys.stderr)

    # Database of common 35mm equivalent focal lengths (Main camera usually)
    # FOV = 2 * arctan(36 / (2 * focal_length_35mm)) * (180 / pi)
    
    # iPhone 14 Pro / 15 Pro Main: 24mm -> ~73.7 deg
    # iPhone 13 Pro: 26mm -> ~69.4 deg
    # iPhone X/11/12: ~26mm -> ~69.4 deg
    # GoPro Wide: often much wider (~118) but varies by mode.
    
    focal_length_35mm = None
    
    if "iphone" in model:
        if "14 pro" in model or "15 pro" in model or "16 pro" in model:
            focal_length_35mm = 24.0
        elif "13 pro" in model:
            focal_length_35mm = 26.0
        else:
            # Generic iPhone fallback (often 26mm or 28mm for older ones)
            focal_length_35mm = 26.0
    
    # If we found a focal length, calculate FOV
    if focal_length_35mm:
        fov = 2 * math.atan(36.0 / (2 * focal_length_35mm)) * (180.0 / math.pi)
        return round(fov, 2)
    
    # 2. Look for explicit focal length in streams (rare for simple video files)
    # Sometimes represented as a float string.
    
    return 73.0 # Default fallback

def main():
    parser = argparse.ArgumentParser(description="Extract FOV from video metadata.")
    parser.add_argument("video_path", help="Path to the input video file.")
    args = parser.parse_args()

    metadata = get_metadata(args.video_path)
    fov = estimate_fov(metadata)
    
    # Print ONLY the number to stdout so it can be captured by bash
    print(fov)

if __name__ == "__main__":
    main()

