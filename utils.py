"""Utility functions for image handling and preset management."""
import requests
from PIL import Image
from io import BytesIO


def download_image_from_url(url: str):
    """Download image from URL and return (PIL Image, None) or (None, error_str)."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img.load()  # Verify image is valid
        return img, None
    except requests.exceptions.Timeout:
        return None, "Timeout connecting to URL"
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP error: {e.code} {e.reason}"
    except requests.exceptions.RequestException as e:
        return None, f"Network error: {str(e)}"
    except IOError as e:
        return None, f"Invalid image format: {str(e)}"
    except Exception as e:
        return None, f"Error: {str(e)}"


def get_image(uploaded_file, url_input):
    """Return (PIL Image | None, error_str | None)."""
    if url_input and url_input.strip():
        img, err = download_image_from_url(url_input.strip())
        return img, err
    elif uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image.load()  # Verify image is valid/corrupt
            return image, None
        except IOError as e:
            return None, f"Invalid image file: {str(e)}"
        except Exception as e:
            return None, f"Error opening image: {str(e)}"
    else:
        return None, "Please upload an image or enter a URL."


def validate_preset_value(name: str, value: float, min_val: float, max_val: float) -> tuple:
    """Validate a preset value is within bounds. Returns (clamped_value, warning_str or None)."""
    if value < min_val:
        return min_val, f"Warning: {name} clamped from {value:.1f} to minimum {min_val:.1f}"
    elif value > max_val:
        return max_val, f"Warning: {name} clamped from {value:.1f} to maximum {max_val:.1f}"
    return value, None


# Preset definitions with validation
PRESETS = {
    "draft": {
        "max_size": 100, "blur": 5, "gamma": 0.8, "smoothing": 3, "hill": 3
    },
    "preview": {
        "max_size": 200, "blur": 2, "gamma": 1.2, "smoothing": 2, "hill": 2
    },
    "high": {
        "max_size": 500, "blur": 1, "gamma": 1.5, "smoothing": 1, "hill": 1
    },
    "extreme": {
        "max_size": 2500, "blur": 0, "gamma": 2.5, "smoothing": 0, "hill": 0.05
    }
}


def get_preset_with_validation(preset_name: str):
    """Get preset values with bounds validation. Returns (dict, warnings_list)."""
    if preset_name not in PRESETS:
        return None, [f"Unknown preset: {preset_name}"]
    
    preset = dict(PRESETS[preset_name])  # Copy to avoid modifying original
    warnings = []
    
    # Validate max_size (50-2000)
    preset["max_size"], w = validate_preset_value("max_size", preset["max_size"], 50, 2000)
    if w: warnings.append(w)
    
    # Validate gamma (0.1-4.0)
    preset["gamma"], w = validate_preset_value("gamma", preset["gamma"], 0.1, 4.0)
    if w: warnings.append(w)
    
    # Validate blur (0-20)
    preset["blur"], w = validate_preset_value("blur", preset["blur"], 0, 20)
    if w: warnings.append(w)
    
    # Validate hill (0.1-20)
    preset["hill"], w = validate_preset_value("hill", preset["hill"], 0.1, 20)
    if w: warnings.append(w)
    
    return preset, warnings
