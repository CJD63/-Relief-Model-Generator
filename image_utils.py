"""Image utility functions: downloading, loading, adjusting, and AI model metadata."""

import requests
from PIL import Image, ImageEnhance
from io import BytesIO


def download_image_from_url(url: str, keep_colors=False):
    """Download an image from a URL.

    Args:
        url: The image URL.
        keep_colors: If False, convert to grayscale ('L') mode.

    Returns:
        (PIL.Image, None) on success, or (None, error_string) on failure.
    """
    try:
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/91.0.4472.124 Safari/537.36'
            )
        }
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        image = Image.open(BytesIO(r.content))
        if not keep_colors and image.mode != 'L':
            image = image.convert('L')
        return image, None
    except requests.exceptions.Timeout:
        return None, 'Timeout error: The request took too long. Please check the URL.'
    except requests.exceptions.HTTPError as e:
        return None, 'HTTP error ' + str(e.response.status_code) + ': Could not download image.'
    except requests.exceptions.RequestException as e:
        return None, 'Network error: ' + str(e)
    except IOError as e:
        return None, 'Invalid image format: ' + str(e)
    except Exception as e:
        return None, 'Unexpected error: ' + str(e)


def get_image(uploaded_file, url_input: str, keep_colors=False):
    """Get a PIL Image from an uploaded file or URL.

    Args:
        uploaded_file: Streamlit UploadedFile or None.
        url_input: URL string or empty.
        keep_colors: If False, convert to grayscale ('L') mode.

    Returns:
        (PIL.Image, None) on success, or (None, error_string) on failure.
    """
    if url_input and url_input.strip():
        img, err = download_image_from_url(url_input.strip(), keep_colors)
        return img, err
    elif uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image.load()
            if not keep_colors and image.mode != 'L':
                image = image.convert('L')
            return image, None
        except IOError as e:
            return None, 'Invalid or corrupted image file: ' + str(e)
        except Exception as e:
            return None, 'Error loading image: ' + str(e)
    return None, 'Please upload an image or enter a URL.'


def apply_image_adjustments(image, brightness=0, contrast=0, saturation=0, sharpness=0):
    """Apply brightness, contrast, saturation, and sharpness adjustments.

    Args:
        image: PIL Image.
        brightness: -100 to 100 (0 = no change).
        contrast: -100 to 100 (0 = no change).
        saturation: -100 to 100 (0 = no change).
        sharpness: -100 to 100 (0 = no change).

    Returns:
        Adjusted PIL Image.
    """
    if brightness != 0:
        factor = 1.0 + (brightness / 100.0)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(max(0.0, factor))

    if contrast != 0:
        factor = 1.0 + (contrast / 100.0)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(max(0.0, factor))

    if saturation != 0:
        factor = 1.0 + (saturation / 100.0)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(max(0.0, factor))

    if sharpness != 0:
        factor = 1.0 + (sharpness / 100.0)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(max(0.0, factor))

    return image


# AI model metadata (used by both single-image and batch tabs)
AI_MODELS = {
    'depth-anything/Depth-Anything-V2-Small-hf': {
        'label': 'Depth Anything V2 Small',
        'size': '~100 MB',
        'speed': '⚡ Fast',
    },
    'depth-anything/Depth-Anything-V2-Base-hf': {
        'label': 'Depth Anything V2 Base',
        'size': '~400 MB',
        'speed': '🐢 Slower',
    },
    'Intel/dpt-hybrid-midas': {
        'label': 'DPT-Hybrid (MiDaS)',
        'size': '~500 MB',
        'speed': '🐢 Slower',
    },
}
