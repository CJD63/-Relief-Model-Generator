"""Depth map generation module.

This module provides the DepthMapGenerator class for converting input images
into grayscale depth maps suitable for 3D relief generation.

Classes:
    DepthMapGenerator: Generates depth maps from images with optional inversion and smoothing.
"""

# depth.py - Vectorized depth map generation
from __future__ import annotations
import numpy as np
import cv2
import time
from PIL import Image
from typing import Optional, Callable

# Module-level pipeline cache to avoid reloading models within the same process
_pipeline_cache: dict = {}

class DepthMapGenerator:
    def __init__(self):
        self._image = None
        self._depth_map = None
        self._width = 0
        self._height = 0

    @property
    def width(self): return self._width
    @property
    def height(self): return self._height
    @property
    def depth_map(self): return self._depth_map

    def load_image(self, image_source, max_size=320):
        if image_source is None:
            raise ValueError('Image source cannot be None')
        img = image_source.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        self._image = img
        self._height, self._width = self._image.shape[:2]
        return self._image

    def create_depth_map(self, invert=False, gamma=1.0, blur_radius=1.0,
                         hill_removal_strength=0.0, small_feature_threshold=0.0,
                         enable_edge_detection=False, edge_strength=1.0,
                         progress_callback=None):
        if self._image is None:
            raise RuntimeError('No image loaded. Call load_image first.')
        if progress_callback: progress_callback(0.05, 'Converting to grayscale...')
        gray = cv2.cvtColor(self._image, cv2.COLOR_RGB2GRAY).astype(np.float64)
        if invert: gray = 255.0 - gray
        if progress_callback: progress_callback(0.15, 'Applying edge detection...')
        if enable_edge_detection and edge_strength > 0:
            gray = self._apply_edge_enhancement(gray, edge_strength)
        if progress_callback: progress_callback(0.25, 'Applying gamma correction...')
        if gamma != 1.0:
            gray = np.power(gray / 255.0, 1.0 / gamma) * 255.0
        if progress_callback: progress_callback(0.4, 'Applying bilateral filter...')
        if blur_radius > 0:
            gray = self._bilateral_filter(gray, blur_radius)
        if progress_callback: progress_callback(0.55, 'Removing small hills...')
        if hill_removal_strength > 0:
            gray = self._remove_small_hills(gray, hill_removal_strength)
        if progress_callback: progress_callback(0.7, 'Smoothing small features...')
        if small_feature_threshold > 0:
            gray = self._small_feature_smoothing(gray, small_feature_threshold)
        if progress_callback: progress_callback(0.85, 'Applying gradient limiting...')
        if blur_radius > 4:
            gray = self._gradient_limiting(gray, blur_radius)
        if progress_callback: progress_callback(0.95, 'Normalizing depth map...')
        self._depth_map = np.clip(gray, 0, 255).astype(np.uint8)
        if progress_callback: progress_callback(1.0, 'Complete')
        return self._depth_map

    def get_depth_map_image(self):
        if self._depth_map is None:
            raise RuntimeError('No depth map. Call create_depth_map first.')
        # Return as 2D grayscale for PIL compatibility (e.g., Image.fromarray expects 2D for 'L' mode)
        return self._depth_map

    def _bilateral_filter(self, img, radius):
        d = max(1, int(radius * 2))
        sigma = max(1, radius * 10)
        return cv2.bilateralFilter(img.astype(np.uint8), d, sigma, sigma).astype(np.float64)

    def _remove_small_hills(self, img, strength):
        if strength <= 0: return img
        result = img.copy()
        threshold = strength * 10
        for _ in range(3):
            blurred = cv2.GaussianBlur(result.astype(np.uint8), (5, 5), 0)
            diff = result - blurred
            mask = np.abs(diff) > threshold
            result[mask] = blurred[mask]
        return self._advanced_smoothing(result)

    def _advanced_smoothing(self, img):
        result = img.copy()
        for _ in range(2):
            blurred = cv2.GaussianBlur(result.astype(np.uint8), (3, 3), 0)
            result = 0.7 * result + 0.3 * blurred
        return result

    def _small_feature_smoothing(self, img, threshold):
        if threshold <= 0: return img
        result = img.copy()
        ksize = max(3, int(threshold / 2) * 2 + 1)
        blurred = cv2.GaussianBlur(result.astype(np.uint8), (ksize, ksize), 0)
        diff = np.abs(result - blurred)
        mask = diff < threshold * 5
        result[mask] = blurred[mask]
        return result

    def _gradient_limiting(self, img_array, blur_radius):
        """Limit gradient extremes using vectorized operations.
        
        Replaces pixel-by-pixel iteration with numpy vectorized operations
        for significant performance improvement on large images.
        """
        result = img_array.copy()
        max_grad = max(5, 30 - blur_radius * 5)
        iterations = max(2, int(blur_radius))
        
        for _ in range(iterations):
            # Vectorized: use rolling window mean via convolution-like approach
            # Pad image for edge handling
            padded = np.pad(result, 1, mode='edge')
            
            # Compute 3x3 neighborhood mean using vectorized operations
            # Each element in the output is the mean of a 3x3 window centered on input
            h, w = result.shape
            neighbor_sum = (
                padded[0:h, 0:w] + padded[0:h, 1:w+1] + padded[0:h, 2:w+2] +
                padded[1:h+1, 0:w] + padded[1:h+1, 1:w+1] + padded[1:h+1, 2:w+2] +
                padded[2:h+2, 0:w] + padded[2:h+2, 1:w+1] + padded[2:h+2, 2:w+2]
            )
            neighbor_mean = neighbor_sum / 9.0
            
            # Vectorized gradient limiting
            diff = neighbor_mean - result
            abs_diff = np.abs(diff)
            
            # Apply limiting only where gradient exceeds max_grad
            limit_mask = abs_diff > max_grad
            
            # For pixels needing limiting: move toward neighborhood mean
            # by exactly max_grad in the appropriate direction
            adjust = np.where(diff > 0, max_grad, -max_grad)
            result = np.where(limit_mask, result + adjust, result)
        
        return result

    def create_ai_depth_map(
        self,
        model_name: str = 'depth-anything/Depth-Anything-V2-Small-hf',
        invert: bool = False,
        blur_radius: float = 0.5,
        gamma: float = 1.0,
        progress_callback: Optional[Callable] = None,
        device: Optional[str] = None
    ) -> np.ndarray:
        '''
        Generate a true depth map using AI-based monocular depth estimation.

        Uses the Hugging Face transformers pipeline with depth-estimation models
        (e.g. Depth Anything V2) to estimate geometric depth from a single RGB image.
        This produces a proper height map where lighter pixels represent higher
        elevation (closer to the viewer), ready for relief generation.

        AI models output depth where higher values = further from camera.
        This method inverts that by default to produce a height map convention
        where higher values = closer to camera (raised areas).

        Args:
            model_name: Hugging Face model ID (default: Depth Anything V2 Small)
            invert: If True, keep raw depth convention (higher = further).
                    Default False produces height map convention (higher = closer).
            blur_radius: Post-process bilateral filter strength
            gamma: Gamma correction for contrast adjustment
            progress_callback: Optional callback(percent, message) for progress
            device: 'cuda', 'cpu', or None (auto-detect). Use 'cuda' for GPU acceleration.

        Returns:
            Depth map as uint8 numpy array (0-255, higher = closer / raised)
        '''
        if self._image is None:
            raise RuntimeError('No image loaded. Call load_image first.')

        if progress_callback:
            progress_callback(0.02, 'Loading AI depth estimation model...')

        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                'AI depth estimation requires the "transformers" library. '
                'Install it with: pip install transformers torch'
            )

        # Check if model is already cached to provide better progress messages
        model_cached = self._is_model_cached(model_name)
        if model_cached:
            if progress_callback:
                progress_callback(0.08, 'Loading AI model from cache...')
        else:
            if progress_callback:
                progress_callback(0.08, 'Downloading AI model (first run: ~100-400 MB, may take 1-3 minutes)...')

        # Determine device: auto-detect CUDA if available, else CPU
        target_device_name = device  # preserve original for GPU memory check
        if device is None:
            import torch
            device = 0 if torch.cuda.is_available() else -1
        elif device == 'cuda':
            device = 0
        elif device == 'cpu':
            device = -1

        # GPU memory safety check: if using CUDA, estimate memory and fall back if needed
        if device >= 0:
            import torch
            try:
                free_mem, total_mem = torch.cuda.mem_get_info(device)
                h, w = self._image.shape[:2]
                # Conservative estimate: model ~500MB + image ~(pixels * 4 * 3) + overhead
                estimated_bytes = 500 * 1024 * 1024 + h * w * 4 * 3 + 200 * 1024 * 1024
                if estimated_bytes > free_mem * 0.75:  # leave 25% headroom
                    warning_msg = (
                        f'GPU memory may be insufficient (free: {free_mem / 1e9:.1f} GB, '
                        f'estimated need: {estimated_bytes / 1e9:.1f} GB). Falling back to CPU.'
                    )
                    if progress_callback:
                        progress_callback(0.07, warning_msg)
                    device = -1
            except Exception:
                pass  # can't check GPU memory, proceed anyway

        # Create the depth estimation pipeline with retry on transient failures
        pipe = self._load_pipeline_with_retry(model_name, device, progress_callback)

        if progress_callback:
            progress_callback(0.25, 'Running AI depth estimation...')

        # Convert the stored RGB numpy array to a PIL Image for the pipeline
        pil_image = Image.fromarray(self._image)
        result = pipe(pil_image)

        if progress_callback:
            progress_callback(0.65, 'Normalizing depth map...')

        # Extract depth array from result
        depth = np.array(result['depth'], dtype=np.float64)

        # Normalize to 0-255 range (higher values = closer to camera)
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max > depth_min:
            depth_normalized = (depth - depth_min) / (depth_max - depth_min) * 255.0
        else:
            depth_normalized = np.full_like(depth, 128.0)

        gray = depth_normalized

        # AI models output depth as "higher = further from camera".
        # For height maps, we want "higher = closer" (raised areas).
        # Invert by default; set invert=True to keep raw depth convention.
        if not invert:
            gray = 255.0 - gray

        if progress_callback:
            progress_callback(0.75, 'Applying gamma correction...')

        # Optional: gamma correction
        if gamma != 1.0:
            gray = np.power(gray / 255.0, 1.0 / gamma) * 255.0

        if progress_callback:
            progress_callback(0.88, 'Applying bilateral filter...')

        # Optional: bilateral filter for smoothing
        if blur_radius > 0:
            gray = self._bilateral_filter(gray, blur_radius)

        if progress_callback:
            progress_callback(0.95, 'Finalizing depth map...')

        self._depth_map = np.clip(gray, 0, 255).astype(np.uint8)
        self._height, self._width = self._depth_map.shape[:2]

        if progress_callback:
            progress_callback(1.0, 'AI depth estimation complete')

        return self._depth_map

    @staticmethod
    def _load_pipeline_with_retry(
        model_name: str,
        device: int,
        progress_callback: Optional[Callable] = None,
        max_retries: int = 3,
        base_delay: float = 2.0
    ):
        '''Load the transformers pipeline with retry logic and exponential backoff.

        Handles transient failures like network timeouts, incomplete downloads,
        and CUDA OOM errors by retrying with increasing delays.

        Args:
            model_name: Hugging Face model ID
            device: Device integer (0 for CUDA, -1 for CPU)
            progress_callback: Optional progress reporter
            max_retries: Maximum number of retry attempts (default 3)
            base_delay: Initial delay between retries in seconds (doubles each retry)

        Returns:
            Loaded transformers pipeline

        Raises:
            RuntimeError: If all retry attempts fail
        '''
        import time as time_module
        from transformers import pipeline

        # Check module-level cache first
        cache_key = (model_name, device)
        if cache_key in _pipeline_cache:
            if progress_callback:
                progress_callback(0.15, 'Using cached AI model...')
            return _pipeline_cache[cache_key]

        last_error = None
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    if progress_callback:
                        progress_callback(0.10, 'Initializing AI model...')
                else:
                    delay = base_delay * (2 ** (attempt - 1))
                    if progress_callback:
                        progress_callback(
                            0.10,
                            f'Retry {attempt}/{max_retries - 1} after error: {str(last_error)[:60]}... '
                            f'(waiting {delay:.0f}s)'
                        )
                    time_module.sleep(delay)

                pipe = pipeline(
                    'depth-estimation',
                    model=model_name,
                    device=device
                )
                _pipeline_cache[cache_key] = pipe
                return pipe

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # CUDA OOM: fall back to CPU immediately
                if 'out of memory' in error_str or 'cuda' in error_str and 'memory' in error_str:
                    if device >= 0:
                        if progress_callback:
                            progress_callback(
                                0.10,
                                'CUDA out of memory. Falling back to CPU...'
                            )
                        try:
                            pipe = pipeline(
                                'depth-estimation',
                                model=model_name,
                                device=-1
                            )
                            # Cache under both keys so future calls skip the OOM path
                            _pipeline_cache[cache_key] = pipe
                            _pipeline_cache[(model_name, -1)] = pipe
                            return pipe
                        except Exception as cpu_e:
                            last_error = cpu_e
                            # Continue retry loop on CPU
                            device = -1
                            continue

                # Network/download errors: retry with backoff
                if attempt < max_retries - 1:
                    continue

        raise RuntimeError(
            f'Failed to load AI model "{model_name}" after {max_retries} attempts. '
            f'Last error: {str(last_error)[:120]}'
        )

    @staticmethod
    def _is_model_cached(model_name: str) -> bool:
        '''Check if a Hugging Face model is already downloaded to the local cache.'''
        try:
            from huggingface_hub import try_to_load_from_cache
            return try_to_load_from_cache(model_name, 'config.json') is not None
        except (ImportError, OSError):
            return False

    def _apply_edge_enhancement(self, img: np.ndarray, strength: float) -> np.ndarray:
        """Enhance edges in the depth map for crisper relief features.
        
        Uses Canny edge detection and Laplacian to find edges and blends them
        back into the grayscale image using fully vectorized operations for
        performance.
        
        Args:
            img: Input grayscale image (float64)
            strength: Edge enhancement strength (0.0 to 2.0)
            
        Returns:
            Edge-enhanced grayscale image
        """
        if strength <= 0:
            return img
        
        # Convert to uint8 for edge detection
        img_uint8 = img.astype(np.uint8)
        
        # Apply Canny edge detection with adaptive thresholds
        low_thresh = max(10, int(50 * strength))
        high_thresh = min(255, int(150 * strength))
        edges = cv2.Canny(img_uint8, low_thresh, high_thresh)
        
        # Dilate edges to make them more prominent
        kernel_size = max(1, int(3 * strength))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Create edge mask (0.0 to 1.0 range)
        edge_mask = edges_dilated.astype(np.float64) / 255.0
        
        # Calculate blend factor based on strength
        blend_factor = min(0.5, strength * 0.25)
        
        # Compute morphological gradient for edge direction detection
        kernel_3x3 = np.ones((3, 3), np.uint8)
        morph_grad = cv2.morphologyEx(img_uint8, cv2.MORPH_GRADIENT, kernel_3x3).astype(np.float64) / 255.0
        
        # Vectorized computation: create depression factor based on gradient
        # High gradient = strong edge = more depression
        depression_factor = np.clip(morph_grad * 2, 0, 1)
        
        # Combine edge mask with depression factor
        # Edge pixels get darkened based on both edge strength and local gradient
        edge_effect = edge_mask * blend_factor * (0.3 + 0.2 * depression_factor)
        
        # Apply darkening to edge pixels (vectorized operation)
        result = img.copy()
        edge_indices = edge_mask > 0
        if np.any(edge_indices):
            result[edge_indices] = img[edge_indices] * (1.0 - edge_effect[edge_indices])
        
        # Also add subtle sharpening using Laplacian for better edge definition
        if strength >= 0.5:
            laplacian = cv2.Laplacian(img_uint8, cv2.CV_64F)
            laplacian_mask = np.abs(laplacian) / 255.0
            sharpen_strength = min(0.15, strength * 0.075)
            
            # Only sharpen non-edge regions slightly
            non_edge_mask = (edge_mask == 0) & (laplacian_mask > 0.05)
            if np.any(non_edge_mask):
                sharpen_effect = laplacian_mask[non_edge_mask] * sharpen_strength
                result[non_edge_mask] = np.clip(
                    result[non_edge_mask] + laplacian[non_edge_mask] * sharpen_strength,
                    0, 255
                )
        
        return result
