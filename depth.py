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
from typing import Optional, Callable

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
