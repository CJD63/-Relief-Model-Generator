"""Relief generation orchestrator module.

This module provides the ReliefGenerator class that orchestrates the complete
pipeline from image to 3D STL relief model.

Classes:
    ReliefGenerator: High-level interface for generating relief models from images.
"""

# relief.py - 3D Relief Model Generator
# Refactored: delegates to specialized modules (depth.py, mesh.py, stl.py)

from __future__ import annotations

import numpy as np
from PIL import Image
from typing import Optional
import requests
from io import BytesIO

from depth import DepthMapGenerator
from mesh import MeshBuilder
from stl import STLExporter


class ReliefGenerator:
    '''
    High-level orchestrator for 3D relief generation.
    Delegates to specialized modules: DepthMapGenerator, MeshBuilder, STLExporter.
    '''

    def __init__(self) -> None:
        self._depth_gen = DepthMapGenerator()
        self._mesh_builder = MeshBuilder()
        self._stl_exporter = STLExporter()
        self._stl_bytes: Optional[bytes] = None
        self._mesh_info: dict = {}

    @property
    def width(self) -> int:
        return self._depth_gen.width

    @property
    def height(self) -> int:
        return self._depth_gen.height

    @property
    def depth_map(self) -> Optional[np.ndarray]:
        return self._depth_gen.depth_map

    def load_image(self, image_source: str | Image.Image, max_size: int = 320) -> np.ndarray:
        '''
        Load image from file path, URL, or PIL Image.

        Args:
            image_source: File path, URL, or PIL Image
            max_size: Maximum dimension for resize

        Returns:
            Resized RGB image as numpy array
        '''
        if isinstance(image_source, str):
            if image_source.startswith(('http://', 'https://')):
                response = requests.get(image_source, timeout=30)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_source)
        elif isinstance(image_source, Image.Image):
            image = image_source
        else:
            raise ValueError('Invalid image source')

        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_array = np.array(image)
        return self._depth_gen.load_image(img_array, max_size)

    def create_depth_map(
        self,
        image: Optional[np.ndarray] = None,
        invert_depth: bool = False,
        blur_radius: float = 0.5,
        gamma: float = 1.1,
        enable_smoothing: bool = True,
        enable_hill_removal: bool = True,
        hill_removal_strength: float = 5.0,
        enable_edge_detection: bool = False,
        edge_strength: float = 1.0,
        progress_callback=None
    ) -> np.ndarray:
        '''
        Generate depth map from image.

        Args:
            image: RGB image array (uses loaded if None)
            invert_depth: Swap raised/recessed areas
            blur_radius: Smoothing strength (0.5 - 10)
            gamma: Contrast adjustment (0.5 - 3.0)
            enable_smoothing: Apply bilateral + NLM denoising
            enable_hill_removal: Remove small artifacts
            hill_removal_strength: Artifact removal strength (1 - 10)
            enable_edge_detection: Enhance edges for crisper relief
            edge_strength: Edge enhancement strength (0.0 to 2.0)
            progress_callback: Optional callback(percent) for progress

        Returns:
            Depth map as uint8 array
        '''
        return self._depth_gen.create_depth_map(
            invert=invert_depth,
            gamma=gamma,
            blur_radius=blur_radius,
            hill_removal_strength=hill_removal_strength if enable_hill_removal else 0.0,
            enable_edge_detection=enable_edge_detection,
            edge_strength=edge_strength,
            progress_callback=progress_callback
        )

    def get_depth_map_image(self) -> Image.Image:
        '''Return depth map as PIL Image.'''
        dm = self._depth_gen.get_depth_map_image()
        return Image.fromarray(dm, 'L')

    def create_relief_mesh(
        self,
        model_width: float = 50.0,
        model_thickness: float = 5.0,
        base_thickness: float = 2.0
    ) -> bool:
        '''
        Create 3D relief mesh from depth map.

        Args:
            model_width: Width of model in mm
            model_thickness: Maximum relief height in mm
            base_thickness: Base plate thickness in mm

        Returns:
            True if successful
        '''
        depth_map = self._depth_gen.depth_map
        if depth_map is None:
            raise ValueError('Depth map not generated. Call create_depth_map() first.')

        self._mesh_builder.create_relief_mesh(
            depth_map=depth_map,
            model_width=model_width,
            model_thickness=model_thickness,
            base_thickness=base_thickness
        )

        self._mesh_info = self._mesh_builder.get_mesh_info()
        return True

    def save_stl(self, output_path: str) -> str:
        '''
        Export mesh to binary STL file.

        Args:
            output_path: Path to output STL file

        Returns:
            Path to created file
        '''
        if self._mesh_builder.vertices is None:
            raise ValueError('Mesh not created. Call create_relief_mesh() first.')

        surface_vertices = self._mesh_builder.get_surface_vertices()
        info = self._mesh_info

        return self._stl_exporter.export_stl(
            surface_vertices=surface_vertices,
            width=info['width'],
            height=info['height'],
            pixel_size=info['pixel_size'],
            base_thickness=info['base_thickness'],
            output_path=output_path
        )

    def get_mesh_info(self) -> dict:
        '''Return mesh dimensions and metadata.'''
        return self._mesh_info.copy()

    def load_image_from_pil(self, pil_image: Image.Image, max_size: int = 320) -> np.ndarray:
        '''Load PIL Image directly (convenience method).'''
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        img_array = np.array(pil_image)
        return self._depth_gen.load_image(img_array, max_size)
