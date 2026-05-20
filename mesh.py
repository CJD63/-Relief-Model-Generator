"""Mesh generation module.

This module provides the MeshBuilder class for converting depth maps into
3D vertex grids representing relief surfaces.

Classes:
    MeshBuilder: Creates 3D mesh vertices from depth maps using vectorized operations.
"""

# mesh.py - Vectorized 3D mesh generation
# Type hints and docstrings added throughout

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


class MeshBuilder:
    '''Handles conversion of depth maps to 3D triangle meshes.'''

    def __init__(self) -> None:
        self._vertices: Optional[np.ndarray] = None
        self._width: int = 0
        self._height: int = 0
        self._model_width: float = 0.0
        self._model_thickness: float = 0.0
        self._base_thickness: float = 0.0
        self._pixel_size: float = 0.0

    @property
    def vertices(self) -> Optional[np.ndarray]:
        return self._vertices

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def pixel_size(self) -> float:
        return self._pixel_size

    def create_relief_mesh(
        self,
        depth_map: np.ndarray,
        model_width: float = 50.0,
        model_thickness: float = 5.0,
        base_thickness: float = 2.0
    ) -> np.ndarray:
        '''
        Create 3D relief mesh from depth map using vectorized operations.

        Args:
            depth_map: 2D array of depth values (0-255)
            model_width: Width of model in mm
            model_thickness: Maximum relief height in mm
            base_thickness: Base plate thickness in mm

        Returns:
            2D array of vertex Z values with shape (height, width)

        Raises:
            ValueError: If depth_map is invalid
        '''
        if depth_map is None or depth_map.size == 0:
            raise ValueError('Invalid depth map provided')

        self._width = depth_map.shape[1]
        self._height = depth_map.shape[0]
        self._model_width = model_width
        self._model_thickness = model_thickness
        self._base_thickness = base_thickness
        self._pixel_size = model_width / self._width

        # Vectorized vertex calculation - NO LOOPS
        self._vertices = (depth_map.astype(np.float64) / 255.0) * model_thickness

        return self._vertices

    def get_mesh_info(self) -> dict:
        '''Return mesh dimensions and metadata.'''
        return {
            'width': self._width,
            'height': self._height,
            'model_width': self._model_width,
            'model_height': self._model_width * self._height / self._width,
            'model_thickness': self._model_thickness,
            'base_thickness': self._base_thickness,
            'total_height': self._model_thickness + self._base_thickness,
            'pixel_size': self._pixel_size,
            'triangle_count': self._calculate_triangle_count()
        }

    def _calculate_triangle_count(self) -> int:
        '''Calculate total triangles for the mesh.
        
        Breakdown:
        - Surface: 2 triangles per quad for (w-1)*(h-1) quads
        - Bottom: same as surface (full coverage)
        - Sides: 4 rectangular sides, each needing 2 triangles per unit
          * Front/back: (w-1) quads * 2 triangles each
          * Left/right: (h-1) quads * 2 triangles each
          * Total sides: 2*(w-1)*2 + 2*(h-1)*2 = 4*(w-1 + h-1)
        '''
        w, h = self._width, self._height
        surface = (w - 1) * (h - 1) * 2  # Top face
        bottom = surface  # Bottom face (mirrored)
        # 4 side faces, each with 2 triangles per unit length
        sides = 2 * (w - 1) * 2 + 2 * (h - 1) * 2
        return surface + bottom + sides

    def get_surface_vertices(self) -> np.ndarray:
        '''Get the 3D surface vertices as (height, width, 3) array.'''
        if self._vertices is None:
            raise ValueError('Mesh not created. Call create_relief_mesh() first.')

        yy, xx, _ = self._create_xy_grid()
        vertices_3d = np.stack([xx, yy, self._vertices], axis=-1)

        return vertices_3d

    def _create_xy_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''Create 2D grid of X and Y coordinates for mesh vertices.
        
        Returns:
            Tuple of (yy, xx, y_coords) where yy/xx are meshgrid arrays
            and y_coords is the 1D y coordinate array (reversed for Y-down convention).
        '''
        pixel_size = self._pixel_size
        height = self._height
        width = self._width
        
        y_coords = np.arange(height) * pixel_size
        y_coords = (height - 1 - y_coords)  # Reverse for Y-down convention
        x_coords = np.arange(width) * pixel_size
        
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
        return yy, xx, y_coords

    def get_base_vertices(self, z_bottom: float) -> Tuple[np.ndarray, np.ndarray]:
        '''Get the 3D base/bottom vertices.'''
        yy, xx, _ = self._create_xy_grid()

        bottom_verts = np.stack([xx, yy, np.full_like(yy, z_bottom)], axis=-1)
        top_verts = np.stack([xx, yy, np.zeros_like(yy)], axis=-1)

        return bottom_verts, top_verts
