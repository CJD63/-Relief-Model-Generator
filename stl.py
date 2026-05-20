"""STL binary export module.

This module provides the STLExporter class for exporting 3D surface meshes
to binary STL format files for 3D printing.

Classes:
    STLExporter: Exports surface vertices to binary STL format.
"""

# stl.py - STL binary file export
from __future__ import annotations
import struct
import numpy as np
from typing import Optional, Tuple


class STLExporter:
    def __init__(self):
        self._vertices = None
        self._width = 0
        self._height = 0
        self._pixel_size = 0.0
        self._base_thickness = 0.0

    def export_stl(self, surface_vertices, width, height, pixel_size, base_thickness, output_path):
        if surface_vertices is None:
            raise ValueError("Surface vertices required")
        self._vertices = surface_vertices
        self._width = width
        self._height = height
        self._pixel_size = pixel_size
        self._base_thickness = base_thickness
        stl_data = self._generate_stl_binary()
        with open(output_path, "wb") as f:
            f.write(stl_data)
        return output_path

    def _generate_stl_binary(self) -> bytearray:
        width = self._width
        height = self._height
        pixel_size = self._pixel_size
        base_thickness = self._base_thickness
        vertices = self._vertices

        num_surface = (width - 1) * (height - 1) * 2
        num_bottom = (width - 1) * (height - 1) * 2
        num_sides = 2 * (width - 1) + 2 * (height - 1)
        total_triangles = num_surface + num_bottom + num_sides * 2

        buffer_size = 84 + total_triangles * 50
        stl_data = bytearray(buffer_size)

        struct.pack_into("<I", stl_data, 80, total_triangles)

        offset = 84
        offset = self._add_surface_triangles(stl_data, offset, vertices, width, height, pixel_size, base_thickness)
        offset = self._add_bottom_triangles(stl_data, offset, width, height, pixel_size, base_thickness)
        offset = self._add_side_triangles(stl_data, offset, vertices, width, height, pixel_size, base_thickness)

        return stl_data

    def _add_surface_triangles(self, stl_data, offset, vertices, width, height, pixel_size, base_thickness):
        w = width - 1
        h = height - 1

        for y in range(h):
            for x in range(w):
                z00 = vertices[y, x, 2]
                z01 = vertices[y, x + 1, 2]
                z10 = vertices[y + 1, x, 2]
                z11 = vertices[y + 1, x + 1, 2]

                x0 = x * pixel_size
                x1 = (x + 1) * pixel_size
                y0 = (height - y - 1) * pixel_size
                y1 = (height - y - 2) * pixel_size

                base0 = base_thickness
                base1 = base_thickness

                v1 = (x0, y0, base0)
                v2 = (x0, y0, z00)
                v3 = (x1, y0, z01)
                normal = self._calculate_normal(v1, v2, v3)
                struct.pack_into("<fff", stl_data, offset, *normal)
                offset += 12
                struct.pack_into("<fff", stl_data, offset, *v1)
                offset += 12
                struct.pack_into("<fff", stl_data, offset, *v2)
                offset += 12
                struct.pack_into("<fff", stl_data, offset, *v3)
                offset += 12

                v1 = (x0, y0, base0)
                v2 = (x1, y0, z01)
                v3 = (x1, y0, base0)
                normal = self._calculate_normal(v1, v2, v3)
                struct.pack_into("<fff", stl_data, offset, *normal)
                offset += 12
                struct.pack_into("<fff", stl_data, offset, *v1)
                offset += 12
                struct.pack_into("<fff", stl_data, offset, *v2)
                offset += 12
                struct.pack_into("<fff", stl_data, offset, *v3)
                offset += 12

        return offset

    def _add_side_triangles(self, stl_data, offset, vertices, width, height, pixel_size, base_thickness):
        w = width
        h = height

        normal = (0.0, -1.0, 0.0)
        for x in range(w - 1):
            z1 = float(vertices[height - 1, x, 2])
            z2 = float(vertices[height - 1, x + 1, 2])
            x0 = x * pixel_size
            x1 = (x + 1) * pixel_size
            y0 = 0.0

            struct.pack_into("<fff", stl_data, offset, *normal)
            offset += 12
            struct.pack_into("<fff", stl_data, offset, x0, y0, base_thickness)
            offset += 12
            struct.pack_into("<fff", stl_data, offset, x0, y0, z1)
            offset += 12
            struct.pack_into("<fff", stl_data, offset, x1, y0, z2)
            offset += 12

            struct.pack_into("<fff", stl_data, offset, *normal)
            offset += 12
            struct.pack_into("<fff", stl_data, offset, x0, y0, base_thickness)
            offset += 12
            struct.pack_into("<fff", stl_data, offset, x1, y0, z2)
            offset += 12
            struct.pack_into("<fff", stl_data, offset, x1, y0, base_thickness)
            offset += 12

        normal = (0.0, 1.0, 0.0)
        for x in range(w - 1):
            z1 = float(vertices[0, x, 2])
            z2 = float(vertices[0, x + 1, 2])
            x0 = x * pixel_size
            x1 = (x + 1) * pixel_size
            y0 = (height - 1) * pixel_size

            struct.pack_into("<fff", stl_data, offset, *normal)
            offset += 12
            struct.pack_into("<fff", stl_data, offset, x0, y0, base_thickness)
            offset += 12
            struct.pack_into("<fff", stl_data, offset, x1, y0, z2)
            offset += 12
            struct.pack_into("<fff", stl_data, offset, x0, y0, z1)
            offset += 12

            struct.pack_into("<fff", stl_data, offset, *normal)
            offset += 12
            struct.pack_into("<fff", stl_data, offset, x0, y0, base_thickness)
            offset += 12
            struct.pack_into("<fff", stl_data, offset, x1, y0, base_thickness)
            offset += 12
            struct.pack_into("<fff", stl_data, offset, x1, y0, z2)
            offset += 12

        normal = (-1.0, 0.0, 0.0)
        for y in range(h - 1):
            z1 = float(vertices[y, 0, 2])
            z2 = float(vertices[y + 1, 0, 2])
            y0 = (height - y - 1) * pixel_size
            y1 = (height - y - 2) * pixel_size

            struct.pack_into("<fff", stl_data, offset, *normal)
            offset += 12
            struct.pack_into("<fff", stl_data, offset, 0.0, y0, base_thickness)
            offset += 12
            struct.pack_into("<fff", stl_data, offset, 0.0, y0, z1)
            offset += 12
            struct.pack_into("<fff", stl_data, offset, 0.0, y1, z2)
            offset += 12

            struct.pack_into("<fff", stl_data, offset, *normal)
            offset += 12
            struct.pack_into("<fff", stl_data, offset, 0.0, y0, base_thickness)
            offset += 12
            struct.pack_into("<fff", stl_data, offset, 0.0, y1, z2)
            offset += 12
            struct.pack_into("<fff", stl_data, offset, 0.0, y1, base_thickness)
            offset += 12

        normal = (1.0, 0.0, 0.0)
        for y in range(h - 1):
            z1 = float(vertices[y, width - 1, 2])
            z2 = float(vertices[y + 1, width - 1, 2])
            y0 = (height - y - 1) * pixel_size
            y1 = (height - y - 2) * pixel_size
            x0 = (width - 1) * pixel_size

            struct.pack_into("<fff", stl_data, offset, *normal)
            offset += 12
            struct.pack_into("<fff", stl_data, offset, x0, y0, base_thickness)
            offset += 12
            struct.pack_into("<fff", stl_data, offset, x0, y1, z2)
            offset += 12
            struct.pack_into("<fff", stl_data, offset, x0, y0, z1)
            offset += 12

            struct.pack_into("<fff", stl_data, offset, *normal)
            offset += 12
            struct.pack_into("<fff", stl_data, offset, x0, y0, base_thickness)
            offset += 12
            struct.pack_into("<fff", stl_data, offset, x0, y1, base_thickness)
            offset += 12
            struct.pack_into("<fff", stl_data, offset, x0, y1, z2)
            offset += 12

        return offset

    def _add_bottom_triangles(self, stl_data, offset, width, height, pixel_size, base_thickness):
        w = width - 1
        h = height - 1

        normal = (0.0, 0.0, -1.0)

        for y in range(h):
            for x in range(w):
                x0 = x * pixel_size
                x1 = (x + 1) * pixel_size
                y0 = (height - y - 1) * pixel_size
                y1 = (height - y - 2) * pixel_size

                struct.pack_into("<fff", stl_data, offset, *normal)
                offset += 12
                struct.pack_into("<fff", stl_data, offset, x0, y0, base_thickness)
                offset += 12
                struct.pack_into("<fff", stl_data, offset, x1, y0, base_thickness)
                offset += 12
                struct.pack_into("<fff", stl_data, offset, x1, y1, base_thickness)
                offset += 12

                struct.pack_into("<fff", stl_data, offset, *normal)
                offset += 12
                struct.pack_into("<fff", stl_data, offset, x0, y0, base_thickness)
                offset += 12
                struct.pack_into("<fff", stl_data, offset, x1, y1, base_thickness)
                offset += 12
                struct.pack_into("<fff", stl_data, offset, x0, y1, base_thickness)
                offset += 12

        return offset

    @staticmethod
    def _calculate_normal(v1: Tuple[float, float, float],
                          v2: Tuple[float, float, float],
                          v3: Tuple[float, float, float]) -> Tuple[float, float, float]:
        ax = v2[0] - v1[0]
        ay = v2[1] - v1[1]
        az = v2[2] - v1[2]
        bx = v3[0] - v1[0]
        by = v3[1] - v1[1]
        bz = v3[2] - v1[2]

        nx = ay * bz - az * by
        ny = az * bx - ax * bz
        nz = ax * by - ay * bx

        length = (nx * nx + ny * ny + nz * nz) ** 0.5
        if length > 0.0:
            return (nx / length, ny / length, nz / length)
        return (0.0, 0.0, 1.0)