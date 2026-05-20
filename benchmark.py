# benchmark.py - Performance benchmarks for 3D Relief Model Generator
import time
import numpy as np
from PIL import Image
from depth import DepthMapGenerator
from mesh import MeshBuilder
from stl import STLExporter
from relief import ReliefGenerator

def benchmark_mesh_generation(sizes=[50, 100, 200, 500]):
    print("=== Mesh Generation Benchmarks ===")
    builder = MeshBuilder()
    for size in sizes:
        depth_map = np.random.randint(0, 255, (size, size), dtype=np.uint8)
        builder.create_relief_mesh(depth_map, model_width=100.0, model_thickness=10.0)
        times = []
        for _ in range(5):
            start = time.perf_counter()
            vertices = builder.create_relief_mesh(depth_map, model_width=100.0, model_thickness=10.0)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        avg_time = sum(times) / len(times)
        vertices_per_sec = (size * size) / avg_time
        print(f"  {size}x{size}: {avg_time*1000:.2f}ms avg, {vertices_per_sec:,.0f} vertices/sec")

def benchmark_stl_export(sizes=[10, 20, 50, 100]):
    print("=== STL Export Benchmarks ===")
    exporter = STLExporter()
    for size in sizes:
        x = np.linspace(0, 100, size)
        y = np.linspace(0, 100, size)
        xx, yy = np.meshgrid(x, y)
        zz = np.random.rand(size, size) * 10.0
        surface = np.stack([xx, yy, zz], axis=-1)
        temp_file = f"bench_{size}.stl"
        exporter.export_stl(surface, size, size, 100.0/size, 2.0, temp_file)
        times = []
        for _ in range(3):
            start = time.perf_counter()
            exporter.export_stl(surface, size, size, 100.0/size, 2.0, temp_file)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        avg_time = sum(times) / len(times)
        triangles_per_sec = (size-1)*(size-1)*6 / avg_time
        print(f"  {size}x{size}: {avg_time*1000:.2f}ms avg, {triangles_per_sec:,.0f} triangles/sec")

def benchmark_depth_map_generation(sizes=[100, 200, 500, 1000]):
    print("=== Depth Map Generation Benchmarks ===")
    gen = DepthMapGenerator()
    for size in sizes:
        img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        gen.load_image(img, max_size=500)
        times = []
        for _ in range(3):
            start = time.perf_counter()
            gen.create_depth_map()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        avg_time = sum(times) / len(times)
        pixels_per_sec = (gen.height * gen.width) / avg_time
        print(f"  {size}x{size} -> {gen.height}x{gen.width}: {avg_time*1000:.2f}ms avg, {pixels_per_sec:,.0f} pixels/sec")

def benchmark_full_pipeline(sizes=[100, 200, 300]):
    print("=== Full Pipeline Benchmarks ===")
    for size in sizes:
        img_array = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        times = []
        for _ in range(2):
            start = time.perf_counter()
            generator = ReliefGenerator()
            generator.load_image_from_pil(img, max_size=200)
            generator.create_depth_map()
            generator.create_relief_mesh()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        avg_time = sum(times) / len(times)
        print(f"  {size}x{size} -> STL: {avg_time*1000:.2f}ms avg")
def benchmark_gradient_limiting_comparison(sizes=[100, 200, 500]):
    """Compare old nested-loop vs new vectorized gradient limiting."""
    print("\n=== Gradient Limiting: Loop vs Vectorized ===")
    
    def old_gradient_limiting(img_array, blur_radius):
        """Original nested-loop implementation."""
        result = img_array.copy()
        max_grad = max(5, 30 - blur_radius * 5)
        iterations = max(2, int(blur_radius))
        
        for _ in range(iterations):
            for y in range(1, result.shape[0] - 1):
                for x in range(1, result.shape[1] - 1):
                    # 3x3 neighborhood mean
                    neighborhood = [
                        result[y-1, x-1], result[y-1, x], result[y-1, x+1],
                        result[y, x-1],   result[y, x],   result[y, x+1],
                        result[y+1, x-1], result[y+1, x], result[y+1, x+1]
                    ]
                    neighbor_mean = sum(neighborhood) / 9.0
                    diff = neighbor_mean - result[y, x]
                    
                    if abs(diff) > max_grad:
                        if diff > 0:
                            result[y, x] = result[y, x] + max_grad
                        else:
                            result[y, x] = result[y, x] - max_grad
        return result
    
    gen = DepthMapGenerator()
    
    for size in sizes:
        img = np.random.randint(0, 255, (size, size), dtype=np.uint8).astype(np.float64)
        
        # Benchmark OLD loop version
        times_old = []
        for _ in range(3):
            start = time.perf_counter()
            old_gradient_limiting(img, 6)
            elapsed = time.perf_counter() - start
            times_old.append(elapsed)
        avg_old = sum(times_old) / len(times_old)
        
        # Benchmark NEW vectorized version
        times_new = []
        for _ in range(3):
            gen2 = DepthMapGenerator()
            gen2._image = np.zeros((size, size, 3), dtype=np.uint8)
            gen2._height, gen2._width = size, size
            start = time.perf_counter()
            gen2._gradient_limiting(img.copy(), 6)
            elapsed = time.perf_counter() - start
            times_new.append(elapsed)
        avg_new = sum(times_new) / len(times_new)
        
        speedup = avg_old / avg_new if avg_new > 0 else 0
        print(f"  {size}x{size}: Loop={avg_old*1000:.1f}ms, Vectorized={avg_new*1000:.1f}ms, Speedup={speedup:.1f}x")

if __name__ == "__main__":
    print("=" * 60)
    print("3D Relief Model Generator - Performance Benchmarks")
    print("=" * 60)
    benchmark_depth_map_generation()
    benchmark_mesh_generation()
    benchmark_stl_export()
    benchmark_full_pipeline()
    benchmark_gradient_limiting_comparison()
    print("=" * 60)
    print("Benchmarks complete!")
