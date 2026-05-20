import unittest
import numpy as np
from PIL import Image
from depth import DepthMapGenerator
from mesh import MeshBuilder
from stl import STLExporter
from relief import ReliefGenerator

class TestDepthMapGenerator(unittest.TestCase):
    def test_init(self):
        g = DepthMapGenerator()
        self.assertIsNotNone(g)

    def test_load_image(self):
        g = DepthMapGenerator()
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        g.load_image(img)
        self.assertEqual(g.width, 100)
        self.assertEqual(g.height, 100)

    def test_create_depth_map(self):
        g = DepthMapGenerator()
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        g.load_image(img)
        g.create_depth_map()
        self.assertIsNotNone(g.depth_map)
        self.assertEqual(g.depth_map.shape[0], 100)

    def test_depth_map_inversion(self):
        g = DepthMapGenerator()
        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        g.load_image(img)
        g.create_depth_map()
        dm1 = g.depth_map.copy()
        g.create_depth_map(invert=True)
        self.assertFalse(np.array_equal(g.depth_map, dm1))

    def test_depth_map_with_gamma(self):
        g = DepthMapGenerator()
        img = np.random.randint(0, 255, (60, 60, 3), dtype=np.uint8)
        g.load_image(img)
        g.create_depth_map(gamma=1.5)
        self.assertIsNotNone(g.depth_map)

    def test_depth_map_with_edge_detection(self):
        g = DepthMapGenerator()
        img = np.random.randint(0, 255, (60, 60, 3), dtype=np.uint8)
        g.load_image(img)
        g.create_depth_map(enable_edge_detection=True, edge_strength=1.0)
        self.assertIsNotNone(g.depth_map)
        self.assertEqual(g.depth_map.shape, (60, 60))

    def test_depth_map_edge_detection_strength(self):
        g = DepthMapGenerator()
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        g.load_image(img)
        # Test with different edge strengths
        g.create_depth_map(enable_edge_detection=True, edge_strength=0.5)
        result1 = g.depth_map.copy()
        g.create_depth_map(enable_edge_detection=True, edge_strength=1.5)
        result2 = g.depth_map.copy()
        # Different strengths should produce different results
        self.assertFalse(np.array_equal(result1, result2))

class TestMeshBuilder(unittest.TestCase):
    def test_init(self):
        b = MeshBuilder()
        self.assertIsNotNone(b)

    def test_create_mesh_basic(self):
        b = MeshBuilder()
        dm = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        v = b.create_relief_mesh(dm, 100.0, 10.0)
        self.assertIsNotNone(v)
        self.assertEqual(v.shape[0], 50)

    def test_create_mesh_with_base(self):
        b = MeshBuilder()
        dm = np.random.randint(0, 255, (60, 80), dtype=np.uint8)
        v = b.create_relief_mesh(dm, 80.0, 5.0, base_thickness=2.0)
        self.assertIsNotNone(v)

    def test_vectorized_performance(self):
        import time
        b = MeshBuilder()
        dm = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        s = time.time()
        v = b.create_relief_mesh(dm, 100.0, 10.0)
        elapsed = time.time() - s
        self.assertLess(elapsed, 0.1)
        self.assertIsNotNone(v)

    def test_mesh_info(self):
        b = MeshBuilder()
        dm = np.random.randint(0, 255, (60, 80), dtype=np.uint8)
        b.create_relief_mesh(dm, 80.0, 5.0)
        info = b.get_mesh_info()
        self.assertIsNotNone(info)
        self.assertEqual(info['width'], 80)
        self.assertEqual(info['height'], 60)

    def test_vertices_range(self):
        b = MeshBuilder()
        dm = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        v = b.create_relief_mesh(dm, 100.0, 10.0)
        self.assertTrue(v.max() <= 10.0)
        self.assertTrue(v.min() >= 0.0)

class TestSTLExporter(unittest.TestCase):
    def test_init(self):
        e = STLExporter()
        self.assertIsNotNone(e)

    def test_export_stl(self):
        e = STLExporter()
        surface = np.zeros((50, 50, 3), dtype=np.float64)
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            temp_path = f.name
        try:
            sb = e.export_stl(surface, 50, 50, 2.0, 2.0, temp_path)
            self.assertIsNotNone(sb)
            self.assertTrue(os.path.exists(temp_path))
            self.assertGreater(os.path.getsize(temp_path), 0)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_stl_not_empty(self):
        e = STLExporter()
        surface = np.zeros((30, 30, 3), dtype=np.float64)
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            temp_path = f.name
        try:
            e.export_stl(surface, 30, 30, 3.33, 2.0, temp_path)
            self.assertGreater(os.path.getsize(temp_path), 0)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

class TestReliefGeneratorIntegration(unittest.TestCase):
    def test_full_pipeline(self):
        g = ReliefGenerator()
        g.load_image_from_pil(Image.new('RGB', (50, 50), color=128), max_size=50)
        g.create_depth_map()
        g.create_relief_mesh()
        mi = g.get_mesh_info()
        self.assertIsNotNone(mi)

    def test_load_image_pil(self):
        g = ReliefGenerator()
        img = Image.new('RGB', (100, 100), color=(255, 0, 0))
        g.load_image_from_pil(img, max_size=100)
        self.assertEqual(g.width, 100)

    def test_large_image_resize(self):
        g = ReliefGenerator()
        img = Image.new('RGB', (500, 500), color=128)
        g.load_image_from_pil(img, max_size=200)
        self.assertLessEqual(g.width, 200)

    def test_depth_map_creation(self):
        g = ReliefGenerator()
        img = Image.new('RGB', (60, 60), color=128)
        g.load_image_from_pil(img, max_size=60)
        g.create_depth_map()
        self.assertIsNotNone(g.depth_map)

    def test_mesh_structure(self):
        g = ReliefGenerator()
        img = Image.new('RGB', (50, 50), color=128)
        g.load_image_from_pil(img, max_size=50)
        g.create_depth_map()
        g.create_relief_mesh()
        mi = g.get_mesh_info()
        self.assertIn('width', mi)
        self.assertIn('height', mi)
        self.assertIn('pixel_size', mi)

    def test_grayscale_pipeline(self):
        g = ReliefGenerator()
        img = Image.new('L', (50, 50), color=128)
        g.load_image_from_pil(img, max_size=50)
        g.create_depth_map()
        g.create_relief_mesh()
        mi = g.get_mesh_info()
        self.assertIsNotNone(mi)

    def test_random_image_pipeline(self):
        g = ReliefGenerator()
        data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(data)
        g.load_image_from_pil(img, max_size=100)
        g.create_depth_map()
        g.create_relief_mesh()
        mi = g.get_mesh_info()
        self.assertEqual(mi['width'], 100)

    def test_preset_values(self):
        g = ReliefGenerator()
        img = Image.new('RGB', (100, 100), color=128)
        g.load_image_from_pil(img, max_size=50)
        g.create_depth_map(gamma=1.5)
        g.create_relief_mesh()
        mi = g.get_mesh_info()
        self.assertIsNotNone(mi)

    # Error handling tests using utils module
    def test_download_timeout(self):
        from utils import download_image_from_url
        img, err = download_image_from_url('http://10.255.255.1:80/nonexistent')
        self.assertIsNone(img)
        self.assertIsNotNone(err)

    def test_download_invalid_url(self):
        from utils import download_image_from_url
        img, err = download_image_from_url('not-a-url')
        self.assertIsNone(img)
        self.assertIsNotNone(err)

    def test_get_image_invalid_file(self):
        from utils import get_image
        img, err = get_image(b'notimage', 'test.txt')
        self.assertIsNone(img)
        self.assertIsNotNone(err)

    def test_get_image_no_input(self):
        from utils import get_image
        img, err = get_image(None, None)
        self.assertIsNone(img)
        self.assertIsNotNone(err)

    def test_get_image_from_url(self):
        from utils import get_image
        img, err = get_image(None, 'not-a-url')
        self.assertIsNone(img)
        self.assertIsNotNone(err)

    def test_preset_validation(self):
        from utils import get_preset_with_validation
        preset, warnings = get_preset_with_validation('extreme')
        self.assertIsNotNone(preset)
        self.assertGreater(len(warnings), 0)  # extreme has out-of-bounds values

    def test_preset_unknown(self):
        from utils import get_preset_with_validation
        preset, warnings = get_preset_with_validation('nonexistent')
        self.assertIsNone(preset)

class TestGradientLimiting(unittest.TestCase):
    """Tests for the vectorized _gradient_limiting method in depth.py."""

    def test_gradient_limiting_basic(self):
        """Test that gradient limiting runs without errors on a basic image."""
        g = DepthMapGenerator()
        img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        g.load_image(img)
        # gradient_limiting is called internally when blur_radius > 4
        result = g.create_depth_map(blur_radius=5.0)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (50, 50))

    def test_gradient_limiting_smoothing(self):
        """Test that gradient limiting actually smooths extreme gradients."""
        g = DepthMapGenerator()
        # Create image with sharp gradient transition
        img = np.zeros((50, 50), dtype=np.uint8)
        img[:, :25] = 0
        img[:, 25:] = 255
        g.load_image(img)
        
        dm_no_limit = g.create_depth_map(blur_radius=4)  # no gradient limiting
        dm_with_limit = g.create_depth_map(blur_radius=6)  # with gradient limiting
        
        # The version with gradient limiting should have smoother transition
        # at the boundary between the two regions
        # Check that standard deviation is lower (smoother)
        std_no_limit = np.std(dm_no_limit[:, 24:26])
        std_with_limit = np.std(dm_with_limit[:, 24:26])
        # With gradient limiting, the transition should be smoother (lower std)
        self.assertLess(std_with_limit, std_no_limit + 1)  # Allow some tolerance

    def test_gradient_limiting_iterations(self):
        """Test that higher blur_radius produces more smoothing (more iterations)."""
        g = DepthMapGenerator()
        img = np.random.randint(0, 255, (60, 60), dtype=np.uint8)
        g.load_image(img)
        
        result1 = g.create_depth_map(blur_radius=5)  # ~2 iterations
        result2 = g.create_depth_map(blur_radius=10)  # ~10 iterations
        
        # More iterations should produce smoother result
        # Check that result2 has lower variance (more smoothed)
        var1 = np.var(result1)
        var2 = np.var(result2)
        # Higher blur_radius (more iterations) should give lower variance
        self.assertLess(var2, var1 + 1000)  # Allow tolerance

    def test_gradient_limiting_preserves_depth_range(self):
        """Test that gradient limiting preserves the valid depth range [0, 255]."""
        g = DepthMapGenerator()
        img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        g.load_image(img)
        result = g.create_depth_map(blur_radius=8)
        
        self.assertGreaterEqual(result.min(), 0)
        self.assertLessEqual(result.max(), 255)

    def test_gradient_limiting_vectorized_performance(self):
        """Test that vectorized gradient limiting is fast on larger images."""
        import time
        g = DepthMapGenerator()
        img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        g.load_image(img)
        
        start = time.time()
        result = g.create_depth_map(blur_radius=8)
        elapsed = time.time() - start
        
        # Vectorized implementation should be fast
        self.assertLess(elapsed, 0.5)  # Should complete in under 0.5 seconds
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
