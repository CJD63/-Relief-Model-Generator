"""Unit tests for relief, depth, mesh, and STL modules.

Tests cover:
- DepthMapGenerator: image loading, grayscale depth maps, AI depth estimation
- ReliefGenerator: orchestration pipeline
- MeshBuilder: 3D mesh generation from depth maps
- STLExporter: binary STL export
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import tempfile
import os

from depth import DepthMapGenerator
from relief import ReliefGenerator
from mesh import MeshBuilder
from stl import STLExporter


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_rgb_image() -> np.ndarray:
    """Create a small synthetic RGB test image with known patterns."""
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    # Left half red, right half blue gradient
    arr[:, :16, 0] = 200  # R channel on left
    arr[:, 16:, 2] = np.linspace(50, 200, 16).astype(np.uint8)  # B gradient
    arr[:, :, 1] = 100  # G channel constant
    return arr


@pytest.fixture
def sample_gray_depth() -> np.ndarray:
    """Create a small synthetic depth map with known values."""
    depth = np.zeros((16, 16), dtype=np.uint8)
    # Center area raised (white = 255), edges lowered (black = 0)
    depth[4:12, 4:12] = 255
    return depth


@pytest.fixture
def depth_gen() -> DepthMapGenerator:
    return DepthMapGenerator()


@pytest.fixture
def relief_gen() -> ReliefGenerator:
    return ReliefGenerator()


@pytest.fixture
def mesh_builder() -> MeshBuilder:
    return MeshBuilder()


@pytest.fixture
def stl_exporter() -> STLExporter:
    return STLExporter()


@pytest.fixture
def mock_ai_depth_result() -> np.ndarray:
    """Mock AI depth estimation result: 32x32 depth map."""
    depth = np.zeros((32, 32), dtype=np.float64)
    # Gradient from top-left (near=255) to bottom-right (far=0)
    for y in range(32):
        for x in range(32):
            depth[y, x] = 255.0 - (x + y) * 4.0
    return np.clip(depth, 0, 255)


# ── DepthMapGenerator Tests ───────────────────────────────────────────────

class TestDepthMapGenerator:
    """Tests for DepthMapGenerator class."""

    def test_load_image_rgb(self, depth_gen, sample_rgb_image):
        """Should load an RGB image and set width/height."""
        result = depth_gen.load_image(sample_rgb_image, max_size=64)
        assert result is not None
        assert depth_gen.width == 32
        assert depth_gen.height == 32

    def test_load_image_resize(self, depth_gen):
        """Should resize large images to max_size."""
        large_img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = depth_gen.load_image(large_img, max_size=50)
        # max dimension should be <= 50
        assert max(depth_gen.height, depth_gen.width) <= 50

    def test_load_image_none_raises(self, depth_gen):
        """Should raise ValueError for None image."""
        with pytest.raises(ValueError, match='cannot be None'):
            depth_gen.load_image(None)

    def test_create_depth_map_basic(self, depth_gen, sample_rgb_image):
        """Should create a depth map from loaded image."""
        depth_gen.load_image(sample_rgb_image)
        result = depth_gen.create_depth_map()
        assert result is not None
        assert result.shape == (32, 32)
        assert result.dtype == np.uint8
        assert 0 <= result.min() <= result.max() <= 255

    def test_create_depth_map_invert(self, depth_gen, sample_rgb_image):
        """Inverted depth map should be the complement of non-inverted."""
        depth_gen.load_image(sample_rgb_image)
        normal = depth_gen.create_depth_map(invert=False)
        inverted = depth_gen.create_depth_map(invert=True)
        # Inverted should be roughly 255 - normal
        assert np.allclose(normal.astype(float), 255.0 - inverted.astype(float), atol=2)

    def test_create_depth_map_no_image_raises(self, depth_gen):
        """Should raise RuntimeError if no image loaded."""
        with pytest.raises(RuntimeError, match='No image loaded'):
            depth_gen.create_depth_map()

    def test_get_depth_map_image(self, depth_gen, sample_rgb_image):
        """Should return a 2D uint8 array for PIL compatibility."""
        depth_gen.load_image(sample_rgb_image)
        depth_gen.create_depth_map()
        dm = depth_gen.get_depth_map_image()
        assert dm is not None
        assert dm.ndim == 2
        assert dm.dtype == np.uint8

    def test_get_depth_map_image_before_creation_raises(self, depth_gen):
        """Should raise RuntimeError if depth map not created yet."""
        with pytest.raises(RuntimeError, match='No depth map'):
            depth_gen.get_depth_map_image()

    def test_bilateral_filter(self, depth_gen, sample_rgb_image):
        """Bilateral filter should produce valid output."""
        depth_gen.load_image(sample_rgb_image)
        result = depth_gen.create_depth_map(blur_radius=3.0)
        assert result.shape == (32, 32)
        assert result.dtype == np.uint8

    def test_gamma_correction(self, depth_gen, sample_rgb_image):
        """Gamma correction should shift intensity distribution."""
        depth_gen.load_image(sample_rgb_image)
        normal = depth_gen.create_depth_map(gamma=1.0)
        boosted = depth_gen.create_depth_map(gamma=2.0)
        # Different gamma should produce different results
        assert not np.array_equal(normal, boosted)

    def test_edge_enhancement(self, depth_gen, sample_rgb_image):
        """Edge enhancement should produce valid output."""
        depth_gen.load_image(sample_rgb_image)
        result = depth_gen.create_depth_map(enable_edge_detection=True, edge_strength=1.0)
        assert result.shape == (32, 32)
        assert result.dtype == np.uint8

    def test_hill_removal(self, depth_gen, sample_rgb_image):
        """Hill removal should produce valid output and smooth features."""
        depth_gen.load_image(sample_rgb_image)
        result = depth_gen.create_depth_map(hill_removal_strength=5.0)
        assert result.shape == (32, 32)
        assert result.dtype == np.uint8

    # ── AI Depth Estimation Tests ────────────────────────────────────

    def test_create_ai_depth_map_mock(self, depth_gen, sample_rgb_image, mock_ai_depth_result):
        """AI depth map should work with mocked pipeline."""
        depth_gen.load_image(sample_rgb_image)

        # Create mock pipeline result: transformers pipeline returns dict with PIL Image
        mock_depth_pil = Image.fromarray(mock_ai_depth_result.astype(np.uint8))
        mock_pipe = MagicMock()
        mock_pipe.return_value = {'depth': mock_depth_pil}

        with patch('depth.DepthMapGenerator._is_model_cached', return_value=True):
            with patch('transformers.pipeline', return_value=mock_pipe):
                result = depth_gen.create_ai_depth_map()

        assert result is not None
        assert result.shape == (32, 32)
        assert result.dtype == np.uint8
        assert 0 <= result.min() <= result.max() <= 255

    def test_create_ai_depth_map_invert(self, depth_gen, sample_rgb_image, mock_ai_depth_result):
        """AI depth map with invert=True keeps raw depth convention."""
        depth_gen.load_image(sample_rgb_image)

        mock_depth_pil = Image.fromarray(mock_ai_depth_result.astype(np.uint8))
        mock_pipe = MagicMock()
        mock_pipe.return_value = {'depth': mock_depth_pil}

        with patch('depth.DepthMapGenerator._is_model_cached', return_value=True):
            with patch('transformers.pipeline', return_value=mock_pipe):
                normal = depth_gen.create_ai_depth_map(invert=False)  # height map convention
                inverted = depth_gen.create_ai_depth_map(invert=True)  # raw depth convention

        # The two outputs should be complements (height map vs raw depth)
        assert not np.array_equal(normal, inverted)

    def test_create_ai_depth_map_no_image_raises(self, depth_gen):
        """AI depth map should raise if no image loaded."""
        with pytest.raises(RuntimeError, match='No image loaded'):
            depth_gen.create_ai_depth_map()

    def test_is_model_cached_nonexistent(self):
        """_is_model_cached should return False for a model that doesn't exist."""
        result = DepthMapGenerator._is_model_cached('nonexistent/model-12345')
        assert result is False

    def test_is_model_cached_handles_import_error(self):
        """_is_model_cached should return False when huggingface_hub is not available."""
        with patch('depth.DepthMapGenerator._is_model_cached', return_value=False):
            result = DepthMapGenerator._is_model_cached('any/model')
            assert result is False

    def test_create_ai_depth_map_with_gamma(self, depth_gen, sample_rgb_image, mock_ai_depth_result):
        """AI depth map with gamma correction should work."""
        depth_gen.load_image(sample_rgb_image)

        mock_depth_pil = Image.fromarray(mock_ai_depth_result.astype(np.uint8))
        mock_pipe = MagicMock()
        mock_pipe.return_value = {'depth': mock_depth_pil}

        with patch('depth.DepthMapGenerator._is_model_cached', return_value=True):
            with patch('transformers.pipeline', return_value=mock_pipe):
                result = depth_gen.create_ai_depth_map(gamma=1.5)

        assert result.shape == (32, 32)
        assert result.dtype == np.uint8

    def test_create_ai_depth_map_with_blur(self, depth_gen, sample_rgb_image, mock_ai_depth_result):
        """AI depth map with bilateral filter should work."""
        depth_gen.load_image(sample_rgb_image)

        mock_depth_pil = Image.fromarray(mock_ai_depth_result.astype(np.uint8))
        mock_pipe = MagicMock()
        mock_pipe.return_value = {'depth': mock_depth_pil}

        with patch('depth.DepthMapGenerator._is_model_cached', return_value=True):
            with patch('transformers.pipeline', return_value=mock_pipe):
                result = depth_gen.create_ai_depth_map(blur_radius=2.0)

        assert result.shape == (32, 32)
        assert result.dtype == np.uint8

    def test_create_ai_depth_map_progress_callback(self, depth_gen, sample_rgb_image, mock_ai_depth_result):
        """Progress callback should be invoked during AI depth map creation."""
        depth_gen.load_image(sample_rgb_image)
        calls = []

        mock_depth_pil = Image.fromarray(mock_ai_depth_result.astype(np.uint8))
        mock_pipe = MagicMock()
        mock_pipe.return_value = {'depth': mock_depth_pil}

        with patch('depth.DepthMapGenerator._is_model_cached', return_value=True):
            with patch('transformers.pipeline', return_value=mock_pipe):
                result = depth_gen.create_ai_depth_map(
                    progress_callback=lambda pct, msg: calls.append((pct, msg))
                )

        assert len(calls) >= 5  # Should have multiple progress updates
        # First call should be at low percent
        assert calls[0][0] < 0.1
        # Last call should be near 100%
        assert calls[-1][0] >= 0.95


# ── ReliefGenerator Tests ─────────────────────────────────────────────────

class TestReliefGenerator:
    """Tests for ReliefGenerator orchestrator."""

    def test_load_image_from_pil(self, relief_gen):
        """Should load a PIL Image and return numpy array."""
        pil_img = Image.new('RGB', (64, 48), color=(128, 64, 32))
        result = relief_gen.load_image_from_pil(pil_img, max_size=100)
        assert result is not None
        assert relief_gen.width == 64
        assert relief_gen.height == 48

    def test_load_image_from_pil_grayscale(self, relief_gen):
        """Should convert grayscale PIL Image to RGB before loading."""
        pil_img = Image.new('L', (32, 32), color=128)
        result = relief_gen.load_image_from_pil(pil_img)
        assert result is not None
        assert relief_gen.width == 32

    def test_create_depth_map(self, relief_gen):
        """Should create depth map via orchestrator."""
        pil_img = Image.new('RGB', (32, 32), color=(100, 150, 200))
        relief_gen.load_image_from_pil(pil_img)
        result = relief_gen.create_depth_map(invert_depth=False)
        assert result is not None
        assert result.dtype == np.uint8

    def test_create_ai_depth_map_through_orchestrator(self, relief_gen, mock_ai_depth_result):
        """Orchestrator should delegate AI depth creation to DepthMapGenerator."""
        pil_img = Image.new('RGB', (32, 32), color=(100, 150, 200))
        relief_gen.load_image_from_pil(pil_img)

        mock_depth_pil = Image.fromarray(mock_ai_depth_result.astype(np.uint8))
        mock_pipe = MagicMock()
        mock_pipe.return_value = {'depth': mock_depth_pil}

        with patch('depth.DepthMapGenerator._is_model_cached', return_value=True):
            with patch('transformers.pipeline', return_value=mock_pipe):
                result = relief_gen.create_ai_depth_map(invert_depth=False)

        assert result is not None
        assert result.shape == (32, 32)
        assert result.dtype == np.uint8

    def test_create_relief_mesh(self, relief_gen):
        """Should create relief mesh from depth map."""
        pil_img = Image.new('RGB', (16, 16), color=(128, 128, 128))
        relief_gen.load_image_from_pil(pil_img)
        relief_gen.create_depth_map()
        success = relief_gen.create_relief_mesh(
            model_width=50.0, model_thickness=5.0, base_thickness=2.0
        )
        assert success is True

    def test_create_relief_mesh_no_depth_raises(self, relief_gen):
        """Should raise if depth map not created before mesh."""
        with pytest.raises(ValueError, match='Depth map not generated'):
            relief_gen.create_relief_mesh()

    def test_get_mesh_info(self, relief_gen):
        """Should return mesh metadata."""
        pil_img = Image.new('RGB', (16, 16), color=(128, 128, 128))
        relief_gen.load_image_from_pil(pil_img)
        relief_gen.create_depth_map()
        relief_gen.create_relief_mesh(model_width=50.0)
        info = relief_gen.get_mesh_info()
        assert 'width' in info
        assert 'height' in info
        assert 'triangle_count' in info
        assert info['width'] == 16
        assert info['height'] == 16

    def test_save_stl(self, relief_gen):
        """Should save STL file to disk."""
        pil_img = Image.new('RGB', (16, 16), color=(128, 128, 128))
        relief_gen.load_image_from_pil(pil_img)
        relief_gen.create_depth_map()
        relief_gen.create_relief_mesh(model_width=50.0)

        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            temp_path = f.name

        try:
            result_path = relief_gen.save_stl(temp_path)
            assert result_path == temp_path
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_get_depth_map_image(self, relief_gen):
        """Should return depth map as PIL Image."""
        pil_img = Image.new('RGB', (32, 32), color=(128, 128, 128))
        relief_gen.load_image_from_pil(pil_img)
        relief_gen.create_depth_map()
        dm_img = relief_gen.get_depth_map_image()
        assert isinstance(dm_img, Image.Image)
        assert dm_img.size == (32, 32)

    def test_save_stl_no_mesh_raises(self, relief_gen):
        """Should raise if mesh not created before STL save."""
        with pytest.raises(ValueError, match='Mesh not created'):
            relief_gen.save_stl('test.stl')


# ── MeshBuilder Tests ─────────────────────────────────────────────────────

class TestMeshBuilder:
    """Tests for MeshBuilder class."""

    def test_create_relief_mesh(self, mesh_builder, sample_gray_depth):
        """Should create vertices from depth map."""
        vertices = mesh_builder.create_relief_mesh(
            sample_gray_depth, model_width=50.0, model_thickness=5.0
        )
        assert vertices is not None
        assert vertices.shape == (16, 16)

    def test_create_relief_mesh_scale(self, mesh_builder, sample_gray_depth):
        """Vertex heights should be proportional to model_thickness."""
        thin = mesh_builder.create_relief_mesh(
            sample_gray_depth, model_width=50.0, model_thickness=2.0
        )
        thick = mesh_builder.create_relief_mesh(
            sample_gray_depth, model_width=50.0, model_thickness=10.0
        )
        # Center area (max height) should scale with thickness
        thin_center_max = thin[4:12, 4:12].max()
        thick_center_max = thick[4:12, 4:12].max()
        assert abs(thick_center_max / thin_center_max - 5.0) < 0.01

    def test_create_relief_mesh_invalid_raises(self, mesh_builder):
        """Should raise for invalid depth map."""
        with pytest.raises(ValueError):
            mesh_builder.create_relief_mesh(np.array([]))
        with pytest.raises(ValueError):
            mesh_builder.create_relief_mesh(None)

    def test_get_mesh_info(self, mesh_builder, sample_gray_depth):
        """Should return correct metadata."""
        mesh_builder.create_relief_mesh(sample_gray_depth, model_width=50.0)
        info = mesh_builder.get_mesh_info()
        assert info['width'] == 16
        assert info['height'] == 16
        assert info['model_width'] == 50.0
        assert info['triangle_count'] > 0

    def test_get_surface_vertices(self, mesh_builder, sample_gray_depth):
        """Surface vertices should be 3D (height, width, 3)."""
        mesh_builder.create_relief_mesh(sample_gray_depth)
        surface = mesh_builder.get_surface_vertices()
        assert surface.ndim == 3
        assert surface.shape[2] == 3  # X, Y, Z
        assert surface.shape[:2] == (16, 16)

    def test_get_base_vertices(self, mesh_builder, sample_gray_depth):
        """Base vertices should be 3D with correct Z values."""
        mesh_builder.create_relief_mesh(sample_gray_depth, model_width=50.0)
        bottom, top = mesh_builder.get_base_vertices(z_bottom=2.0)
        assert bottom.ndim == 3
        assert top.ndim == 3
        assert bottom.shape[2] == 3


# ── STLExporter Tests ─────────────────────────────────────────────────────

class TestSTLExporter:
    """Tests for STLExporter class."""

    def test_export_stl(self, stl_exporter, sample_gray_depth):
        """Should produce a valid binary STL file."""
        mesh = MeshBuilder()
        mesh.create_relief_mesh(sample_gray_depth, model_width=50.0)
        surface = mesh.get_surface_vertices()
        info = mesh.get_mesh_info()

        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            temp_path = f.name

        try:
            result = stl_exporter.export_stl(
                surface_vertices=surface,
                width=info['width'],
                height=info['height'],
                pixel_size=info['pixel_size'],
                base_thickness=info['base_thickness'],
                output_path=temp_path,
            )
            assert result == temp_path
            assert os.path.exists(temp_path)
            # STL files are at least 84 bytes (header + triangle count)
            assert os.path.getsize(temp_path) >= 84

            # Verify STL header
            with open(temp_path, 'rb') as f:
                header = f.read(80)
                # Header can be anything, but bytes 80-83 are triangle count (uint32)
                f.seek(80)
                count_bytes = f.read(4)
                triangle_count = int.from_bytes(count_bytes, 'little')
                assert triangle_count > 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_export_stl_none_vertices_raises(self, stl_exporter):
        """Should raise for None vertices."""
        with pytest.raises(ValueError):
            stl_exporter.export_stl(None, 16, 16, 1.0, 2.0, 'test.stl')

    def test_calculate_normal(self, stl_exporter):
        """Normal calculation should produce unit vectors."""
        normal = stl_exporter._calculate_normal(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
        # Normal should be (0, 0, 1) for XY plane triangle
        assert abs(normal[0]) < 1e-10
        assert abs(normal[1]) < 1e-10
        assert abs(normal[2] - 1.0) < 1e-10

    def test_calculate_normal_degenerate(self, stl_exporter):
        """Degenerate (collinear) vertices should produce fallback normal."""
        normal = stl_exporter._calculate_normal(
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.0),
            (1.0, 1.0, 0.0),
        )
        # Fallback for zero-area triangle is (0, 0, 1)
        assert normal == (0.0, 0.0, 1.0)


# ── Retry Logic & GPU Fallback Tests ──────────────────────────────────────

# Clear the module-level pipeline cache before retry/GPU tests
# (it persists across tests and would return stale mocks)
@pytest.fixture(autouse=True)
def _clear_pipeline_cache():
    from depth import _pipeline_cache
    _pipeline_cache.clear()
    yield
    _pipeline_cache.clear()


class TestRetryLogic:
    """Tests for _load_pipeline_with_retry in DepthMapGenerator."""

    def test_load_pipeline_succeeds_first_try(self):
        """Should return pipeline on first successful attempt."""
        mock_pipe = MagicMock()
        with patch('transformers.pipeline', return_value=mock_pipe):
            result = DepthMapGenerator._load_pipeline_with_retry(
                'test/model', device=-1
            )
            assert result is mock_pipe

    def test_load_pipeline_retries_on_transient_error(self):
        """Should retry on transient errors and eventually succeed."""
        mock_pipe = MagicMock()
        call_count = [0]

        def flaky_pipeline(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError('Network timeout')
            return mock_pipe

        with patch('transformers.pipeline', side_effect=flaky_pipeline):
            with patch('time.sleep', return_value=None):
                result = DepthMapGenerator._load_pipeline_with_retry(
                    'test/model', device=-1
                )
                assert result is mock_pipe
                assert call_count[0] == 3

    def test_load_pipeline_raises_after_max_retries(self):
        """Should raise RuntimeError after all retries exhausted."""
        call_count = [0]

        def failing_pipeline(*args, **kwargs):
            call_count[0] += 1
            raise ConnectionError('Network timeout')

        with patch('transformers.pipeline', side_effect=failing_pipeline):
            with patch('time.sleep', return_value=None):
                with pytest.raises(RuntimeError, match='Failed to load AI model'):
                    DepthMapGenerator._load_pipeline_with_retry(
                        'test/model', device=-1, max_retries=3
                    )
                assert call_count[0] == 3

    def test_load_pipeline_progress_callback_on_retry(self):
        """Should invoke progress callback with retry messages."""
        call_count = [0]
        progress_calls = []

        def flaky_pipeline(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError('Network timeout')
            return MagicMock()

        with patch('transformers.pipeline', side_effect=flaky_pipeline):
            with patch('time.sleep', return_value=None):
                result = DepthMapGenerator._load_pipeline_with_retry(
                    'test/model', device=-1,
                    progress_callback=lambda pct, msg: progress_calls.append((pct, msg))
                )
        # Should have at least one progress call (initial + retry)
        assert len(progress_calls) >= 2
        # Second call should mention retry
        assert 'Retry' in progress_calls[1][1] or 'retry' in progress_calls[1][1].lower()

    def test_load_pipeline_exponential_backoff(self):
        """Should use increasing delays between retries."""
        call_count = [0]
        sleep_times = []

        def flaky_pipeline(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 4:
                raise ConnectionError('transient')
            return MagicMock()

        with patch('transformers.pipeline', side_effect=flaky_pipeline):
            with patch('time.sleep', side_effect=lambda s: sleep_times.append(s)):
                DepthMapGenerator._load_pipeline_with_retry(
                    'test/model', device=-1, max_retries=4, base_delay=2.0
                )
        # Delays should be: 2.0, 4.0, 8.0 (doubling each time)
        assert len(sleep_times) == 3
        assert sleep_times[0] == 2.0
        assert sleep_times[1] == 4.0
        assert sleep_times[2] == 8.0

    def test_load_pipeline_cuda_oom_fallback(self):
        """Should fall back to CPU on CUDA out-of-memory error."""
        mock_cpu_pipe = MagicMock()
        pipeline_calls = []

        def oom_then_cpu(*args, **kwargs):
            pipeline_calls.append(kwargs.get('device'))
            if kwargs.get('device', -1) >= 0:
                raise RuntimeError('CUDA out of memory')
            return mock_cpu_pipe

        with patch('transformers.pipeline', side_effect=oom_then_cpu):
            result = DepthMapGenerator._load_pipeline_with_retry(
                'test/model', device=0
            )
            assert result is mock_cpu_pipe
            # First call tried CUDA (device=0), second call fell back to CPU (device=-1)
            assert len(pipeline_calls) >= 2
            assert pipeline_calls[0] == 0
            assert pipeline_calls[1] == -1

    def test_load_pipeline_cuda_oom_progress_callback(self):
        """Should report CUDA fallback via progress callback."""
        progress_calls = []

        def oom_pipeline(*args, **kwargs):
            if kwargs.get('device', -1) >= 0:
                raise RuntimeError('CUDA out of memory')
            return MagicMock()

        with patch('transformers.pipeline', side_effect=oom_pipeline):
            result = DepthMapGenerator._load_pipeline_with_retry(
                'test/model', device=0,
                progress_callback=lambda pct, msg: progress_calls.append((pct, msg))
            )
        # Should have a fallback message
        fallback_msgs = [c[1] for c in progress_calls if 'CPU' in c[1] or 'falling back' in c[1].lower()]
        assert len(fallback_msgs) >= 1

    def test_load_pipeline_non_retryable_error(self):
        """Should raise immediately on ImportError (not retry)."""
        with patch('transformers.pipeline', side_effect=ImportError('No module named transformers')):
            with patch('time.sleep', return_value=None):
                with pytest.raises(RuntimeError, match='Failed to load AI model'):
                    DepthMapGenerator._load_pipeline_with_retry(
                        'test/model', device=-1, max_retries=2
                    )


class TestGPUFallback:
    """Tests for GPU memory safety in create_ai_depth_map."""

    def test_gpu_memory_check_falls_back_to_cpu(self, depth_gen, sample_rgb_image, mock_ai_depth_result):
        """Should fall back to CPU when GPU memory is insufficient."""
        depth_gen.load_image(sample_rgb_image)

        mock_depth_pil = Image.fromarray(mock_ai_depth_result.astype(np.uint8))
        mock_pipe = MagicMock()
        mock_pipe.return_value = {'depth': mock_depth_pil}

        # Mock CUDA available but with very low free memory
        with patch('depth.DepthMapGenerator._is_model_cached', return_value=True):
            with patch('transformers.pipeline', return_value=mock_pipe):
                with patch('torch.cuda.is_available', return_value=True):
                    with patch('torch.cuda.mem_get_info', return_value=(100 * 1024 * 1024, 8 * 1024 * 1024 * 1024)):
                        result = depth_gen.create_ai_depth_map(device='cuda')
                        assert result is not None
                        assert result.shape == (32, 32)

    def test_gpu_memory_check_proceeds_when_sufficient(self, depth_gen, sample_rgb_image, mock_ai_depth_result):
        """Should use GPU when memory is sufficient."""
        depth_gen.load_image(sample_rgb_image)

        mock_depth_pil = Image.fromarray(mock_ai_depth_result.astype(np.uint8))
        mock_pipe = MagicMock()
        mock_pipe.return_value = {'depth': mock_depth_pil}
        pipeline_device = []

        def track_device(*args, **kwargs):
            pipeline_device.append(kwargs.get('device'))
            return mock_pipe

        with patch('depth.DepthMapGenerator._is_model_cached', return_value=True):
            with patch('transformers.pipeline', side_effect=track_device):
                with patch('torch.cuda.is_available', return_value=True):
                    with patch('torch.cuda.mem_get_info', return_value=(6 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024)):
                        result = depth_gen.create_ai_depth_map(device='cuda')
                        assert result is not None
                        # Should have used CUDA (device=0), not CPU (-1)
                        assert pipeline_device[0] == 0

    def test_gpu_memory_check_handles_exception(self, depth_gen, sample_rgb_image, mock_ai_depth_result):
        """Should proceed when GPU memory check raises an exception."""
        depth_gen.load_image(sample_rgb_image)

        mock_depth_pil = Image.fromarray(mock_ai_depth_result.astype(np.uint8))
        mock_pipe = MagicMock()
        mock_pipe.return_value = {'depth': mock_depth_pil}

        with patch('depth.DepthMapGenerator._is_model_cached', return_value=True):
            with patch('transformers.pipeline', return_value=mock_pipe):
                with patch('torch.cuda.is_available', return_value=True):
                    with patch('torch.cuda.mem_get_info', side_effect=RuntimeError('GPU not accessible')):
                        result = depth_gen.create_ai_depth_map(device='cuda')
                        assert result is not None
                        assert result.shape == (32, 32)

    def test_cpu_device_no_gpu_check(self, depth_gen, sample_rgb_image, mock_ai_depth_result):
        """Should skip GPU memory check when device is CPU."""
        depth_gen.load_image(sample_rgb_image)

        mock_depth_pil = Image.fromarray(mock_ai_depth_result.astype(np.uint8))
        mock_pipe = MagicMock()
        mock_pipe.return_value = {'depth': mock_depth_pil}

        pipeline_device = []

        def track_device(*args, **kwargs):
            pipeline_device.append(kwargs.get('device'))
            return mock_pipe

        with patch('depth.DepthMapGenerator._is_model_cached', return_value=True):
            with patch('transformers.pipeline', side_effect=track_device):
                result = depth_gen.create_ai_depth_map(device='cpu')
                assert result is not None
                assert pipeline_device[0] == -1
