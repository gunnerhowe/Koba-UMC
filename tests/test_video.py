"""Tests for video loader and UMC video compression."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from umc.data.video_loader import load_video, save_video, get_video_info


def _create_test_video(path: str, n_frames: int = 10, width: int = 32, height: int = 32):
    """Create a small test video file."""
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (width, height), isColor=True)
    rng = np.random.RandomState(42)
    for _ in range(n_frames):
        frame = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestVideoLoader:
    """Test video loading and saving."""

    def test_load_video(self):
        """Load a video file into float32 array."""
        with tempfile.TemporaryDirectory() as tmp:
            vid_path = str(Path(tmp) / "test.mp4")
            _create_test_video(vid_path, n_frames=5, width=16, height=16)

            data = load_video(vid_path)
            assert data.ndim == 3
            assert data.shape[0] == 5  # n_frames
            assert data.shape[1] == 16 * 16  # height * width
            assert data.shape[2] == 3  # RGB
            assert data.dtype == np.float32
            assert 0 <= data.min() and data.max() <= 1.0

    def test_load_max_frames(self):
        """max_frames limits the number of loaded frames."""
        with tempfile.TemporaryDirectory() as tmp:
            vid_path = str(Path(tmp) / "test.mp4")
            _create_test_video(vid_path, n_frames=10)

            data = load_video(vid_path, max_frames=3)
            assert data.shape[0] == 3

    def test_save_and_reload(self):
        """Save and reload roundtrip."""
        with tempfile.TemporaryDirectory() as tmp:
            vid_path = str(Path(tmp) / "test.mp4")
            _create_test_video(vid_path, n_frames=5, width=16, height=16)

            data = load_video(vid_path)
            out_path = str(Path(tmp) / "out.mp4")
            save_video(data, out_path, height=16, width=16, fps=30.0)

            assert Path(out_path).exists()
            reloaded = load_video(out_path)
            assert reloaded.shape[0] == data.shape[0]
            assert reloaded.shape[2] == data.shape[2]

    def test_get_video_info(self):
        """get_video_info returns metadata."""
        with tempfile.TemporaryDirectory() as tmp:
            vid_path = str(Path(tmp) / "test.mp4")
            _create_test_video(vid_path, n_frames=10, width=32, height=24)

            info = get_video_info(vid_path)
            assert info["width"] == 32
            assert info["height"] == 24
            assert info["n_frames"] == 10
            assert info["fps"] > 0

    def test_compress_video(self):
        """Video can be compressed via umc.compress()."""
        import umc

        with tempfile.TemporaryDirectory() as tmp:
            vid_path = str(Path(tmp) / "test.mp4")
            _create_test_video(vid_path, n_frames=5, width=16, height=16)

            data = load_video(vid_path)
            compressed = umc.compress(data, mode="lossless")
            decoded = umc.decompress(compressed)
            np.testing.assert_array_equal(data, decoded)

    def test_quantized_8_video(self):
        """Video compresses well with quantized_8 mode."""
        import umc

        with tempfile.TemporaryDirectory() as tmp:
            vid_path = str(Path(tmp) / "test.mp4")
            _create_test_video(vid_path, n_frames=10, width=32, height=32)

            data = load_video(vid_path)
            compressed = umc.compress(data, mode="quantized_8")
            decoded = umc.decompress(compressed)
            assert decoded.shape == data.shape
            # quantized_8 should compress video well (image-like data)
            ratio = data.nbytes / len(compressed)
            assert ratio > 2.0

    def test_file_not_found(self):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_video("/nonexistent/video.mp4")
