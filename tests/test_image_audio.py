"""Tests for image and audio loaders."""

import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestImageLoader:
    """Test image loading and saving."""

    def test_load_save_roundtrip(self):
        """Save and reload an image."""
        from PIL import Image
        from umc.data.image_loader import load_images, save_images

        # Create a test image
        rng = np.random.RandomState(42)
        pixels = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "test.png"
            Image.fromarray(pixels).save(str(img_path))

            # Load
            data = load_images(str(img_path))
            assert data.shape == (1, 64 * 64, 3)
            assert data.dtype == np.float32
            assert 0 <= data.min() and data.max() <= 1.0

            # Save
            out_path = Path(tmp) / "out.png"
            save_images(data, str(out_path), height=64, width=64)
            assert out_path.exists()

            # Reload and compare
            reloaded = load_images(str(out_path))
            np.testing.assert_allclose(data, reloaded, atol=1.0 / 255)

    def test_load_grayscale(self):
        """Grayscale images load with 1 channel."""
        from PIL import Image
        from umc.data.image_loader import load_images

        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "gray.png"
            gray = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
            Image.fromarray(gray, mode="L").save(str(img_path))

            data = load_images(str(img_path))
            assert data.shape == (1, 32 * 32, 1)

    def test_compress_image(self):
        """Images can be compressed via umc.compress()."""
        from PIL import Image
        from umc.data.image_loader import load_images
        import umc

        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "test.png"
            pixels = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(str(img_path))

            data = load_images(str(img_path))
            compressed = umc.compress(data, mode="lossless")
            decoded = umc.decompress(compressed)
            np.testing.assert_array_equal(data, decoded)

    def test_directory_load(self):
        """Loading a directory of images works."""
        from PIL import Image
        from umc.data.image_loader import load_images

        with tempfile.TemporaryDirectory() as tmp:
            for i in range(3):
                pixels = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(str(Path(tmp) / f"img{i}.png"))

            data = load_images(tmp)
            assert data.shape == (3, 16 * 16, 3)


class TestAudioLoader:
    """Test audio loading and saving."""

    def test_load_save_roundtrip(self):
        """Save and reload a WAV file."""
        import scipy.io.wavfile as wavfile
        from umc.data.audio_loader import load_audio, save_audio

        rng = np.random.RandomState(42)
        # Generate 1 second of mono audio at 16kHz
        sr = 16000
        audio = (rng.randn(sr) * 0.5).astype(np.float32)
        audio_int16 = (audio * 32767).astype(np.int16)

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "test.wav"
            wavfile.write(str(wav_path), sr, audio_int16)

            # Load
            data = load_audio(str(wav_path), window_size=1024)
            assert data.ndim == 3
            assert data.shape[1] == 1024
            assert data.shape[2] == 1  # mono
            assert data.dtype == np.float32

            # Save
            out_path = Path(tmp) / "out.wav"
            save_audio(data, str(out_path), sample_rate=sr)
            assert out_path.exists()

    def test_compress_audio(self):
        """Audio can be compressed via umc.compress()."""
        import scipy.io.wavfile as wavfile
        import umc

        rng = np.random.RandomState(42)
        sr = 16000
        audio = (rng.randn(sr) * 0.5).astype(np.float32)
        audio_int16 = (audio * 32767).astype(np.int16)

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "test.wav"
            wavfile.write(str(wav_path), sr, audio_int16)

            from umc.data.audio_loader import load_audio
            data = load_audio(str(wav_path))
            compressed = umc.compress(data, mode="lossless")
            decoded = umc.decompress(compressed)
            np.testing.assert_array_equal(data, decoded)

    def test_stereo_audio(self):
        """Stereo audio loads correctly."""
        import scipy.io.wavfile as wavfile
        from umc.data.audio_loader import load_audio

        rng = np.random.RandomState(42)
        sr = 16000
        stereo = (rng.randn(sr, 2) * 0.5 * 32767).astype(np.int16)

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "stereo.wav"
            wavfile.write(str(wav_path), sr, stereo)

            data = load_audio(str(wav_path), window_size=512)
            assert data.shape[1] == 512
            assert data.shape[2] == 2  # stereo

    def test_short_audio_padding(self):
        """Audio shorter than window_size gets padded."""
        import scipy.io.wavfile as wavfile
        from umc.data.audio_loader import load_audio

        sr = 16000
        short = np.zeros(100, dtype=np.int16)

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "short.wav"
            wavfile.write(str(wav_path), sr, short)

            data = load_audio(str(wav_path), window_size=1024)
            assert data.shape == (1, 1024, 1)
