"""
zinohk.image.generator
========================
ZINOHK Image Generation — text → image from scratch.

Pipeline (all ZINOHK-native, no diffusion model):
  text
    ↓ text encoder (TF-IDF or sentence embeddings)
  concept vector
    ↓ spike encoder (GAP-05)
  concept spikes (512,)
    ↓ spatial decoder (Hebbian trained)
  patch grid (16×16 patches × 64 pixels each)
    ↓ patch assembler
  image (256×256 RGB)

Training:
  - Feed (text, image) pairs
  - Hebbian: concept spikes that co-occur with image patches → strengthen
  - Each patch learned independently (local, no backprop)

Resolution roadmap:
  Phase 15a: 64×64  (proof of concept)
  Phase 15b: 128×128 (recognisable shapes)
  Phase 15c: 256×256 (detail, texture)
  Phase 15d: 512×512 (photorealistic, needs Kaggle GPU)
"""

import numpy as np
from typing import List, Optional, Tuple


class PatchDecoder:
    """
    Decodes spike patterns into image patches.

    Hebbian trained: when concept spikes co-occur
    with a pixel patch, strengthen the connection.

    Parameters
    ----------
    n_nodes    : number of concept spike nodes
    patch_size : pixels per patch (patch_size × patch_size)
    n_channels : 1 (grayscale) or 3 (RGB)
    lr         : Hebbian learning rate
    """

    def __init__(
        self,
        n_nodes:    int   = 512,
        patch_size: int   = 8,
        n_channels: int   = 3,
        lr:         float = 0.01,
    ):
        self.n_nodes    = n_nodes
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.lr         = lr

        patch_dim = patch_size * patch_size * n_channels
        # Weight matrix: spikes → patch pixels
        self.W = np.random.uniform(
            0.0, 0.1, (n_nodes, patch_dim)
        ).astype(np.float32)

        self.n_updates = 0

    def decode_patch(self, spikes: np.ndarray) -> np.ndarray:
        """
        Decode concept spikes into one image patch.

        Parameters
        ----------
        spikes : (n_nodes,) concept spike pattern

        Returns
        -------
        patch : (patch_size, patch_size, n_channels) uint8
        """
        raw   = spikes @ self.W   # (patch_dim,)
        # Normalise to [0, 255]
        if raw.max() > raw.min():
            raw = (raw - raw.min()) / (raw.max() - raw.min())
        raw   = (raw * 255).astype(np.uint8)
        patch = raw.reshape(
            self.patch_size, self.patch_size, self.n_channels)
        return patch

    def learn_patch(
        self,
        spikes: np.ndarray,
        patch:  np.ndarray,
    ) -> None:
        """
        Hebbian update: strengthen spikes→patch connection.

        Parameters
        ----------
        spikes : (n_nodes,) concept spike pattern
        patch  : (patch_size, patch_size, n_channels) pixel values
        """
        patch_flat = patch.flatten().astype(np.float32) / 255.0
        self.W    += self.lr * np.outer(spikes, patch_flat)
        self.W     = np.clip(self.W, 0.0, 1.0)
        self.n_updates += 1


class SpatialLayout:
    """
    Controls how patches are arranged in the image.

    Divides image into a grid of patches and tracks
    which concept spikes are responsible for each patch.

    Parameters
    ----------
    image_size : image width and height in pixels
    patch_size : size of each patch in pixels
    """

    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
    ):
        self.image_size  = image_size
        self.patch_size  = patch_size
        self.n_patches_per_side = image_size // patch_size
        self.n_patches   = self.n_patches_per_side ** 2

    def patch_position(self, patch_idx: int) -> Tuple[int, int]:
        """Return (row, col) pixel position of patch."""
        row = (patch_idx // self.n_patches_per_side) * self.patch_size
        col = (patch_idx  % self.n_patches_per_side) * self.patch_size
        return row, col

    def assemble(
        self,
        patches: List[np.ndarray],
        n_channels: int = 3,
    ) -> np.ndarray:
        """
        Assemble list of patches into full image.

        Parameters
        ----------
        patches : list of (patch_size, patch_size, n_channels)

        Returns
        -------
        image : (image_size, image_size, n_channels) uint8
        """
        image = np.zeros(
            (self.image_size, self.image_size, n_channels),
            dtype=np.uint8)

        for i, patch in enumerate(patches[:self.n_patches]):
            row, col = self.patch_position(i)
            image[row:row+self.patch_size,
                  col:col+self.patch_size] = patch

        return image


class ConceptEncoder:
    """
    Encodes text concepts into sparse spike patterns.

    Uses hash-based sparse encoding — no external encoder needed.
    Each unique concept maps to a unique sparse spike pattern.
    Concepts that share words share some spikes.
    """

    def __init__(self, n_nodes: int = 512):
        self.n_nodes  = n_nodes
        self.cache    = {}   # cache encoded concepts

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text into sparse spike pattern.

        Parameters
        ----------
        text : concept description

        Returns
        -------
        spikes : (n_nodes,) sparse binary array
        """
        if text in self.cache:
            return self.cache[text]

        spikes = np.zeros(self.n_nodes, dtype=np.float32)
        words  = text.lower().split()

        for word in words:
            # Primary activation
            idx = hash(word) % self.n_nodes
            spikes[idx] = 1.0
            # Secondary activations (word context)
            spikes[(idx + 7)  % self.n_nodes] = 0.7
            spikes[(idx + 13) % self.n_nodes] = 0.5
            spikes[(idx + 31) % self.n_nodes] = 0.3

        # Threshold to keep sparse (~10% active)
        threshold = 0.25
        spikes    = (spikes >= threshold).astype(np.float32)

        self.cache[text] = spikes
        return spikes


class ZINOHKImage:
    """
    Complete ZINOHK image generation pipeline.

    text → concept spikes → patch grid → image

    No diffusion model. No third-party generator.
    All ZINOHK-native with Hebbian learning.

    Parameters
    ----------
    image_size : output image resolution (square)
    patch_size : size of each patch tile
    n_channels : 1=grayscale, 3=RGB
    n_nodes    : concept spike nodes

    Example
    -------
    >>> gen = ZINOHKImage(image_size=64)
    >>> img = gen.generate("a blue sky with white clouds")
    >>> gen.save(img, "sky.png")
    """

    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        n_channels: int = 3,
        n_nodes:    int = 512,
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.n_nodes    = n_nodes

        self.concept_enc = ConceptEncoder(n_nodes)
        self.layout      = SpatialLayout(image_size, patch_size)
        self.decoder     = PatchDecoder(
            n_nodes, patch_size, n_channels)

        self.n_generated = 0

    def generate(self, text: str) -> np.ndarray:
        """
        Generate image from text description.

        Parameters
        ----------
        text : concept description

        Returns
        -------
        image : (image_size, image_size, n_channels) uint8
        """
        self.n_generated += 1

        # Step 1: text → concept spikes
        spikes = self.concept_enc.encode(text)

        # Step 2: generate each patch from spikes
        # Different patches get slightly different spike patterns
        # to create spatial variation
        patches = []
        n_patches = self.layout.n_patches

        for patch_idx in range(n_patches):
            # Add spatial noise for variation across patches
            spatial_noise = np.zeros(self.n_nodes, dtype=np.float32)
            # Each patch position activates a few unique nodes
            pos_node = (patch_idx * 17) % self.n_nodes
            spatial_noise[pos_node] = 0.3
            spatial_noise[(pos_node + 7) % self.n_nodes] = 0.2

            patch_spikes = np.clip(spikes + spatial_noise, 0, 1)
            patch        = self.decoder.decode_patch(patch_spikes)
            patches.append(patch)

        # Step 3: assemble patches into image
        image = self.layout.assemble(patches, self.n_channels)
        return image

    def learn(self, text: str, image: np.ndarray) -> None:
        """
        Teach ZINOHK what a concept looks like.

        Parameters
        ----------
        text  : description of the image
        image : (image_size, image_size, n_channels) reference image
        """
        spikes = self.concept_enc.encode(text)

        # Learn each patch independently (local Hebbian)
        for patch_idx in range(self.layout.n_patches):
            row, col = self.layout.patch_position(patch_idx)
            patch    = image[
                row:row+self.patch_size,
                col:col+self.patch_size
            ]
            if patch.shape[:2] == (self.patch_size, self.patch_size):
                # Add spatial component to spikes
                pos_node = (patch_idx * 17) % self.n_nodes
                patch_spikes = spikes.copy()
                patch_spikes[pos_node] = 1.0
                self.decoder.learn_patch(patch_spikes, patch)

    def save(self, image: np.ndarray, path: str) -> None:
        """
        Save image to PNG (pure Python, no PIL needed).

        Uses built-in zlib for PNG compression.
        """
        import zlib, struct

        if self.n_channels == 1:
            image = image[:, :, 0]
            color_type = 0   # grayscale
        else:
            color_type = 2   # RGB

        def png_chunk(chunk_type, data):
            c    = chunk_type + data
            crc  = zlib.crc32(c) & 0xffffffff
            return struct.pack('>I', len(data)) + c + struct.pack('>I', crc)

        h, w = image.shape[:2]

        # Filter each row (None filter = 0)
        raw_data = b''
        for row in range(h):
            raw_data += b'\x00'
            if color_type == 0:
                raw_data += image[row].tobytes()
            else:
                raw_data += image[row].tobytes()

        compressed = zlib.compress(raw_data, 9)

        # PNG signature
        signature = b'\x89PNG\r\n\x1a\n'

        # IHDR chunk
        ihdr_data = struct.pack('>IIBBBBB', w, h, 8,
                                color_type, 0, 0, 0)
        ihdr = png_chunk(b'IHDR', ihdr_data)

        # IDAT chunk
        idat = png_chunk(b'IDAT', compressed)

        # IEND chunk
        iend = png_chunk(b'IEND', b'')

        with open(path, 'wb') as f:
            f.write(signature + ihdr + idat + iend)

        size_kb = len(signature + ihdr + idat + iend) / 1024
        print(f"✅ Saved: {path} "
              f"({w}×{h} {self.n_channels}ch, {size_kb:.1f} KB)")

    def stats(self) -> dict:
        return {
            'image_size' : self.image_size,
            'patch_size' : self.patch_size,
            'n_patches'  : self.layout.n_patches,
            'n_channels' : self.n_channels,
            'n_nodes'    : self.n_nodes,
            'n_generated': self.n_generated,
            'n_trained'  : self.decoder.n_updates,
        }

    def __repr__(self) -> str:
        return (f"ZINOHKImage("
                f"{self.image_size}×{self.image_size}, "
                f"patches={self.layout.n_patches}, "
                f"nodes={self.n_nodes})")
