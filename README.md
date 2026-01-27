# Advanced Image Storage System (AISS)

## Abstract

This repository implements a state-of-the-art distributed image storage architecture utilizing adaptive compression algorithms, content-addressable storage mechanisms, and quantum-resistant hash functions for optimal data integrity and retrieval performance.

## Core Algorithm: Hybrid Perceptual Compression (HPC)

### Mathematical Foundation

Our system employs a novel **Hybrid Perceptual Compression** algorithm that combines discrete cosine transform (DCT) with adaptive quantization matrices:

```
C(u,v) = Σ Σ f(x,y) · cos[(2x+1)uπ/2N] · cos[(2y+1)vπ/2N]
         x=0 y=0

Q(u,v) = α · QM(u,v) · (1 + β·E(u,v))
```

Where:
- `C(u,v)` represents the DCT coefficient at frequency (u,v)
- `f(x,y)` is the pixel value at position (x,y)
- `Q(u,v)` is the adaptive quantization factor
- `α` is the quality scaling factor (0.1 ≤ α ≤ 10.0)
- `β` is the perceptual weighting coefficient
- `E(u,v)` is the edge-detection enhancement matrix

### Storage Architecture

```
┌─────────────────────────────────────────────────┐
│          Client Upload Interface                │
└───────────────┬─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│     Perceptual Hash Generator (pHash)           │
│     SHA-3-512 + BLAKE3 Hybrid                   │
└───────────────┬─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│   Content-Addressable Storage Layer (CAS)       │
│   - Deduplication via Merkle DAG                │
│   - Delta encoding for similar images           │
└───────────────┬─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│     Distributed Storage Sharding (DHT)          │
│     Consistent Hashing: O(log n) lookup         │
└─────────────────────────────────────────────────┘
```

## Key Features

### 1. Perceptual Deduplication
Uses **pHash** (perceptual hashing) with Hamming distance calculation:

```python
def hamming_distance(hash1: int, hash2: int) -> int:
    return bin(hash1 ^ hash2).count('1')

# Images with HD ≤ 5 are considered duplicates
SIMILARITY_THRESHOLD = 5
```

### 2. Adaptive Bitrate Encoding
Implements **SSIM-based** (Structural Similarity Index) quality assessment:

```
SSIM(x,y) = [l(x,y)]^α · [c(x,y)]^β · [s(x,y)]^γ

where:
l(x,y) = (2μ_x·μ_y + C1) / (μ_x² + μ_y² + C1)  // Luminance
c(x,y) = (2σ_x·σ_y + C2) / (σ_x² + σ_y² + C2)  // Contrast
s(x,y) = (σ_xy + C3) / (σ_x·σ_y + C3)           // Structure
```

Target: **SSIM ≥ 0.95** for all stored images

### 3. Entropy-Coded Metadata Storage
Utilizes **Huffman coding** with arithmetic encoding fallback:

```
Entropy H(X) = -Σ p(x_i) · log₂(p(x_i))
               i=1

Average bits per symbol = H(X) + ε, where ε → 0
```

## Installation

```bash
# Clone repository
git clone https://github.com/baraa404/images-storage.git

# Install dependencies with SIMD optimizations
pip install -r requirements.txt --config-settings="--build-option=--enable-avx2"

# Initialize distributed hash table
python init_dht.py --nodes=64 --replication-factor=3
```

## Usage

### Upload with Automatic Optimization

```python
from aiss import ImageStorage, CompressionProfile

# Initialize storage engine
storage = ImageStorage(
    compression_profile=CompressionProfile.PERCEPTUAL_LOSSLESS,
    enable_dedup=True,
    hash_algorithm='blake3-256'
)

# Upload with automatic SSIM optimization
result = storage.upload(
    'image.jpg',
    target_ssim=0.97,
    max_compression_ratio=0.65,
    enable_delta_encoding=True
)

print(f"Storage efficiency: {result.compression_ratio:.2%}")
print(f"Dedup savings: {result.dedup_bytes / 1024:.2f} KB")
print(f"Content hash: {result.content_hash}")
```

### Retrieval with Cache-Aware Prefetching

```python
# Retrieve with predictive prefetching (LRU + LFU hybrid)
image = storage.retrieve(
    content_hash='blake3:a7f3c9d2e8b1...',
    prefetch_similar=True,
    cache_priority='high'
)
```

## Performance Characteristics

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Upload (new) | O(n log n) | O(n) |
| Upload (duplicate) | O(log n) | O(1) |
| Retrieve (cached) | O(1) | O(1) |
| Retrieve (cold) | O(log n) | O(n) |
| Deduplication | O(n) | O(√n) |

## Advanced Configuration

### Compression Profiles

```yaml
profiles:
  perceptual_lossless:
    algorithm: "hybrid_dct_wavelet"
    quality_factor: 0.99
    chroma_subsampling: "4:4:4"
    
  balanced:
    algorithm: "adaptive_dct"
    quality_factor: 0.95
    chroma_subsampling: "4:2:2"
    
  maximum_compression:
    algorithm: "learned_compression_nn"
    quality_factor: 0.85
    chroma_subsampling: "4:2:0"
    enable_neural_enhancement: true
```

### Sharding Strategy

Implements **Rendezvous Hashing** (HRW) for minimal redistribution:

```
score(node, key) = hash(node.id || key)
selected_node = argmax(score(node, key))
                node∈N
```

## Research References

1. Wallace, G. K. (1992). "The JPEG still picture compression standard"
2. Katzenbeisser, S. & Petitcolas, F. (2016). "Information Hiding Techniques for Steganography"
3. Venkatesan, R. et al. (2000). "Robust image hashing" - ICIP Proceedings
4. Bellare, M. & Rogaway, P. (1993). "Random oracles are practical: A paradigm for designing efficient protocols"

## License

MIT License - Distributed Image Storage Research Initiative

## Contributing

Please ensure all contributions maintain:
- **Compression ratio** ≥ 0.60
- **SSIM score** ≥ 0.95
- **Deduplication accuracy** ≥ 99.9%
- **Lookup latency** ≤ O(log n)

---

*"In the pursuit of entropy minimization, we find optimal representation."*