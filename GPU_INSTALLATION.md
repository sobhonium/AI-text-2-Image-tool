# 🚀 GPU Installation Guide

This guide will help you install the AI Image Generator with GPU acceleration for maximum performance.

## 🎯 Performance Benefits

| Hardware | Time per Image (512x512) | Quality |
|----------|-------------------------|---------|
| CPU | 30-60 seconds | Good |
| Apple Silicon (M1/M2/M3) | 10-20 seconds | Good |
| NVIDIA GPU (RTX 3060+) | 5-15 seconds | Excellent |
| High-end GPU (RTX 4090) | 2-8 seconds | Excellent |

## 🖥️ Platform-Specific Installation

### 🍎 macOS (Apple Silicon M1/M2/M3)

**Automatic GPU acceleration with Metal Performance Shaders (MPS):**

```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt

# Test GPU support
python cli_interface.py --gpu-info
```

**Expected output:**
```
🚀 Using Apple Silicon GPU (MPS)
💾 Using Metal Performance Shaders
```

### 🖥️ macOS (Intel)

**CPU only - no GPU acceleration available:**
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### 🐧 Linux / Windows (NVIDIA GPU)

**1. Install NVIDIA Drivers:**
- **Linux**: `sudo apt install nvidia-driver-525` (or latest)
- **Windows**: Download from [NVIDIA website](https://www.nvidia.com/drivers/)

**2. Install CUDA Toolkit:**
- **Linux**: `sudo apt install nvidia-cuda-toolkit`
- **Windows**: Download from [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)

**3. Install PyTorch with CUDA:**

For **CUDA 11.8** (most stable):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For **CUDA 12.1** (newer):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**4. Install other dependencies:**
```bash
pip install -r requirements.txt
```

**5. Test GPU support:**
```bash
python cli_interface.py --gpu-info
```

**Expected output:**
```
🚀 GPU: NVIDIA GeForce RTX 4090
💾 Total VRAM: 24.0GB
💾 Free VRAM: 23.5GB
📊 CUDA Version: 11.8
```

## 🔧 GPU Memory Optimization

### For 4-6GB VRAM (GTX 1060, RTX 2060):
```python
# Use smaller images and fewer steps
generator = ImageGenerator()
image, seed = generator.generate_image(
    prompt="Your prompt",
    width=512,
    height=512,
    num_inference_steps=20,  # Fewer steps
    guidance_scale=7.5
)
```

### For 8-12GB VRAM (RTX 3070, RTX 3080):
```python
# Standard settings work well
generator = ImageGenerator()
image, seed = generator.generate_image(
    prompt="Your prompt",
    width=768,
    height=768,
    num_inference_steps=30,
    guidance_scale=7.5
)
```

### For 16GB+ VRAM (RTX 4080, RTX 4090):
```python
# Can use larger images and more steps
generator = ImageGenerator()
image, seed = generator.generate_image(
    prompt="Your prompt",
    width=1024,
    height=1024,
    num_inference_steps=50,
    guidance_scale=7.5
)
```

## 🚀 Performance Tips

### 1. **Enable Memory Optimizations** (Automatic)
The code automatically enables:
- Attention slicing
- VAE slicing
- Model CPU offload
- Half-precision (FP16)

### 2. **Use Appropriate Image Sizes**
```bash
# Fast generation
python cli_interface.py "Your prompt" --width 512 --height 512

# High quality
python cli_interface.py "Your prompt" --width 768 --height 768

# Maximum quality (requires 16GB+ VRAM)
python cli_interface.py "Your prompt" --width 1024 --height 1024
```

### 3. **Optimize Steps vs Quality**
```bash
# Fast (20 steps)
python cli_interface.py "Your prompt" --steps 20

# Balanced (30 steps)
python cli_interface.py "Your prompt" --steps 30

# High quality (50+ steps)
python cli_interface.py "Your prompt" --steps 50
```

## 🐛 Troubleshooting

### CUDA Out of Memory Error
```bash
# Reduce image size
python cli_interface.py "Your prompt" --width 512 --height 512

# Reduce steps
python cli_interface.py "Your prompt" --steps 20

# Force CPU mode
python cli_interface.py "Your prompt" --cpu-only
```

### CUDA Not Available
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Slow Generation
```bash
# Check GPU usage
nvidia-smi

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Use smaller images
python cli_interface.py "Your prompt" --width 512 --height 512
```

## 📊 Monitoring GPU Usage

### Command Line:
```bash
# Show GPU info
python cli_interface.py --gpu-info

# Monitor during generation
watch -n 1 nvidia-smi
```

### Web Interface:
The web interface automatically shows:
- Device being used
- Generation time
- Memory usage (for CUDA)

## 🎯 Quick Test

After installation, test your setup:

```bash
# Test GPU detection
python cli_interface.py --gpu-info

# Generate a test image
python cli_interface.py "A beautiful sunset over mountains" --steps 20

# Check generation time
python cli_interface.py "A cute robot" --steps 30 --width 768 --height 768
```

## 🔗 Useful Links

- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
- [NVIDIA Drivers](https://www.nvidia.com/drivers/)
- [Stable Diffusion Models](https://huggingface.co/models?search=stable-diffusion)

---

**Happy GPU-accelerated image generation! 🚀✨** 