# 🚀 Quick Start Guide

Get your AI Image Generator up and running in minutes!

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Internet connection (for downloading models)

**Optional but recommended:**
- NVIDIA GPU with CUDA support (for faster generation)
- At least 8GB RAM (16GB+ recommended)

## ⚡ Quick Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test installation:**
   ```bash
   python test_installation.py
   ```

3. **Start generating images!**

## 🎯 Three Ways to Use

### 1. 🌐 Web Interface (Easiest)
```bash
python web_interface.py
```
Then open your browser to `http://localhost:7860`

### 2. 💻 Command Line
```bash
# Basic usage
python cli_interface.py "A beautiful sunset over mountains"

# With custom settings
python cli_interface.py "A cute robot" --steps 50 --guidance 8.0
```

### 3. 📚 Python Code
```python
from image_generator import ImageGenerator

generator = ImageGenerator()
image, seed = generator.generate_image("A magical forest")
generator.save_image(image, "my_image")
```

## 🎨 Your First Image

Try this simple command to generate your first image:

```bash
python cli_interface.py "A serene lake reflecting mountains at sunset, oil painting style"
```

The image will be saved in the `generated_images/` folder.

## 🔧 Troubleshooting

**If you get dependency errors:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**If generation is slow:**
- Use GPU if available
- Reduce image size (try 512x512)
- Reduce inference steps (try 20-30)

**If you get memory errors:**
- Close other applications
- Use smaller image size
- Use CPU mode: `export CUDA_VISIBLE_DEVICES=""`

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Try the [example_usage.py](example_usage.py) for more examples
- Experiment with different prompts and parameters
- Check out different models with `python cli_interface.py --list-models`

## 🆘 Need Help?

- Check the console output for error messages
- Ensure all dependencies are installed: `python test_installation.py`
- Try the web interface for easier debugging
- Use default parameters first, then experiment

---

**Happy Image Generating! 🎨✨** 