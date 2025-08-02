import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import time
from typing import Optional, Tuple


def get_gpu_info():
    """Get detailed GPU information."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        "device_count": 0,
        "device_name": "CPU",
        "memory_total": 0,
        "memory_free": 0
    }
    
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)
        info["memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info["memory_free"] = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info["device_name"] = "Apple Silicon GPU (MPS)"
    
    return info


def print_gpu_info():
    """Print detailed GPU information."""
    info = get_gpu_info()
    
    print("🖥️  Hardware Information:")
    print("=" * 40)
    
    if info["cuda_available"]:
        print(f"🚀 GPU: {info['device_name']}")
        print(f"💾 Total VRAM: {info['memory_total']:.1f}GB")
        print(f"💾 Free VRAM: {info['memory_free']:.1f}GB")
        print(f"📊 CUDA Version: {torch.version.cuda}")
    elif info["mps_available"]:
        print(f"🚀 GPU: {info['device_name']}")
        print("💾 Using Metal Performance Shaders")
    else:
        print("💻 CPU: No GPU acceleration available")
        print("💡 Consider using GPU for faster generation")
    
    print(f"🐍 PyTorch Version: {torch.__version__}")
    print("=" * 40)


class ImageGenerator:
    """
    A class to generate images from text prompts using Stable Diffusion.
    """
    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", use_gpu: bool = True):
        """
        Initialize the image generator with a Stable Diffusion model.
        
        Args:
            model_id (str): The model ID to use for image generation
            use_gpu (bool): Whether to use GPU if available (default: True)
        """
        self.model_id = model_id
        self.pipeline = None
        self.use_gpu = use_gpu
        
        # Determine the best available device
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16  # Use half precision for GPU memory efficiency
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        elif use_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"  # Apple Silicon GPU
            self.dtype = torch.float32
            print("🚀 Using Apple Silicon GPU (MPS)")
        else:
            self.device = "cpu"
            self.dtype = torch.float32
            if use_gpu:
                print("⚠️  GPU requested but not available, falling back to CPU")
            else:
                print("💻 Using CPU")
        
        print(f"📊 Device: {self.device}, Precision: {self.dtype}")
        
    def load_model(self):
        """Load the Stable Diffusion pipeline with GPU optimizations."""
        try:
            print(f"🔄 Loading model: {self.model_id}")
            
            # Load with appropriate settings for the device
            if self.device == "cuda":
                # GPU optimizations
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                    variant="fp16" if self.dtype == torch.float16 else None
                )
                # Enable memory efficient attention if available
                if hasattr(self.pipeline, 'enable_attention_slicing'):
                    self.pipeline.enable_attention_slicing()
                if hasattr(self.pipeline, 'enable_vae_slicing'):
                    self.pipeline.enable_vae_slicing()
                if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                    self.pipeline.enable_model_cpu_offload()
                    
            elif self.device == "mps":
                # Apple Silicon optimizations
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                
            else:
                # CPU optimizations
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Print memory usage for GPU
            if self.device == "cuda":
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"💾 GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
            print(f"✅ Model {self.model_id} loaded successfully on {self.device}!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None
    ) -> Tuple[Image.Image, int]:
        """
        Generate an image from a text prompt with GPU optimizations.
        
        Args:
            prompt (str): The text prompt describing the image to generate
            negative_prompt (str, optional): What to avoid in the image
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): How closely to follow the prompt
            width (int): Image width
            height (int): Image height
            seed (int, optional): Random seed for reproducible results
            
        Returns:
            Tuple[Image.Image, int]: Generated image and the seed used
        """
        if self.pipeline is None:
            self.load_model()
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
            seed = torch.randint(0, 2**32, (1,)).item()
        
        # Clear GPU cache before generation if using CUDA
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        print(f"🎨 Generating image: {prompt[:50]}...")
        print(f"⚙️  Steps: {num_inference_steps}, Guidance: {guidance_scale}, Size: {width}x{height}")
        
        # Generate the image
        with torch.no_grad():  # Disable gradient computation for inference
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator
            )
        
        # Print GPU memory usage after generation
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1e9
            print(f"💾 GPU Memory after generation: {allocated:.2f}GB")
        
        return result.images[0], seed
    
    def save_image(self, image: Image.Image, filename: str, output_dir: str = "generated_images"):
        """
        Save the generated image to a file.
        
        Args:
            image (Image.Image): The image to save
            filename (str): The filename to save as
            output_dir (str): Directory to save the image in
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Ensure filename has .png extension
        if not filename.endswith('.png'):
            filename += '.png'
        
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)
        print(f"Image saved to: {filepath}")
        return filepath


def main():
    """Example usage of the ImageGenerator class."""
    # Initialize the generator
    generator = ImageGenerator()
    
    # Example prompts
    prompts = [
        "A beautiful sunset over mountains, digital art style",
        "A cute robot playing with a cat in a garden",
        "A futuristic city with flying cars and neon lights"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\nGenerating image {i+1}: {prompt}")
        
        # Generate image
        image, seed = generator.generate_image(
            prompt=prompt,
            negative_prompt="blurry, low quality, distorted",
            num_inference_steps=30,
            guidance_scale=7.5
        )
        
        # Save image
        filename = f"generated_image_{i+1}"
        generator.save_image(image, filename)
        print(f"Generated with seed: {seed}")


if __name__ == "__main__":
    main() 