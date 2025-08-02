#!/usr/bin/env python3
"""
Example usage of the AI Image Generator.

This script demonstrates various ways to use the image generator
with different prompts, parameters, and models.
"""

from image_generator import ImageGenerator
import os


def example_basic_usage():
    """Basic usage example."""
    print("🎨 Example 1: Basic Usage")
    print("=" * 40)
    
    generator = ImageGenerator()
    
    # Generate a simple image
    image, seed = generator.generate_image(
        prompt="A beautiful sunset over mountains",
        negative_prompt="blurry, low quality"
    )
    
    # Save the image
    generator.save_image(image, "basic_sunset")
    print(f"✅ Generated image with seed: {seed}")


def example_advanced_parameters():
    """Example with advanced parameters."""
    print("\n🎨 Example 2: Advanced Parameters")
    print("=" * 40)
    
    generator = ImageGenerator()
    
    # Generate with custom parameters
    image, seed = generator.generate_image(
        prompt="A futuristic city with flying cars and neon lights, cyberpunk style",
        negative_prompt="blurry, low quality, distorted, ugly, dark",
        num_inference_steps=50,  # More steps for better quality
        guidance_scale=8.5,      # Higher guidance for more adherence to prompt
        width=768,               # Larger image
        height=512,              # Landscape aspect ratio
        seed=42                  # Fixed seed for reproducible results
    )
    
    generator.save_image(image, "cyberpunk_city")
    print(f"✅ Generated cyberpunk city with seed: {seed}")


def example_different_models():
    """Example using different models."""
    print("\n🎨 Example 3: Different Models")
    print("=" * 40)
    
    # Try different models
    models = [
        "runwayml/stable-diffusion-v1-5",
        "prompthero/openjourney",  # Midjourney-style model
    ]
    
    prompt = "A magical forest with glowing mushrooms and fairy lights, fantasy art"
    
    for i, model_id in enumerate(models, 1):
        print(f"\n🔄 Loading model: {model_id}")
        generator = ImageGenerator(model_id=model_id)
        
        try:
            image, seed = generator.generate_image(
                prompt=prompt,
                negative_prompt="blurry, low quality, distorted",
                num_inference_steps=30,
                guidance_scale=7.5
            )
            
            filename = f"magical_forest_model_{i}"
            generator.save_image(image, filename)
            print(f"✅ Generated with {model_id} (seed: {seed})")
            
        except Exception as e:
            print(f"❌ Error with model {model_id}: {e}")


def example_batch_generation():
    """Example of generating multiple images with variations."""
    print("\n🎨 Example 4: Batch Generation")
    print("=" * 40)
    
    generator = ImageGenerator()
    
    # Different prompts for variety
    prompts = [
        "A cute robot playing with a cat in a garden, watercolor style",
        "A steampunk airship flying through clouds, detailed illustration",
        "A serene lake reflecting mountains at sunset, oil painting style",
        "A space station orbiting Earth, sci-fi art"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n🔄 Generating image {i}/{len(prompts)}: {prompt[:50]}...")
        
        image, seed = generator.generate_image(
            prompt=prompt,
            negative_prompt="blurry, low quality, distorted, ugly",
            num_inference_steps=30,
            guidance_scale=7.5
        )
        
        filename = f"batch_image_{i}"
        generator.save_image(image, filename)
        print(f"✅ Saved as {filename} (seed: {seed})")


def example_style_variations():
    """Example showing different artistic styles."""
    print("\n🎨 Example 5: Style Variations")
    print("=" * 40)
    
    generator = ImageGenerator()
    
    base_prompt = "A majestic dragon"
    styles = [
        "oil painting style",
        "digital art style",
        "watercolor style",
        "anime style",
        "photorealistic style"
    ]
    
    for i, style in enumerate(styles, 1):
        full_prompt = f"{base_prompt}, {style}"
        print(f"\n🔄 Generating: {full_prompt}")
        
        image, seed = generator.generate_image(
            prompt=full_prompt,
            negative_prompt="blurry, low quality, distorted",
            num_inference_steps=30,
            guidance_scale=7.5
        )
        
        filename = f"dragon_{style.replace(' ', '_').replace('style', '')}"
        generator.save_image(image, filename)
        print(f"✅ Saved as {filename} (seed: {seed})")


def example_parameter_experimentation():
    """Example showing how different parameters affect the output."""
    print("\n🎨 Example 6: Parameter Experimentation")
    print("=" * 40)
    
    generator = ImageGenerator()
    
    base_prompt = "A peaceful cottage in a meadow"
    
    # Test different guidance scales
    guidance_scales = [5.0, 7.5, 10.0, 12.0]
    
    for guidance in guidance_scales:
        print(f"\n🔄 Testing guidance scale: {guidance}")
        
        image, seed = generator.generate_image(
            prompt=base_prompt,
            negative_prompt="blurry, low quality, distorted",
            num_inference_steps=30,
            guidance_scale=guidance
        )
        
        filename = f"cottage_guidance_{int(guidance)}"
        generator.save_image(image, filename)
        print(f"✅ Saved as {filename} (seed: {seed})")


def main():
    """Run all examples."""
    print("🚀 AI Image Generator - Example Usage")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("generated_images", exist_ok=True)
    
    try:
        # Run examples
        example_basic_usage()
        example_advanced_parameters()
        example_different_models()
        example_batch_generation()
        example_style_variations()
        example_parameter_experimentation()
        
        print("\n🎉 All examples completed successfully!")
        print("📁 Check the 'generated_images' folder for your images.")
        
    except KeyboardInterrupt:
        print("\n❌ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")


if __name__ == "__main__":
    main() 