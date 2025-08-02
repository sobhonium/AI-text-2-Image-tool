#!/usr/bin/env python3
"""
Command-line interface for the AI Image Generator.
"""

import argparse
import sys
import time
from image_generator import ImageGenerator, print_gpu_info
import os


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate images from text prompts using Stable Diffusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_interface.py "A beautiful sunset over mountains"
  python cli_interface.py "A cute robot" --steps 50 --guidance 8.0
  python cli_interface.py "Futuristic city" --width 768 --height 512 --seed 42
  python cli_interface.py "Magical forest" --output my_image.png
        """
    )
    
    parser.add_argument(
        "prompt",
        type=str,
        help="Text prompt describing the image to generate"
    )
    
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="blurry, low quality, distorted, ugly",
        help="Negative prompt (what to avoid in the image)"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of inference steps (10-100, default: 30)"
    )
    
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.5,
        help="Guidance scale (1.0-20.0, default: 7.5)"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width in pixels (256-1024, default: 512)"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height in pixels (256-1024, default: 512)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (without extension)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_images",
        help="Output directory (default: generated_images)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Model ID to use for generation"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    
    parser.add_argument(
        "--gpu-info",
        action="store_true",
        help="Show GPU information and exit"
    )
    
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU usage (disable GPU)"
    )
    
    return parser.parse_args()


def list_available_models():
    """List some popular Stable Diffusion models."""
    models = [
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-2-1",
        "CompVis/stable-diffusion-v1-4",
        "prompthero/openjourney",
        "dreamlike-art/dreamlike-photoreal-2.0",
        "stabilityai/stable-diffusion-xl-base-1.0"
    ]
    
    print("🎨 Available Models:")
    print("=" * 50)
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    print("\n💡 You can use any of these models with the --model argument.")
    print("   More models are available on Hugging Face Hub!")


def validate_arguments(args):
    """Validate command line arguments."""
    if args.steps < 10 or args.steps > 100:
        print("❌ Error: Steps must be between 10 and 100")
        sys.exit(1)
    
    if args.guidance < 1.0 or args.guidance > 20.0:
        print("❌ Error: Guidance scale must be between 1.0 and 20.0")
        sys.exit(1)
    
    if args.width < 256 or args.width > 1024:
        print("❌ Error: Width must be between 256 and 1024")
        sys.exit(1)
    
    if args.height < 256 or args.height > 1024:
        print("❌ Error: Height must be between 256 and 1024")
        sys.exit(1)
    
    # Ensure width and height are multiples of 64
    if args.width % 64 != 0:
        print(f"⚠️  Warning: Width {args.width} is not a multiple of 64. Using {args.width - (args.width % 64)}")
        args.width = args.width - (args.width % 64)
    
    if args.height % 64 != 0:
        print(f"⚠️  Warning: Height {args.height} is not a multiple of 64. Using {args.height - (args.height % 64)}")
        args.height = args.height - (args.height % 64)


def main():
    """Main function for the CLI interface."""
    try:
        args = parse_arguments()
    except SystemExit:
        # Show helpful usage examples when no arguments provided
        print("🎨 AI Image Generator - Command Line Interface")
        print("=" * 50)
        print("\n📝 Usage Examples:")
        print("  python cli_interface.py \"A beautiful sunset over mountains\"")
        print("  python cli_interface.py \"A cute robot playing with a cat\" --steps 50")
        print("  python cli_interface.py \"Futuristic city\" --width 768 --height 512")
        print("\n🔧 Utility Commands:")
        print("  python cli_interface.py --gpu-info")
        print("  python cli_interface.py --list-models")
        print("  python cli_interface.py --help")
        print("\n💡 For more help, run: python cli_interface.py --help")
        return
    
    if args.list_models:
        list_available_models()
        return
    
    if args.gpu_info:
        print_gpu_info()
        return
    
    # Validate arguments
    validate_arguments(args)
    
    print("🎨 AI Image Generator")
    print("=" * 40)
    
    # Show GPU info
    print_gpu_info()
    
    print(f"📝 Prompt: {args.prompt}")
    print(f"🚫 Negative: {args.negative_prompt}")
    print(f"⚙️  Steps: {args.steps}")
    print(f"🎯 Guidance: {args.guidance}")
    print(f"📏 Size: {args.width}x{args.height}")
    print(f"🎲 Seed: {args.seed if args.seed else 'Random'}")
    print(f"🤖 Model: {args.model}")
    print()
    
    try:
        # Initialize generator
        print("🔄 Loading model...")
        start_time = time.time()
        
        generator = ImageGenerator(
            model_id=args.model,
            use_gpu=not args.cpu_only
        )
        generator.load_model()
        
        load_time = time.time() - start_time
        print(f"⏱️  Model loaded in {load_time:.2f} seconds")
        
        # Generate image
        print("🎨 Generating image...")
        gen_start_time = time.time()
        
        image, used_seed = generator.generate_image(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            width=args.width,
            height=args.height,
            seed=args.seed
        )
        
        gen_time = time.time() - gen_start_time
        print(f"⏱️  Image generated in {gen_time:.2f} seconds")
        
        # Determine output filename
        if args.output:
            filename = args.output
        else:
            # Create filename from prompt
            safe_prompt = "".join(c for c in args.prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_prompt = safe_prompt.replace(' ', '_')[:30]  # Limit length
            filename = f"{safe_prompt}_{used_seed}"
        
        # Save image
        print("💾 Saving image...")
        filepath = generator.save_image(image, filename, args.output_dir)
        
        print("✅ Success!")
        print(f"🖼️  Image saved: {filepath}")
        print(f"🎲 Used seed: {used_seed}")
        
    except KeyboardInterrupt:
        print("\n❌ Generation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 