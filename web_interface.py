import gradio as gr
from image_generator import ImageGenerator, print_gpu_info
import os
import time
from datetime import datetime


class WebInterface:
    """
    A web interface for the image generator using Gradio.
    """
    
    def __init__(self):
        """Initialize the web interface."""
        print("🚀 Initializing AI Image Generator Web Interface...")
        print_gpu_info()
        
        self.generator = ImageGenerator(use_gpu=True)
        self.generator.load_model()
    
    def generate_image_web(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted, ugly",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: int = -1
    ):
        """
        Generate an image for the web interface.
        
        Args:
            prompt (str): The text prompt
            negative_prompt (str): What to avoid in the image
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): How closely to follow the prompt
            width (int): Image width
            height (int): Image height
            seed (int): Random seed (-1 for random)
            
        Returns:
            tuple: (image, seed_used, info_text)
        """
        try:
            # Use random seed if -1 is provided
            actual_seed = None if seed == -1 else seed
            
            # Generate the image with timing
            start_time = time.time()
            image, used_seed = self.generator.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                seed=actual_seed
            )
            gen_time = time.time() - start_time
            
            # Create info text
            info_text = f"""
            **Generation Details:**
            - **Prompt:** {prompt}
            - **Negative Prompt:** {negative_prompt}
            - **Steps:** {num_inference_steps}
            - **Guidance Scale:** {guidance_scale}
            - **Size:** {width}x{height}
            - **Seed:** {used_seed}
            - **Generation Time:** {gen_time:.2f} seconds
            - **Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            return image, used_seed, info_text
            
        except Exception as e:
            error_text = f"Error generating image: {str(e)}"
            return None, -1, error_text
    
    def create_interface(self):
        """Create and return the Gradio interface."""
        
        # Define the interface
        with gr.Blocks(title="Text-to-Image Generator", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🎨 AI Image Generator")
            gr.Markdown("Generate beautiful images from text descriptions using Stable Diffusion!")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Input controls
                    gr.Markdown("## 📝 Input Settings")
                    
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe the image you want to generate...",
                        lines=3,
                        value="A beautiful sunset over mountains, digital art style"
                    )
                    
                    negative_prompt_input = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="What to avoid in the image...",
                        lines=2,
                        value="blurry, low quality, distorted, ugly"
                    )
                    
                    with gr.Row():
                        steps_input = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=30,
                            step=1,
                            label="Inference Steps"
                        )
                        
                        guidance_input = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            value=7.5,
                            step=0.1,
                            label="Guidance Scale"
                        )
                    
                    with gr.Row():
                        width_input = gr.Slider(
                            minimum=256,
                            maximum=1024,
                            value=512,
                            step=64,
                            label="Width"
                        )
                        
                        height_input = gr.Slider(
                            minimum=256,
                            maximum=1024,
                            value=512,
                            step=64,
                            label="Height"
                        )
                    
                    seed_input = gr.Number(
                        label="Seed (-1 for random)",
                        value=-1,
                        precision=0
                    )
                    
                    generate_btn = gr.Button(
                        "🎨 Generate Image",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=1):
                    # Output display
                    gr.Markdown("## 🖼️ Generated Image")
                    
                    image_output = gr.Image(
                        label="Generated Image",
                        type="pil"
                    )
                    
                    seed_output = gr.Number(
                        label="Used Seed",
                        interactive=False
                    )
                    
                    info_output = gr.Markdown(
                        label="Generation Info"
                    )
            
            # Example prompts
            gr.Markdown("## 💡 Example Prompts")
            example_prompts = [
                "A cute robot playing with a cat in a garden, watercolor style",
                "A futuristic city with flying cars and neon lights, cyberpunk style",
                "A magical forest with glowing mushrooms and fairy lights, fantasy art",
                "A serene lake reflecting mountains at sunset, oil painting style",
                "A steampunk airship flying through clouds, detailed illustration"
            ]
            
            with gr.Row():
                for prompt in example_prompts:
                    gr.Button(
                        prompt,
                        size="sm"
                    ).click(
                        lambda p=prompt: p,
                        outputs=prompt_input
                    )
            
            # Connect the generate button
            generate_btn.click(
                fn=self.generate_image_web,
                inputs=[
                    prompt_input,
                    negative_prompt_input,
                    steps_input,
                    guidance_input,
                    width_input,
                    height_input,
                    seed_input
                ],
                outputs=[
                    image_output,
                    seed_output,
                    info_output
                ]
            )
            
            # Allow Enter key to generate
            prompt_input.submit(
                fn=self.generate_image_web,
                inputs=[
                    prompt_input,
                    negative_prompt_input,
                    steps_input,
                    guidance_input,
                    width_input,
                    height_input,
                    seed_input
                ],
                outputs=[
                    image_output,
                    seed_output,
                    info_output
                ]
            )
        
        return interface


def main():
    """Launch the web interface."""
    print("🚀 Starting AI Image Generator Web Interface...")
    print("📝 Loading model (this may take a few moments)...")
    
    # Create and launch the interface
    interface = WebInterface()
    app = interface.create_interface()
    
    print("✅ Interface ready! Opening in your browser...")
    app.launch(
        share=False,  # Set to True to create a public link
        server_name="0.0.0.0",
        server_port=7860
    )


if __name__ == "__main__":
    main() 