#!/usr/bin/env python3
"""
Test script to verify that all dependencies are installed correctly.
"""

import sys
import importlib


def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name}: {e}")
        return False


def main():
    """Test all required dependencies."""
    print("🧪 Testing AI Image Generator Dependencies")
    print("=" * 50)
    
    # Required packages
    packages = [
        ("torch", "PyTorch"),
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
        ("accelerate", "Accelerate"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("gradio", "Gradio"),
    ]
    
    all_good = True
    
    for module, name in packages:
        if not test_import(module, name):
            all_good = False
    
    print("\n" + "=" * 50)
    
    if all_good:
        print("🎉 All dependencies are installed correctly!")
        print("\n🚀 You can now use the AI Image Generator:")
        print("   • Web Interface: python web_interface.py")
        print("   • CLI Interface: python cli_interface.py")
        print("   • Examples: python example_usage.py")
    else:
        print("❌ Some dependencies are missing.")
        print("📦 Please install them with: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main() 