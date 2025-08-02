#!/usr/bin/env python3
"""
Setup script for the AI Image Generator project.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-image-generator",
    version="1.0.0",
    author="AI Image Generator",
    author_email="your.email@example.com",
    description="A Python project that generates images from text prompts using Stable Diffusion",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/text-to-image-project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Artistic Software",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-image-generator=cli_interface:main",
            "ai-image-web=web_interface:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="ai, image-generation, stable-diffusion, text-to-image, machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/text-to-image-project/issues",
        "Source": "https://github.com/yourusername/text-to-image-project",
        "Documentation": "https://github.com/yourusername/text-to-image-project#readme",
    },
) 