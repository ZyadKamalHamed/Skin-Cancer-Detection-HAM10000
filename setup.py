"""
Setup script for Skin Cancer Detection System
Professional package installation and dependency management
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="skin-cancer-detection",
    version="1.0.0",
    author="Zyad Kamal Hamed",
    author_email="zyad2408@live.com.au",
    description="Advanced deep learning system for automated skin cancer detection using computer vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ZyadKamalHamed/Skin-Cancer-Detection-HAM10000",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "deployment": [
            "streamlit>=1.25.0",
            "gradio>=3.40.0",
            "gunicorn>=21.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "skin-cancer-train=scripts.train:main",
            "skin-cancer-evaluate=scripts.evaluate:main",
            "skin-cancer-predict=scripts.predict:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/ZyadKamalHamed/Skin-Cancer-Detection-HAM10000/issues",
        "Source": "https://github.com/ZyadKamalHamed/Skin-Cancer-Detection-HAM10000",
        "Documentation": "https://github.com/ZyadKamalHamed/Skin-Cancer-Detection-HAM10000/blob/main/docs/",
    },
)