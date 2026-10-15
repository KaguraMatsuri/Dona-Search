from setuptools import find_packages, setup

setup(
    name="dona-search",
    version="0.1.0",
    packages=find_packages(),
    package_data={"dona_search": ["resources/*.yaml"]},
    install_requires=[
        "transformers>=5.19.0,<5.20",
        "mlx>=0.32.3,<0.33; platform_system == 'Darwin' and platform_machine == 'arm64'",
        "mlx-lm>=0.32.0,<0.33; platform_system == 'Darwin' and platform_machine == 'arm64'",
        "httpx>=0.27.0",
        "selenium>=4.26,<5",
        "lxml>=6.1.3,<7",
        "trafilatura>=1.9.0",
        "pydantic>=2.6.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "numpy>=1.24.0",
        "rich>=15,<16",
        "questionary>=2.1,<3",
        "huggingface-hub>=2.2,<3",
        "safetensors>=0.7,<1",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0", "black>=24.0.0", "ruff>=0.3.0"],
        "training": ["torch>=2.14.1,<2.15", "datasets>=5.1.0,<6", "accelerate>=0.28.0", "peft>=0.10.0", "scikit-learn>=1.4.0"],
        "eval": ["scikit-learn>=1.4.0"],
    },
    entry_points={"console_scripts": ["dona-search=dona_search.__main__:main"]},
    python_requires=">=3.11",
)
