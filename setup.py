from pathlib import Path

from setuptools import find_packages, setup

README = Path(__file__).parent / "README.md"

setup(
    name="obfusai",
    version="0.9.0",
    description="Obfuscate LLM prompts, roles and tool schemas while keeping them executable by LLMs.",
    long_description=README.read_text(encoding="utf-8") if README.exists() else "",
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests", "tests.*", "examples")),
    package_data={"obfusai": ["py.typed"]},
    url="https://github.com/GANGMU-SBOM/ObfusAI",
    project_urls={
        "Source": "https://github.com/GANGMU-SBOM/ObfusAI",
        "Issues": "https://github.com/GANGMU-SBOM/ObfusAI/issues",
    },
    keywords=["llm", "prompt", "obfuscation", "prompt-injection", "security", "openai", "anthropic", "langchain", "llamaindex"],
    license="MIT",
    python_requires=">=3.9",
    install_requires=[],
    extras_require={
        "langchain": ["langchain-core>=0.2.0"],
        "llamaindex": ["llama-index-core>=0.12.0"],
        "dev": ["pytest>=7.0", "langchain-core>=0.2.0", "llama-index-core>=0.12.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Security",
        "Typing :: Typed",
    ],
    entry_points={"console_scripts": ["obfusai=obfusai.cli:main"]},
    zip_safe=False,
)
