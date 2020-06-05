import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pl_face_recognition",
    version="0.1.0",
    author="Platanus",
    author_email="andres.cadiz@platan.us",
    description="A package for out of the box face recognition learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/platanus/face-recognition",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'torch>=1.0.0',
        'torchvision',
        'sklearn',
        'numpy',
        'facenet-pytorch',
        'opencv-python',
    ],
    python_requires='>=3.6',
)
