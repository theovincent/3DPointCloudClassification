from setuptools import setup

setup(
    name="3DPointCloudClassification",
    version="0.1",
    description="Classifies 3D points.",
    packages=["Classifier3D"],
    requires=["setuptools", "wheel"],
    install_requires=["numpy", "scikit-learn", "tqdm"],
    extras_require={"dev": ["black"]},
)
