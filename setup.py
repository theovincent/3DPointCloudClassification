from setuptools import setup

setup(
    name="3DPointCloudClassification",
    version="0.1",
    description="Classifies 3D points.",
    packages=["classifier_3D", "range_net"],
    requires=["setuptools", "wheel"],
    install_requires=["numpy", "scikit-learn", "tqdm", "pandas", "matplotlib"],
    extras_require={"dev": ["black", "ipykernel"]},
    entry_points={
        "console_scripts": [
            "classify_features=classifier_3D.classify_features:classify_features_cli",
            "compute_features=classifier_3D.feature_extraction.compute_features:compute_features_cli",
            "filter_predictions=classifier_3D.filter_predictions:filter_predictions_cli",
            "create_dataset=range_net.create_dataset:create_dataset_cli",
            "merge_labels=range_net.merge_labels:merge_labels_cli",
        ],
    },
)
