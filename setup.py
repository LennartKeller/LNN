from setuptools import find_packages, setup

setup(
    name="LNN",
    description="_L_ennart's _N_eural _N_etwork modeling boilerplate code...",
    author="Lennart Keller",
    author_email="lennartkeller@gmail.com",
    version="0.1.0",
    python_requires=">3.10",
    package_dir={"": "src"},
    # packages=find_packages("src")
    include_package_data=True,
    install_requires=[
        "trident-core @ git+https://github.com/fdschmidt93/trident.git",
        "hydra-core >= 1.3.0",
        "lightning",
        "torch",
        "torchaudio",
        "dataclasses-json",
        "numpy",
        "iso-639",
        "click",
    ],
    entry_points={
        "console_scripts": [
            "sguardian = lnn.slurm:sguardian",
        ],
    },
)
