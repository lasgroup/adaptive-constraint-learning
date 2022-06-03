import setuptools

setuptools.setup(
    name="constrained_bai",
    version="0.1dev",
    description="Constraint Best-Arm Identification and Adaptive Constraint Learning",
    long_description=open("README.md").read(),
    url="https://github.com/lasgroup/adaptive-constraint-learning/",
    author="David Lindner",
    author_email="dev@davidlindner.me",
    # We freeze library versions for reproducibility
    # Everything likely also works with more recent versions
    install_requires=[
        "numpy==1.21.2",
        "matplotlib==3.4.3",
        "scipy==1.7.1",
        "sacred==0.8.2",
        "frozendict==2.0.7",
    ],
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    zip_safe=True,
    entry_points={},
)
