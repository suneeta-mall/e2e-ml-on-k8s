import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    deps_list = fh.read().split()

setuptools.setup(
    name="pylib",
    version="1.0.0",
    author="Suneeta Mall",
    author_email="suneeta.mall@nearmap.com",
    description="Core function lib for e2e ml example",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/suneeta-mall/e2e-ml-on-k8s",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=deps_list,
    zip_safe=False,
    package_data={'': ['templates/katib-hp-tunning.yaml', 'templates/model-serving.yaml',
                       'templates/tf-training.yaml']},
    include_package_data=True
)
