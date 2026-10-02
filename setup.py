from setuptools import find_packages, setup


def get_requirements(file_path):
    with open(file_path) as file_obj:
        return [line.strip() for line in file_obj if line.strip()]


setup(
    name="student-performance-prediction",
    version="0.0.1",
    author="Ahmad Bahir Salamzai",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
