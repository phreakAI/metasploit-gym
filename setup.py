# MIT License
#
# Copyright (c) 2023 PhreakAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

""" Setup of the metasploit-gym package.
"""

from typing import List
import setuptools

_VERSION = "0.1.0"

# Short description.
short_description = "A gymnasium environment for metasploit pentesting software."

# Packages needed for the environment to run.
# The compatible release operator (`~=`) is used to match any candidate version
# that is expected to be compatible with the specified version.
REQUIRED_PACKAGES = [
    "gymnasium ~= 0.27.1",
    "numpy ~= 1.21.0",
    "git+https://github.com/DanMcInerney/pymetasploit3.git",
]

# Packages which are only needed for testing code.
TEST_PACKAGES = [
    "pytest ~= 6.2.4",
]  # type: List[str]

# Loading the "long description" from the projects README file.
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="metasploit-gym",
    version=_VERSION,
    author="PhreakAI (@SJCaldwell)",
    author_email="shane@phreak.ai",
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/phreakAI/metasploit-gym",
    download_url="https://github.com/phreakAI/metasploit-gym/releases",
    # Contained modules and scripts:
    packages=setuptools.find_packages(),
    package_data={},
    install_requires=REQUIRED_PACKAGES,
    tests_require=REQUIRED_PACKAGES + TEST_PACKAGES,
    # PyPI package information:
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    license="MIT License",
    python_requires=">=3.6",
    keywords=" ".join(
        [
            "metasploit" "hacking" "pentesting" "pentest" "metasploit-gym" "metasploit-gymnasium",
            "Gymnasium" "Gym",
            "OpenAI-Gym",
            "Reinforcement-Learning",
            "Reinforcement-Learning-Environment",
        ]
    ),
    entry_points={
        "console_scripts": [
            "metasploit_gym = metasploit_gym.cli:main",
        ],
    },
)
