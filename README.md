# PUffy
*Put a label on it*
## Welcome to PUffy! 🐡
PUffy is a Python library designed for creating the best possible models given a dataset with **positive-unlabeled data**. It implements two-step classification approach. The package is also suitable for considerations on **PU classification** with fully labeled dataset, as the package presents labeling methods to artificially mimic labeling mechanisms resulting in positive-unlabeled data. Puffy consists of four modules, namely `labelling_methods`, `rn_methods`, `training` and `pu_pipeline`.

So far, it offers:

🐡 4 labeling methods:
  - one method assuming SCAR scenario,

  - three methods being variants of SAR setting,

🐡 3 techniques for identifying reliable-negatives sample from the unlabeled data:
  - a technique based on KMeans,
  
  - a technique based on KNN,
  
  - a technique based on Spy analysis,
  
  - an iterative classification model for identifying reliable negatives, using Logistic Regression classification,

🐡 visualizations of the model performance,

🐡 a naive classification method for comparison,

🐡 a pipeline, with which presented experiments were conducted.

As of today, the package is meant for binary classification problems.

## Documentation
Full documentation is available here: [**I'm here!**](https://kopyta.github.io/PUffy/)


## Installation

Thank you for choosing Puffy! This section provides instructions on how to install and set up PUffy on your system.

### Prerequisites

Before you begin, ensure you have the following prerequisites:

🐡 Python 3.6 or later installed on your system. You can download Python from [https://www.python.org/downloads/](https://www.python.org/downloads/).
🐡 Ensure `pip` (Python package installer) is available. If it's not installed, you can follow the instructions at [https://pip.pypa.io/en/stable/installation/](https://pip.pypa.io/en/stable/installation/).

### Step by step 

### Step 1: Locate the Package

Visit the GitHub repository to find the latest release of Puffy:
[https://github.com/kopyta/PUffy](https://github.com/kopyta/PUffy)

### Step 2: Choose a Release

Navigate to the "Releases" section and choose the version you want to install. You can find the releases at:
[https://github.com/kopyta/PUffy/releases](https://github.com/kopyta/PUffy/releases)

### Step 3: Download the Package

Download the appropriate package for your platform:

#### Option 1: Wheel file (`.whl`)
Run the following commands:
```bash
wget https://github.com/kopyta/PUffy/releases/v1.0/puffy-1.0-py3-none-any.whl
pip install puffy-1.0-py3-none-any.whl
```
or
#### Option 2: Tarball file (`.tar.gz`)
Run the following commands:
```bash
wget https://github.com/kopyta/PUffy/releases/v1.0/puffy-1.0.tar.gz
pip install puffy-1.0.tar.gz
```

### Step 4: Congratulations! 🐡
You've successfully installed Puffy! Get ready to embark on the exciting challenges of PU data classification. Enjoy exploring the capabilities of Puffy and make the most out of its features! For more information visit our documentation: [🐡**CLick**🐡](https://kopyta.github.io/PUffy/)

Happy coding! 🚀

##
The package is a result of the bachelor's thesis, origining from [this repository](https://github.com/zuzannakotlinska/PU_data_classification)
