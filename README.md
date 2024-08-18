# DNA Sequencing Machine Learning Project

This project applies machine learning techniques to DNA sequencing data to classify sequences and predict outcomes. This has been adapted from the original by [DNA Sequencing Classifier](https://github.com/krishnaik06/DNA-Sequencing-Classifier.git). I made some changes with the visualisations and played around with some of the analysis. 

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Modeling](#modeling)
- [Results](#results)
- [License](#license)

## Project Overview

The goal of this project is to develop a machine learning model that can accurately classify DNA sequences. The project explores various preprocessing techniques, model selection, and evaluation metrics to handle challenges such as class imbalance and feature selection.

## Installation

To run this project, you'll need to have Python installed along with the following packages:

- `numpy`
- `pandas`
- `scikit-learn`
- `imbalanced-learn`
- `matplotlib`
- `seaborn`

You can install the required packages using pip:

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn
```

## Usage
Clone the Repository:

```bash
git clone https://github.com/montahac/dna-sequencing-ml.git
cd dna-sequencing-ml
```

Run the Jupyter Notebook: Open the Jupyter Notebook `ml_dna.ipynb` to explore the data, preprocess it, and train the model.

Execute the Script: If you prefer running a script, execute `ml_dna.py` to perform the entire workflow.

## Data
The dataset used in this project consists of DNA sequences with associated labels. The data is preprocessed to handle missing values and normalize features. Ensure that the data files (human.txt, chimpanzee.txt, dog.txt) are located in the correct directory as specified in the script.

## Modeling
The project explores various machine learning models, including:

- Multinomial Naive Bayes
- Decision Trees
- Random Forests

## Results
The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. The project also includes visualizations to illustrate the model's performance and the impact of preprocessing techniques.

Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.






