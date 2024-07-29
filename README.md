# Author Stance Prediction

This project focuses on predicting the stance of authors from Arabic text inputs. Utilizing several natural language processing techniques and machine learning models, this tool preprocesses Arabic texts, extracts relevant features using ARABERT, and then classifies the stance into predefined categories. The project leverages libraries such as `transformers`, `scikit-learn`, `gensim`, `nltk`, and others for efficient processing and analysis.

## Installation

Ensure you have Python installed on your system. Then, clone this repository and navigate into the project directory. To install the required dependencies, run the following commands:

```bash
pip install transformers scikit-learn camel-tools gensim nltk huggingface_hub requests
```

## Datasets

The project uses custom datasets stored in CSV files (`train-3.csv` and `test-3.csv`). Ensure these files are present in the project directory. These datasets should contain the texts in Arabic along with their labels for training and testing purposes.

## Preprocessing

The Arabic texts are preprocessed to remove unnecessary characters, stopwords, and then stemmed using the `nltk` library. This step is crucial for cleaning the data and making it suitable for feature extraction.

## Feature Extraction

ARABERT, a BERT model pre-trained on Arabic language texts, is utilized for feature extraction. The project extracts pooled output features from ARABERT to represent the text inputs.

## Model Training

A Support Vector Classifier (SVC) model is trained on the extracted features along with TF-IDF vectorized text and additional features like encoded sentiments and targets. The model aims to classify the stance of the author based on the preprocessed text.

## Usage

After installation and dataset setup, run the included Jupyter notebook (`stance_prediction.ipynb`) to preprocess the data, train the model, and predict stances on unseen data. The notebook guides you through each step of the process, from reading the datasets to visualizing results.

```python
# Example: Preprocessing a new text input
preprocessed_text = preprocess_arabic_text("Your Arabic text here")
```

```python
# Example: Predicting the stance of a new text input
# Ensure to transform your input text using the same preprocessing and feature extraction steps as the training data
# stance_prediction = model.predict([transformed_text])
```

## Contributing

Contributions to the project are welcome. Please feel free to fork the repository, make your changes, and submit a pull request.
