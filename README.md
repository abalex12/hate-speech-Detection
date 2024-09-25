# Hate Speech Detection

This project demonstrates the implementation of a machine learning model to automatically detect hate speech in social media posts, specifically tweets. Hate speech can take many forms, but it generally involves communication that attacks or uses derogatory language based on characteristics such as religion, race, ethnicity, nationality, gender, or any other identity factor. Given the rise of harmful speech online, this project aims to classify tweets into categories such as Hate Speech, Offensive Language, or No Hate and Offensive content.

## Dataset

The dataset used in this project is sourced from Kaggle, originally collected from Twitter. It contains the following columns:

- **index**: An index column.
- **count**: The frequency of certain keywords.
- **hate_speech**: The level of hate speech in the tweet.
- **offensive_language**: The level of offensive language used in the tweet.
- **neither**: Whether the tweet contains neither hate nor offensive content.
- **class**: The classification of the tweet:
  - `0`: Hate Speech
  - `1`: Offensive Language
  - `2`: No Hate and Offensive
- **tweet**: The actual text of the tweet.

## Approach and Methodology

### Data Preprocessing

The raw tweets contain noisy data that needs to be cleaned before training a model. The following steps are used to clean the data:

1. Convert text to lowercase.
2. Remove URLs, HTML tags, punctuation, and numeric values.
3. Remove common stopwords (e.g., "and", "the", "is") which do not contribute significantly to the meaning.
4. Apply stemming to reduce words to their base forms (e.g., "running" becomes "run").

### Feature Extraction

To convert the cleaned text data into a format that can be used by a machine learning model, we use the `CountVectorizer` from Scikit-learn. This tool converts text into a matrix of token counts (also known as a Bag of Words model), where each word's occurrence is represented as a feature for training the model.

### Splitting Data

The dataset is split into a training set and a testing set. This is important to evaluate the model's performance on unseen data.

### Model Training

A Decision Tree Classifier is used as the machine learning model to classify the tweets. Decision Trees are often used for classification tasks and work by learning decision rules inferred from the data.

### Model Testing

After training the model, it is tested with new input to see how well it performs in identifying hate speech. The model is designed to classify input sentences correctly.

## Project Structure

- **Data Preprocessing**: The function to clean the dataset prepares the raw tweet data.
- **Feature Extraction**: The `CountVectorizer` converts the text data into a numerical format suitable for machine learning.
- **Model Training**: A `DecisionTreeClassifier` is trained on the preprocessed data to detect hate speech.
- **Prediction**: The trained model can classify new text as either Hate Speech, Offensive Language, or No Hate and Offensive.

## Key Features

- **Data Preprocessing**: Effective cleaning of noisy Twitter data, including removal of URLs, special characters, and stopwords.
- **Model Accuracy**: Decision Tree model trained on hate speech data for precise classification.
- **Text Processing Techniques**: Use of NLP techniques like stemming and stopword removal to improve model performance.

## Results

This project successfully trains a machine learning model to classify tweets into categories based on their content. By preprocessing the data, applying a `CountVectorizer`, and using a `DecisionTreeClassifier`, we can detect hate speech with a reasonable degree of accuracy.

## Future Improvements

- **Model Optimization**: The model can be further optimized by trying more advanced classifiers like Random Forest or Gradient Boosting, or by tuning hyperparameters of the current model.
- **Deep Learning**: Implementing deep learning models such as LSTMs or transformers may improve accuracy for large-scale datasets.
- **More Data**: Incorporating more diverse datasets from other social media platforms would improve the robustness of the model.

## Example Output

The model can be used to classify sentences, such as identifying hate speech in various test cases.

## Requirements

- Python 3.x
- Scikit-learn
- NLTK
- Pandas
- NumPy
