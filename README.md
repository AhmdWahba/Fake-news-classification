# Fake News Classification

This project addresses the detection of fake news using Natural Language Processing (NLP) techniques. It utilizes pretrained GloVe word embeddings combined with a logistic regression model to classify news headlines as either fake or real.

## Problem Statement

The spread of fake news presents a major threat to public trust and informed decision-making. This project builds a binary classifier that distinguishes between genuine and fabricated news headlines based on semantic and syntactic features extracted from the text.

## Technologies Used

- Python 3
- Jupyter Notebook
- Scikit-learn
- GloVe Embeddings (glove.6B.100d.txt)
- NumPy, Pandas, Matplotlib, Seaborn

## Dataset

The dataset is included in `New_Task.csv` and contains news headlines labeled as either fake or real.

### Preprocessing Steps

- Lowercasing
- Removing non-alphabetic characters
- Tokenization
- Padding sequences to fixed length

## Approach

1. **Tokenization and Padding**  
   Headlines were converted into sequences using Keras Tokenizer. Sequences were padded to ensure consistent input length.

2. **Word Embeddings**  
   GloVe pretrained embeddings (100-dimensional vectors) were loaded and used to create the embedding matrix.

3. **Model**  
   A Logistic Regression model was trained using the average of the word vectors for each headline as input features.

4. **Evaluation**  
   The dataset was split into 80% training and 20% testing data. Model performance was evaluated using accuracy, precision, recall, and F1 score.

## Results

### Confusion Matrix

|               | Predicted Fake | Predicted Real |
|---------------|----------------|----------------|
| **Actual Fake** | 189            | 11             |
| **Actual Real** | 13             | 187            |

### Classification Report

| Class       | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| Fake (0)    | 0.94      | 0.95   | 0.95     |
| Real (1)    | 0.94      | 0.93   | 0.93     |
| **Average** | **0.94**  | **0.94** | **0.94**   |

### Accuracy

Accuracy: 0.94


The model demonstrated strong performance with high precision and recall for both classes, indicating that it generalizes well to unseen data.

## File Structure

| File               | Description                                      |
|--------------------|--------------------------------------------------|
| `Fake_News.ipynb`  | Main notebook containing code and explanations   |
| `New_Task.csv`     | Dataset of news headlines                        |
| `midterm.html`     | HTML export of the notebook (optional)           |

## Note on GloVe Embeddings

This project uses `glove.6B.100d.txt` for generating word vectors. Due to its size, it is not included in this repository.

To download the GloVe file:

wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip

Place glove.6B.100d.txt in the same directory as the notebook, or update the file path accordingly.

#License

This project is for academic and educational purposes.
