# Analysis-using-Supervised-ML-Algorithms
The project applies supervised machine learning algorithms to an image dataset (MNIST), a tabular dataset (Spambase), and a text dataset (20NG) to interpret the data through analysis of top-ranked features.

# Problem Statement
<img width="1907" height="114" alt="image" src="https://github.com/user-attachments/assets/8c36d49a-0424-4838-ba7e-d12952f7784d" />

# Solution Summary
- Conducted **supervised classification** on three diverse datasets:
  - **MNIST**
  - **Spambase**
  - **20 Newsgroups (20NG)**

- **Classification Algorithms**:
  - **L2-regularized Logistic Regression**
  - **Decision Trees**

- **Data Preprocessing**:
  - MNIST:
    - Reshaped and binarized images.
    - Represented as sparse matrices.
  - Spambase:
    - Loaded tabular data with feature names.
  - 20NG:
    - Applied NLP pipeline:
      - Cleaning, lowercasing, tokenization, stopword removal, valid word filtering.
    - Vectorized using **TF-IDF**.

- **Model Interpretation**:
  - Logistic Regression:
    - Top 30 features by **highest absolute coefficient values**.
  - Decision Trees:
    - Top 30 features by **first splits**.
  - Enabled:
    - Visual interpretation (MNIST).
    - Direct feature naming (Spambase, 20NG).

- **Decision Tree Complexity Analysis (20NG)**:
  - Evaluated performance under varying tree sizes (depth, splits).
  - Explored impact of **model complexity**.

- **Performance Evaluation**:
  - Used **accuracy, precision, recall, f1-score** for comprehensive assessment.
  - Provided cross-dataset, cross-algorithm comparison.

- **Conclusion**:
  - Delivered interpretable classification pipelines across image, tabular, and text data.
  - Combined quantitative and qualitative insights into model behavior and feature importance.



