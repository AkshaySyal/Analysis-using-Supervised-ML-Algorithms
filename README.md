# Data-Analysis-using-Supervised-ML-Algorithms-Feature-Selection-Algorithms
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

# Problem Statement
<img width="1506" height="82" alt="image" src="https://github.com/user-attachments/assets/c70df008-15d9-4fad-9081-5c0fe42dc769" />

# Solution Summary
- Conducted **pairwise feature selection for text classification** on the **20 Newsgroups (20NG)** dataset.

- **Data Preprocessing**:
  - Removed non-alphanumeric characters and newlines.
  - Lowercased text, tokenized words.
  - Removed English stopwords.
  - Filtered valid words using **nltk.corpus.wordnet.synsets**.
  - Vectorized text with **TfidfVectorizer** → sparse matrix (11314 × 32863, sparsity ~0.9983).

- **Methodology**:
  - Split data into training and testing sets.
  - Performed feature selection with **SelectKBest** (top 200 features) using:
    - **Chi-squared (chi2)**
    - **Mutual Information Gain (mutual_info_classif)**
  - Applied two classifiers:
    - Logistic Regression (`penalty='l2'`, `solver='lbfgs'`, `max_iter=1000`, `multi_class='multinomial'`).
    - Decision Trees:
      - **Low-depth version**
      - **Max-depth version**
    - Both with `class_weight='balanced'`.

- **Key Insights**:
  - **Feature selection reduced accuracy** compared to HW3A-PB1 baseline:
    - Logistic Regression:
      - HW3A-PB1 → **0.7163**
      - chi2 → **0.5117**
      - mutual_info → **0.2554**
    - Low-depth Decision Tree:
      - HW3A-PB1 → **0.4565**
      - chi2 → **0.4141**
      - mutual_info → **0.1732**
    - Max-depth Decision Tree:
      - HW3A-PB1 → **0.4724**
      - chi2 → **0.4295**
      - mutual_info → **0.1710**
  - **Chi-squared outperformed Mutual Information** consistently across models.
    - Higher accuracy, precision, recall, and f1-score.
    - More effective at identifying relevant features for 20NG text classification.

- **Conclusion**:
  - Highlighted the critical role of feature selection criteria in model performance.
  - Demonstrated chi2’s superiority over mutual information in this text classification context.

# Problem Statement
<img width="1511" height="73" alt="image" src="https://github.com/user-attachments/assets/2d45210a-51f1-4f40-9fdb-a83646692636" />

# Solution Summary
- Conducted **L1 feature selection** for text classification on the **20 Newsgroups (20NG)** dataset.

- **Data Preprocessing**:
  - Removed non-alphanumeric characters and newlines.
  - Lowercased text, tokenized words.
  - Removed English stopwords.
  - Filtered valid words using **nltk.corpus.wordnet.synsets**.
  - Vectorized text with **TfidfVectorizer** → sparse matrix (11314 × 32863, sparsity ~0.9983).

- **Feature Selection Pipeline**:
  - Fitted **Lasso (L1-regularized regression)** model (`alpha=0.0027`) on training data.
  - Selected **top 200 features** based on absolute regression coefficients.
  - Reconstructed dataset using only these selected features.

- **Classification**:
  - Trained **Logistic Regression** model:
    - `penalty='l2'`, `max_iter=1000`, `solver='lbfgs'`, `multi_class='multinomial'`.
  - Evaluated model on L1-selected feature set.

- **Key Insights**:
  - **Substantial degradation in accuracy**:
    - Baseline (full features) → **0.7163**
    - After L1 selection (top 200) → **0.4658**
    - Indicates L1 selection reduced feature space but at a significant performance cost.
  - **Imbalanced class performance**:
    - Post-selection classification report showed:
      - Some classes (e.g., class 19) with **very low recall (0.03)**.
      - Others maintained relatively better metrics.
    - Suggests L1-selected features were unevenly discriminative across categories.

- **Conclusion**:
  - L1 regularization effectively reduced dimensionality but failed to preserve overall or balanced classification performance on the 20NG dataset.



