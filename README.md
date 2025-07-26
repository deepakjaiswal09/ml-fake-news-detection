# 📰 Fake News Detection using Machine Learning

## 📌 Project Overview

Fake News Detection is a **Machine Learning**-based project that aims to classify whether a given news article is *genuine* or *fake*. The primary focus is on applying **Natural Language Processing (NLP)** techniques along with supervised learning models to analyze the text data and predict the authenticity of news content.

This project is ideal for college students and researchers looking to implement **text classification** problems using ML.

---

## 📂 Problem Statement

With the rapid spread of online news and social media posts, **misinformation and fake news** have become a major threat to society. This project provides an ML-based solution to classify news articles as **real** or **fake** using their content and title by training on labeled datasets.

---

## ⚙️ Tech Stack Used

| Category                | Tools / Libraries                                         |
|-------------------------|------------------------------------------------------------|
| Programming Language    | Python                                                    |
| ML Libraries            | Scikit-learn, Pandas, NumPy                               |
| NLP Libraries           | NLTK, re (Regex), TfidfVectorizer                         |
| ML Models Used          | Logistic Regression, Passive Aggressive Classifier, Random Forest, Naive Bayes |
| Visualization           | Matplotlib, Seaborn                                       |
| Dataset                 | Kaggle: `Fake and Real News Dataset`                      |
| IDE                     | Jupyter Notebook / VS Code                                |

---

## 🧠 Machine Learning Models Implemented

- ✅ **Logistic Regression**
- ✅ **Random Forest Classifier**
- ✅ **Naive Bayes**
- ✅ **Passive Aggressive Classifier**
- ✅ **Support Vector Machine (SVM)** _(optional depending on final implementation)_

---

## 🛠️ Project Workflow

1. **Data Collection**
   - Dataset contains news articles with labels (`real`, `fake`)
   - Source: Kaggle

2. **Data Cleaning and Preprocessing**
   - Removing punctuation, stopwords, and special characters
   - Tokenization and Lemmatization using `nltk`

3. **Exploratory Data Analysis (EDA)**
   - Word clouds, frequency distributions
   - Fake vs Real distribution graphs

4. **Text Vectorization**
   - TF-IDF Vectorizer
   - Bag-of-Words (optional)

5. **Model Training**
   - Trained on 80% data
   - Tested on 20% data
   - Cross-validation applied

6. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrix and ROC Curve

7. **Output**
   - Predicts if a given news is real or fake
   - Deployment-ready ML pipeline

---

## 📊 Sample Output

```bash
Enter the News Headline or Content: "Breaking: NASA announces Mars base in 2030"
Prediction: ✅ Real News
```

---

## 📁 Folder Structure

```bash
├── Fake-News-Detection/
│   ├── data/                       # Dataset files
│   ├── models/                     # Trained ML models (optional)
│   ├── notebook/                   # Jupyter notebooks
│   ├── src/                        # Python scripts
│   ├── README.md
│   ├── requirements.txt
│   └── LICENSE
```

---

## 📈 Results

| Model                      | Accuracy  |
|---------------------------|-----------|
| Logistic Regression        | 96.2%     |
| Random Forest Classifier   | 94.8%     |
| Naive Bayes                | 93.1%     |
| Passive Aggressive Class.  | 95.4%     |

---

## 📄 Research Reference Papers

- ["Fake News Detection on Social Media: A Data Mining Perspective"](https://arxiv.org/abs/1708.01967)
- ["Detecting Fake News with NLP and ML"](https://ieeexplore.ieee.org/document/9054105)

---

## 📌 How to Run This Project

1. Clone the repo:
```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the notebook:
```bash
jupyter notebook fake_news_detection.ipynb
```

---

## 🙋 Contact & Support

> This project was made with ❤️ by **Deepak Jaiswal**

📧 **Email:** jaiswaldeepak9238@gmail.com

Need Full Project with Explanation Video, PPT, Report & Source Code?

📩 Feel free to reach out!

---

## 📜 License

This project is licensed under the [MIT License](LICENSE). Feel free to use it for academic and learning purposes.

---
