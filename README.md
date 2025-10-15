# Sentiment-Based Product Recommendation System

This repository contains a **Sentiment-Based Product Recommendation System**, which combines **Collaborative Filtering** techniques and **Sentiment Analysis** to recommend products based on user preferences and sentiment from reviews. The system includes a Flask web application deployed on **Heroku** for live interaction.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Model Details](#model-details)
- [Deployment](#deployment)
- [File Structure](#file-structure)
- [Conclusion](#conclusion)

---

## Project Overview

E-commerce platforms rely heavily on personalized recommendations. This project enhances such recommendations by analyzing customer reviews and ratings. It integrates **Sentiment Analysis** with **User-User Collaborative Filtering** to deliver more tailored and relevant product suggestions.

The system enables users to:
- Receive product recommendations based on preferences of similar users.
- Refine recommendations by analyzing the sentiment of reviews.

---

## Key Features

1. **User-User Collaborative Filtering**: Offers recommendations by identifying users with similar tastes.
2. **Sentiment Filtering**: Emphasizes items with positively reviewed sentiment.
3. **Flask Web Application**: Provides an interactive and user-friendly interface.
4. **Cloud Deployment**: Easily accessible through a **Heroku** deployment.
5. **Reusable Models**: Stores trained models and preprocessing artifacts as pickle files for efficient reuse.

---

## Technologies Used

- **Python 3.11.9**: Main programming language
- **Flask**: Web application framework
- **Scikit-learn**: Used for model building and evaluation
- **Pandas** and **NumPy**: For data handling and analysis
- **NLTK** and **spaCy**: For text processing and analysis
- **Heroku**: Platform used for deployment
- **Pickle**: For model serialization

---

## Setup Instructions

### Prerequisites

1. Install **Python 3.13+**.
2. Install **Git** to clone the repository.
3. Have a Heroku account ready if deploying to the cloud.

### Step 1: Clone the repository

```bash
git clone https://github.com/rajeshmishra05/capstone_project-sentiment-based-product-recommendation-system.git
```

### Step 2: Install dependencies

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Step 3: Load models and data

Ensure the following pickle files are in the `pickle_files` directory:
- `model.pkl`: Sentiment analysis model
- `final_model.pkl`: Final trained model
- `user_final_rating.pkl`: Collaborative filtering rating matrix
- `count_vector.pkl`: CountVectorizer object
- `tfidf_transformer.pkl`: TF-IDF transformer
- `RandomForest_classifier.pkl`: Random Forest classifier model

Also, ensure the `sample30.csv` dataset under the folder `data` is available for testing.

### Step 4: Run the Flask application

Start the Flask development server:

```bash
python app.py
```

Open a browser and navigate to `http://127.0.0.1:5000/` to view the app.

---

## Model Details

### 1. Sentiment Analysis

- **Input**: Cleaned review text.
- **Model**: Logistic Regression using TF-IDF features.
- **Output**: Sentiment classification (Positive, Negative, Neutral).
- **Metrics**: Accuracy, Precision, Recall, F1-Score.

### 2. User-User Collaborative Filtering

- **Input**: User-product interaction matrix.
- **Technique**: Cosine similarity between users.
- **Output**: Estimated ratings for products not yet rated.
- **Metric**: RMSE.

### Integration

The final recommendation engine merges sentiment filtering with collaborative filtering predictions. It ranks products by their predicted rating and sentiment polarity.

---

## Deployment

### Local Deployment

Follow the instructions above to set up and run locally.

### Heroku Deployment

Follow these steps to deploy the application on Heroku:

---

### **1. Prerequisites**
- Install [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
- Ensure you have Git installed
- Have a Heroku account

---

### **2. Prepare for Deployment**
- Ensure you have these files in your project:
  - `Procfile` with the content: `web: gunicorn app:app`
  - `requirements.txt` with all dependencies
  - `runtime.txt` specifying Python version
- All pickle files should be in the correct directories:
  ```
  pickle_files/
  ├── model.pkl
  ├── user_final_rating.pkl
  ├── count_vector.pkl
  ├── tfidf_transformer.pkl
  └── RandomForest_classifier.pkl
  ```

---

### **3. Deploy to Heroku**
1. Login to Heroku CLI:
   ```bash
   heroku login
   ```

2. Create a new Heroku app:
   ```bash
   heroku create your-app-name
   ```

3. Push your code:
   ```bash
   git push heroku main
   ```

4. Ensure at least one instance is running:
   ```bash
   heroku ps:scale web=1
   ```

5. Open the deployed app:
   ```bash
   heroku open
   ```

---

### **4. Troubleshooting**
- View logs if there are issues:
  ```bash
  heroku logs --tail
  ```
- Check if all files are present:
  ```bash
  heroku run ls
  ```
- Verify Python packages:
  ```bash
  heroku run pip list
  ```

---

## File Structure

```
sentiment-based-recommendations/
├── app.py                    # Flask application
├── model.py                  # Core recommendation logic
├── generate_matrix.py        # Script to generate recommendation matrix
├── checkpoints/              # Directory with intermediate model objects 
│   └── xgb_best_model.pkl
├── pickle_files/            # Directory with saved model objects
│   ├── model.pkl            # Sentiment analysis model
│   ├── final_model.pkl      # Final trained model
│   ├── user_final_rating.pkl # User recommendations matrix
│   ├── count_vector.pkl     # Text vectorization model
│   ├── tfidf_transformer.pkl # TF-IDF transformation model
│   └── RandomForest_classifier.pkl # Random Forest model
├── requirements.txt         # Required packages
├── runtime.txt             # Python version specification
├── Procfile               # Heroku deployment configuration
├── README.md              # Project documentation
├── Sentiment-based-product-recommendation-system.ipynb  # Jupyter notebook with model development
├── data/                  # Test dataset
│   ├── sample30.csv
│   └── Data+Attribute+Description.csv
└── templates/             # HTML templates for the web app
    └── index.html
```

---

## Conclusion

This sentiment-aware product recommendation system significantly enhances user experience by blending collaborative filtering with sentiment insights. The system provides more meaningful and relevant suggestions and is accessible via a lightweight Flask app, with deployment supported on DigitalOcean.