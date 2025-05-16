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

- **Python 3.13+**: Main programming language.
- **Flask**: Web application framework.
- **Scikit-learn**: Used for model building and evaluation.
- **Pandas** and **NumPy**: For data handling and analysis.
- **Heroku**: Platform used for deployment.
- **Pickle**: For model serialization.

---

## Setup Instructions

### Prerequisites

1. Install **Python 3.13+**.
2. Install **Git** to clone the repository.
3. Have a Heroku account ready if deploying to the cloud.

### Step 1: Clone the repository

```bash
git clone https://github.com/Pavani89/capstone_project-sentiment-based-product-recommendation-system.git
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

Ensure the following pickle files are in the `pickle_file` directory:
- `model.pkl`: Sentiment analysis model.
- `user_final_rating.pkl`: Collaborative filtering rating model.
- `count_vector.pkl`: CountVectorizer object.
- `tfidf_transformer.pkl`: TF-IDF transformer.
- `RandomForest_classifier.pkl`: Checkpoint file for randome forest model.

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

### Digital Ocean Deployment

Here are 5 summarized steps to deploy the app on DigitalOcean:

---

### **1. Set Up a Droplet**
- Access your [DigitalOcean account](https://www.digitalocean.com/).
- Launch a new droplet with Ubuntu 22.04 or similar.
- Choose specs and region, then create the droplet.
- SSH into the droplet:
  ```bash
  ssh root@your_droplet_ip
  ```

---

### **2. Install Required Software**
- Run system updates and install essentials:
  ```bash
  sudo apt update && sudo apt upgrade -y
  sudo apt install python3 python3-pip python3-venv nginx git -y
  ```
- Install Gunicorn:
  ```bash
  pip3 install gunicorn
  ```

---

### **3. Upload and Configure Flask App**
- Clone or transfer the app:
  ```bash
  git clone https://github.com/Pavani89/capstone_project-sentiment-based-product-recommendation-system.git /var/www/flask-app
  cd /var/www/flask-app
  ```
- Set up the environment and dependencies:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```
- Test with Gunicorn:
  ```bash
  gunicorn --bind 0.0.0.0:8000 app:app
  ```

---

## File Structure

```
sentiment-based-recommendations/
├── app.py                    # Flask application
├── model.py                  # Core recommendation logic
├── checkpoints               # Directory with intermediate model objects 
│   ├── xgn_best_model.pkl
├── pickle_file/              # Directory with saved model objects
│   ├── model.pkl
│   ├── final_model.pkl
│   ├── user_final_rating.pkl
│   ├── count_vector.pkl
│   ├── tfidf_transformer.pkl
│   ├── user_final_rating.pkl
├── requirements.txt          # Required packages
├── README.md                 # Project documentation
├── data                      # Test dataset
│   ├── sample30.csv
└── templates                 # HTML templates for the web app
│   ├── index.html
```

---

## Conclusion

This sentiment-aware product recommendation system significantly enhances user experience by blending collaborative filtering with sentiment insights. The system provides more meaningful and relevant suggestions and is accessible via a lightweight Flask app, with deployment supported on DigitalOcean.