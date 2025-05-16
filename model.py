# Importing Libraries
import pandas as pd
import re, nltk, spacy
import pickle as pk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load pre-trained models and vectorizers
count_vector = pk.load(open('pickle_files/count_vector.pkl', 'rb'))
tfidf_transformer = pk.load(open('pickle_files/tfidf_transformer.pkl', 'rb'))
model = pk.load(open('pickle_files/model.pkl', 'rb'))
recommend_matrix = pk.load(open('pickle_files/user_final_rating.pkl', 'rb'))

# Load spaCy language model with disabled components for efficiency
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

# Read product data
product_df = pd.read_csv('/data/sample30.csv', sep=",")

# Text Preprocessing Functions

def remove_special_characters(text, remove_digits=True):
    """Remove special characters (and optionally digits) from text."""
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    return re.sub(pattern, '', text)

def to_lowercase(words):
    """Convert list of words to lowercase."""
    return [word.lower() for word in words]

def remove_punctuation_and_splchars(words):
    """Remove punctuation and special characters from list of words."""
    return [remove_special_characters(re.sub(r'[^\w\s]', '', word), True) for word in words if word]

stopword_list = stopwords.words('english')

def remove_stopwords(words):
    """Filter out stopwords from list of words."""
    return [word for word in words if word not in stopword_list]

def stem_words(words):
    """Stem words using Lancaster stemmer."""
    stemmer = LancasterStemmer()
    return [stemmer.stem(word) for word in words]

def lemmatize_verbs(words):
    """Lemmatize verbs in list of words using WordNet."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, pos='v') for word in words]

def normalize(words):
    """Normalize words by lowercasing, removing punctuation, and stopwords."""
    words = to_lowercase(words)
    words = remove_punctuation_and_splchars(words)
    words = remove_stopwords(words)
    return words

def lemmatize(words):
    """Lemmatize list of words."""
    return lemmatize_verbs(words)

def model_predict(text):
    """Predict sentiment label for given text using trained model."""
    word_vector = count_vector.transform(text)
    tfidf_vector = tfidf_transformer.transform(word_vector)
    return model.predict(tfidf_vector)

def normalize_and_lemmaize(input_text):
    """Normalize and lemmatize input text."""
    input_text = remove_special_characters(input_text)
    words = nltk.word_tokenize(input_text)
    words = normalize(words)
    return ' '.join(lemmatize(words))

def recommend_products(user_name):
    """Generate top 20 product recommendations with sentiment predictions for a user."""
    recommend_matrix = pk.load(open('pickle_files/user_final_rating.pkl', 'rb'))
    product_list = pd.DataFrame(recommend_matrix.loc[user_name].sort_values(ascending=False)[:20])
    product_frame = product_df[product_df.name.isin(product_list.index.tolist())]
    output_df = product_frame[['name', 'reviews_text']]
    output_df['lemmatized_text'] = output_df['reviews_text'].map(normalize_and_lemmaize)
    output_df['predicted_sentiment'] = model_predict(output_df['lemmatized_text'])
    return output_df

def top5_products(df):
    """Return top 5 products with highest positive sentiment percentage."""
    total_product = df.groupby(['name']).agg('count')
    rec_df = df.groupby(['name', 'predicted_sentiment']).agg('count').reset_index()
    merge_df = pd.merge(rec_df, total_product['reviews_text'], on='name')
    merge_df['%percentage'] = (merge_df['reviews_text_x'] / merge_df['reviews_text_y']) * 100
    merge_df = merge_df.sort_values(ascending=False, by='%percentage')
    return pd.DataFrame(merge_df['name'][merge_df['predicted_sentiment'] == 1][:5])
