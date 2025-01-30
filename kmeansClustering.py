import os
import re
import shutil
import matplotlib.pyplot as plt
from indicnlp.tokenize.sentence_tokenize import sentence_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from langdetect import detect

# Directory containing text files
directory_path = '/home/megh/Desktop/Hindi Articles'

# Path to the Marathi stopwords file
marathi_stopwords_file_path = '/home/megh/Desktop/stopwords_marathi.txt'

# Path to the English stopwords file
english_stopwords_file_path = '/home/megh/Desktop/stopwords_english.txt'

# Path for the output file in the same directory as the raw text files
output_file_path = os.path.join(directory_path, 'clustering_results.txt')

# Function to load stopwords from a file
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = set(line.strip() for line in file)
    return stopwords

# Load stopwords for Marathi and English
marathi_stopwords = load_stopwords(marathi_stopwords_file_path)
english_stopwords = load_stopwords(english_stopwords_file_path)

# Function to detect language of the text
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'

# Preprocess text based on language
def preprocess_text(text, lang):
    if lang == 'mr':
        # Tokenize Marathi sentences
        sentences = sentence_split(text, lang='mr')
        stop_words = marathi_stopwords
    elif lang == 'en':
        # Split English text by sentences using regex
        sentences = re.split(r'(?<=[.!?])\s+', text)
        stop_words = english_stopwords
    else:
        # If language is unknown, return the original text
        return text

    # Preprocess each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        # Remove punctuation
        sentence = re.sub(r'[^\w\s]', '', sentence)
        
        # Tokenize words
        words = sentence.split()
        
        # Remove stopwords
        filtered_words = [word for word in words if word not in stop_words]
        
        # Reconstruct the sentence
        preprocessed_sentence = ' '.join(filtered_words)
        preprocessed_sentences.append(preprocessed_sentence)
    
    # Reconstruct the text
    preprocessed_text = ' '.join(preprocessed_sentences)
    return preprocessed_text

# Load text data from files
texts = []
file_names = []
languages = []

for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Detect language of the text
        lang = detect_language(text)
        languages.append(lang)
        
        # Preprocess the text based on detected language
        preprocessed_text = preprocess_text(text, lang)
        texts.append(preprocessed_text)
        file_names.append(filename)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Function to find the optimal number of clusters automatically
def find_optimal_clusters(data, max_clusters=10):
    distortions = []
    silhouette_scores = []
    K = range(2, min(max_clusters + 1, data.shape[0]))  # Number of clusters should be <= number of samples
    
    for k in K:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
        
        labels = kmeans.labels_
        if len(set(labels)) > 1:  # Silhouette score needs at least 2 clusters
            silhouette_avg = silhouette_score(data, labels)
            silhouette_scores.append(silhouette_avg)
        else:
            silhouette_scores.append(-1)

    # Plot Elbow Method
    plt.figure(figsize=(8, 6))
    plt.plot(K, distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')
    plt.grid(True)
    plt.show()

    # Plot Silhouette Scores
    plt.figure(figsize=(8, 6))
    plt.plot(K, silhouette_scores, marker='x', color='orange')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Method')
    plt.grid(True)
    plt.show()

    # Find the optimal number of clusters based on silhouette score
    valid_silhouettes = [score for score in silhouette_scores if score != -1]
    if valid_silhouettes:
        optimal_k = K[silhouette_scores.index(max(valid_silhouettes))]
    else:
        optimal_k = 2
    return optimal_k

# Automatically determine the optimal number of clusters
optimal_clusters = find_optimal_clusters(X, max_clusters=10)
print(f"Optimal number of clusters: {optimal_clusters}")

# Apply K-Means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, n_init=10, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Evaluate clustering
if len(set(labels)) > 1:
    silhouette_avg = silhouette_score(X, labels)
    print(f"Silhouette Score: {silhouette_avg}")
else:
    print("Silhouette Score cannot be calculated with fewer than 2 clusters.")

# Create cluster directories and copy files into them
for cluster_num in set(labels):
    cluster_dir = os.path.join(directory_path, f'Cluster_{cluster_num}')
    os.makedirs(cluster_dir, exist_ok=True)

for file_name, label in zip(file_names, labels):
    original_file_path = os.path.join(directory_path, file_name)
    target_dir = os.path.join(directory_path, f'Cluster_{label}')
    shutil.copy(original_file_path, target_dir)

# Save clustering results to a text file
with open(output_file_path, 'w', encoding='utf-8') as file:
    for file_name, label in zip(file_names, labels):
        file.write(f"{file_name}: Cluster {label}\n")

print(f"\nClustering results written to {output_file_path}")
print(f"Files copied to the corresponding cluster directories in {directory_path}.")