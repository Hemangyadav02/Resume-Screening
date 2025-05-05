import os
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

# Simple text cleaning function
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text

# Load and clean Job Description
with open('job_description.txt', 'r') as file:
    jd_text = clean_text(file.read())

# Load and clean Resumes
resume_dir = 'resumes'
resumes = {}
for filename in os.listdir(resume_dir):
    if filename.endswith('.txt'):
        with open(os.path.join(resume_dir, filename), 'r') as file:
            resumes[filename] = clean_text(file.read())

# Combine all texts (JD + Resumes) for TF-IDF vectorization
documents = [jd_text] + list(resumes.values())

# Vectorize with stop word removal
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

# Compute Cosine Similarity between JD (index 0) and each resume
similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

# Display Rankings
print("\nResume Match Scores:")
ranked_resumes = sorted(zip(resumes.keys(), similarities), key=lambda x: x[1], reverse=True)
for filename, score in ranked_resumes:
    print(f"{filename}: {score:.2f}")
