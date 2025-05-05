This project implements an AI-powered resume screening system that automates the process of matching candidate resumes with job descriptions. It leverages Natural Language Processing (NLP) techniques to analyze and compare resumes, helping recruiters efficiently identify the most relevant candidates for a job role.

The system works by reading both the job description and resumes from text files, then preprocessing the text to ensure uniformity. Preprocessing includes tasks such as converting all text to lowercase, removing punctuation, and eliminating stopwords (common words that don’t add significant meaning to the text).

Once the text is cleaned, the system uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to transform the textual data into numerical vectors that capture the importance of words in the context of the job description and resumes. The system then computes cosine similarity, which measures the similarity between the job description and each resume based on these vectors.

After calculating the similarity scores, the system ranks the resumes from most to least relevant, allowing recruiters to quickly assess how well each candidate fits the job description. This automation greatly speeds up the process of resume screening, ensuring that recruiters focus on the most qualified candidates.
