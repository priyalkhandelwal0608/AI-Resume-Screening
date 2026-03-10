# AI Resume Screening

AI Resume Screening is a Natural Language Processing (NLP) project that ranks resumes against job descriptions using two approaches:
- **TF‑IDF + Cosine Similarity** (keyword‑based matching)
- **Sentence‑BERT Embeddings** (semantic similarity)

This project demonstrates how recruiters can automate resume shortlisting by comparing candidate resumes with job requirements.

---

# Features
- Clean text preprocessing (lowercasing, stopword removal, lemmatization).
- TF‑IDF vectorization with cosine similarity scoring.
- Semantic similarity using `sentence-transformers` (BERT).
- Flask web app with an attractive HTML interface.
- Resume ranking results displayed with similarity scores.

---

#Project Structure
ai-resume-screening/
│── app.py                # Flask app with HTML interface for resume screening
│── main.py               # Console demo to run ranking directly
│── ranker.py             # Contains ResumeRankerTFIDF and ResumeRankerBERT classes
│── preprocess.py         # Text cleaning and preprocessing utilities (NLTK)
│── requirements.txt      # Python dependencies
│── README.md             # Project documentation
│
├── data/                 # Input data folder
│   ├── job_description.txt   # Sample job description
│   └── resumes.txt           # Sample resumes (separated by ---)
│
├── __pycache__/          # Auto-generated Python cache files
│
└── saved_models/         #  Folder for saved TF-IDF vectorizer or embeddings
