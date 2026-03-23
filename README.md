# AI Resume Screening & Ranking System

An intelligent NLP-based recruitment tool that ranks resumes against job descriptions using both keyword-based matching and semantic context.

---

## 🌟 Features
* **Text Preprocessing**: Automated cleaning pipeline using NLTK for lowercasing, stopword removal, and lemmatization.
* **TF-IDF Similarity**: Ranks resumes based on term frequency and keyword overlap using Cosine Similarity.
* **Sentence-BERT (SBERT)**: High-accuracy semantic ranking that understands the meaning behind job requirements, not just the keywords.
* **Flask Web Interface**: Interactive web dashboard to paste job descriptions and batch-process resumes.
* **Modular Architecture**: Separated logic for preprocessing, ranking engines, and web deployment.

---

## 📂 Project Structure
```text
ai-resume-screening/
│── app.py                # Flask web application interface
│── main.py               # Terminal-based demo script
│── ranker.py             # TF-IDF and BERT ranking logic classes
│── preprocess.py         # NLP text cleaning and lemmatization
│── requirements.txt      # List of Python dependencies
│── README.md             # Project documentation
├── data/                 # Sample input data
│   ├── job_description.txt
│   └── resumes.txt       # Multiple resumes separated by '---'
├── static/               # CSS styles for the web UI
└── templates/            # HTML files (index.html)
