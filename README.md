# AI Resume Screening & Ranking System

An intelligent NLP-based recruitment tool that ranks resumes against job descriptions using both keyword-based matching and semantic context.

---

##  Features
* **Text Preprocessing**: Automated cleaning pipeline using NLTK for lowercasing, stopword removal, and lemmatization.
* **TF-IDF Similarity**: Ranks resumes based on term frequency and keyword overlap using Cosine Similarity.
* **Sentence-BERT (SBERT)**: High-accuracy semantic ranking that understands the meaning behind job requirements, not just the keywords.
* **Flask Web Interface**: Interactive web dashboard to paste job descriptions and batch-process resumes.
* **Modular Architecture**: Separated logic for preprocessing, ranking engines, and web deployment.

---
Installation
Clone the Repository

Bash
git clone [https://github.com/your-username/ai-resume-screening.git](https://github.com/your-username/ai-resume-screening.git)
cd ai-resume-screening
Set Up Virtual Environment

Bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
Install Dependencies

Bash
pip install -r requirements.txt
Initialize NLTK Data

Python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
🚀 How to Run
Option 1: Web Interface (Recommended)
Launch the Flask app to use the graphical interface:

Bash
python app.py
Visit http://127.0.0.1:5000 in your browser.

Option 2: Console Demo
Run the ranking logic directly on the sample files in the data/ folder:

Bash
python main.py
📊 Technical Implementation
Vectorization: scikit-learn TfidfVectorizer.

Embeddings: sentence-transformers (all-MiniLM-L6-v2).

##  Project Structure
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
