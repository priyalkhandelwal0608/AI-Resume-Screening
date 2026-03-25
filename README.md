# AI Resume Screening & Ranking System  

A high-performance, NLP-powered recruitment tool that automatically ranks resumes against job descriptions using both keyword-based matching and deep semantic understanding. Built with **Flask**, **Scikit-Learn**, and **Sentence-Transformers**.

---

##  Features


- **Text Preprocessing Pipeline**: Cleans and normalizes text using tokenization, stopword removal, and lemmatization.
- **TF-IDF Similarity**: Uses **TfidfVectorizer** and **Cosine Similarity** to measure keyword overlap between resumes and job descriptions.
- **Semantic Matching (SBERT)**: Leverages transformer-based embeddings to understand contextual meaning beyond keywords.

---

##  Tech Stack

* **Backend:** Python 3.x, Flask  
* **Data Processing:** Pandas, NumPy  
* **Machine Learning & NLP:**  
  - Scikit-Learn (TfidfVectorizer, Cosine Similarity)  
  - Sentence-Transformers (all-MiniLM-L6-v2)  
  - NLTK (text preprocessing)  
* **Frontend:** HTML5, CSS3  
* **Deployment:** Flask (Local / Cloud)

---

##  Installation & Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ai-resume-screening.git
    cd ai-resume-screening
    ```

2. **Create virtual environment:**
    ```bash
    python -m venv venv
    ```

3. **Activate environment:**
    ```bash
    # Windows
    venv\Scripts\activate

    # macOS/Linux
    source venv/bin/activate
    ```

4. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5. **Run the application:**
    ```bash
    python app.py
    ```

6. Open browser:
    ```
    http://127.0.0.1:5000
    ```

---

##  Core Methodology

### 1. TF-IDF Based Ranking
The system preprocesses job descriptions and resumes using NLP techniques and converts them into numerical vectors using **TfidfVectorizer**.  
It then computes **Cosine Similarity** between the job description and each resume to rank candidates based on keyword relevance.

### 2. Semantic Ranking using SBERT
Using **Sentence-Transformers (all-MiniLM-L6-v2)**, the system generates dense embeddings for job descriptions and resumes.  
It computes similarity using cosine distance, allowing it to understand semantic meaning rather than just keyword overlap.


---

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
