import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from preprocess import clean_text

class ResumeRankerTFIDF:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit(self, job_description, resumes):
        jd_clean = clean_text(job_description)
        resumes_clean = [clean_text(r) for r in resumes]

        corpus = [jd_clean] + resumes_clean
        tfidf_matrix = self.vectorizer.fit_transform(corpus)

        jd_vec = tfidf_matrix[0]
        resume_vecs = tfidf_matrix[1:]

        similarities = cosine_similarity(jd_vec, resume_vecs).flatten()
        ranked = sorted(list(zip(resumes, similarities)), key=lambda x: x[1], reverse=True)
        return ranked

    def save_model(self, path="saved_models/vectorizer.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load_model(self, path="saved_models/vectorizer.pkl"):
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)

class ResumeRankerBERT:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def fit(self, job_description, resumes):
        jd_emb = self.model.encode(job_description, convert_to_tensor=True)
        resume_embs = self.model.encode(resumes, convert_to_tensor=True)

        similarities = util.cos_sim(jd_emb, resume_embs).cpu().numpy().flatten()
        ranked = sorted(list(zip(resumes, similarities)), key=lambda x: x[1], reverse=True)
        return ranked