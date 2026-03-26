from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from preprocess import clean_text

# Global model load for performance
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

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
        return sorted(list(zip(resumes, similarities)), key=lambda x: x[1], reverse=True)

class ResumeRankerBERT:
    def __init__(self):
        self.model = bert_model

    def fit(self, job_description, resumes):
        jd_emb = self.model.encode(job_description, convert_to_tensor=True)
        resume_embs = self.model.encode(resumes, convert_to_tensor=True)
        similarities = util.cos_sim(jd_emb, resume_embs).cpu().numpy().flatten()
        return sorted(list(zip(resumes, similarities)), key=lambda x: x[1], reverse=True)

def rank(job_description, resumes, method="tfidf"):
    if method == "tfidf":
        return ResumeRankerTFIDF().fit(job_description, resumes)
    return ResumeRankerBERT().fit(job_description, resumes)