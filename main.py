from ranker import ResumeRankerTFIDF, ResumeRankerBERT
import os

def load_data():
    jd_path = "data/job_description.txt"
    resumes_path = "data/resumes.txt"
    
    # Ensure directory and files exist
    if not os.path.exists("data"):
        os.makedirs("data")
        
    with open(jd_path, "w") as f:
        f.write("Looking for a Python Developer with experience in Flask and Machine Learning.")
    
    with open(resumes_path, "w") as f:
        f.write("Experienced Python developer specialized in Flask APIs.\n---\nData Scientist with ML expertise.")

    job_description = open(jd_path).read()
    # Splitting resumes by the separator as per original project structure
    resumes = open(resumes_path).read().split("\n---\n")
    return job_description, resumes

if __name__ == "__main__":
    jd, resumes = load_data()

    print("\n--- TF-IDF Ranking ---")
    tfidf_ranker = ResumeRankerTFIDF()
    for r, s in tfidf_ranker.fit(jd, resumes):
        print(f"Score: {s:.4f} | Resume: {r[:50].strip()}...")

    print("\n--- BERT Ranking ---")
    bert_ranker = ResumeRankerBERT()
    for r, s in bert_ranker.fit(jd, resumes):
        print(f"Score: {s:.4f} | Resume: {r[:50].strip()}...")