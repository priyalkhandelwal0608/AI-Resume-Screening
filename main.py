from ranker import ResumeRankerTFIDF, ResumeRankerBERT

def load_data():
    job_description = open("data/job_description.txt").read()
    resumes = open("data/resumes.txt").read().split("\n---\n")
    return job_description, resumes

if __name__ == "__main__":
    jd, resumes = load_data()

    print(" TF-IDF Ranking:")
    tfidf_ranker = ResumeRankerTFIDF()
    for r, s in tfidf_ranker.fit(jd, resumes):
        print(f"Score: {s:.4f} | Resume: {r[:60]}...")

    print("\n BERT Ranking:")
    bert_ranker = ResumeRankerBERT()
    for r, s in bert_ranker.fit(jd, resumes):
        print(f"Score: {s:.4f} | Resume: {r[:60]}...")
