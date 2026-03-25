from flask import Flask, render_template, request
import ranker 

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        jd = request.form.get("job_description", "")
        resumes_text = request.form.get("resumes", "")
        method = request.form.get("method", "tfidf")

        resumes = [r.strip() for r in resumes_text.split("---") if r.strip()]
        
        if not jd or not resumes:
            return render_template("index.html", error="Please provide both JD and Resumes.")

        results = ranker.rank(jd, resumes, method=method)
        return render_template("index.html", results=results, method=method)

    return render_template("index.html")

@app.route("/demo", methods=["GET", "POST"])
def demo():
    jd = """Seeking Data Scientist: Python, Machine Learning, SQL, AWS."""
    resumes = [
        "John: Python developer, 3 years ML, scikit-learn, SQL.",
        "Jane: NLP expert, TensorFlow, PyTorch, AWS cloud.",
        "Mike: Java developer, backend focused, limited ML."
    ]
    # Running BERT for the demo
    results = ranker.rank(jd, resumes, method="bert")
    return render_template("index.html", results=results, method="Demo (BERT)")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
