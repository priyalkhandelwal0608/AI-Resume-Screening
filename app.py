from flask import Flask, request, render_template_string
from ranker import ResumeRankerTFIDF, ResumeRankerBERT

app = Flask(__name__)
ranker_tfidf = ResumeRankerTFIDF()
ranker_bert = ResumeRankerBERT()

# Modern HTML template
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI Resume Screening</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1f1c2c, #928dab);
            margin: 0;
            padding: 0;
            color: #333;
        }
        header {
            background: #2c3e50;
            color: #fff;
            padding: 20px;
            text-align: center;
        }
        header h1 {
            margin: 0;
            font-size: 2.2em;
        }
        .container {
            max-width: 900px;
            margin: 40px auto;
            background: #fff;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        }
        label {
            font-weight: bold;
            display: block;
            margin-top: 20px;
            margin-bottom: 8px;
        }
        textarea {
            width: 100%;
            height: 120px;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #ccc;
            resize: vertical;
            font-size: 14px;
        }
        button {
            background: #2c3e50;
            color: #fff;
            border: none;
            padding: 12px 20px;
            margin-top: 20px;
            margin-right: 10px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
        }
        button:hover {
            background: #34495e;
        }
        .result {
            margin-top: 30px;
        }
        .resume {
            background: #f9f9f9;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            border: 1px solid #eee;
        }
    </style>
</head>
<body>
    <header>
        <h1>AI Resume Screening</h1>
        <p>Rank resumes against job descriptions using TF-IDF or BERT</p>
    </header>
    <div class="container">
        <form method="POST" action="/rank">
            <label>Job Description:</label>
            <textarea name="job_description" required></textarea>

            <label>Resumes (separate with ---):</label>
            <textarea name="resumes" required></textarea>

            <button type="submit" name="method" value="tfidf">Rank with TF-IDF</button>
            <button type="submit" name="method" value="bert">Rank with BERT</button>
        </form>

        {% if results %}
        <div class="result">
            <h2>Ranking Results ({{ method.upper() }})</h2>
            {% for r, s in results %}
                <div class="resume">
                    <b>Score:</b> {{ "%.4f"|format(s) }} <br>
                    <b>Resume:</b> {{ r[:300] }}...
                </div>
            {% endfor %}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_PAGE)

@app.route("/rank", methods=["POST"])
def rank_resumes():
    job_description = request.form["job_description"]
    resumes = request.form["resumes"].split("---")
    method = request.form["method"]

    if method == "tfidf":
        results = ranker_tfidf.fit(job_description, resumes)
    else:
        results = ranker_bert.fit(job_description, resumes)

    return render_template_string(HTML_PAGE, results=results, method=method)

if __name__ == "__main__":
    app.run(debug=True)