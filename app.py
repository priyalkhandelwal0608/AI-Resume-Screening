import os
import fitz  # PyMuPDF
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import ranker

app = Flask(__name__)
UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# PRE-DEFINED DEMO DATA
DEMO_JD = "Python Developer with Flask, SQL, and Machine Learning experience"
DEMO_RESUMES = [
    {
        "name": "Candidate_High_Match.pdf", 
        "text": "Expert Python Developer. Extensive experience in Flask web frameworks, SQL databases, and Machine Learning models."
    },
    {
        "name": "Candidate_Medium_Match.pdf", 
        "text": "Software Engineer with a focus on Python and Flask development. Some exposure to data tools."
    },
    {
        "name": "Candidate_Low_Match.pdf", 
        "text": "Graphic Designer and Creative Lead with 5 years experience in Adobe Suite and Branding."
    }
]

def extract_pdf_text(filepath):
    text = ""
    with fitz.open(filepath) as doc:
        for page in doc:
            text += page.get_text()
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        is_demo = request.form.get("is_demo") == "true"
        method = request.form.get("method", "tfidf")
        resumes_data = []

        if is_demo:
            jd = DEMO_JD
            resumes_data = list(DEMO_RESUMES)
        else:
            jd = request.form.get("job_description", "")
            manual_text = request.form.get("resumes", "").strip()
            if manual_text:
                resumes_data.append({"name": "Pasted Resume", "text": manual_text})

            files = request.files.getlist("resume_files")
            for file in files:
                if file and file.filename.endswith('.pdf'):
                    filename = secure_filename(file.filename)
                    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(path)
                    extracted_text = extract_pdf_text(path)
                    resumes_data.append({"name": filename, "text": extracted_text})
                    os.remove(path) 

        if not resumes_data or not jd:
            return render_template("index.html", error="Please provide input or use the Demo.")

        just_texts = [r['text'] for r in resumes_data]
        raw_results = ranker.rank(jd, just_texts, method=method)

        final_results = []
        temp_data = list(resumes_data) 
        for text, score in raw_results:
            try:
                idx = next(i for i, r in enumerate(temp_data) if r['text'] == text)
                matched_item = temp_data.pop(idx)
                
                # Convert decimal (0.8) to a whole number (80.0)
                display_score = round(float(score) * 100, 1)
                
                final_results.append({
                    "name": matched_item['name'], 
                    "score": display_score
                })
            except StopIteration:
                continue

        return render_template("index.html", results=final_results, method=method.upper(), demo_used=is_demo, demo_jd=DEMO_JD)

    return render_template("index.html")

if __name__ == "__main__":
    # CRITICAL: use_reloader=False stops the OneDrive infinite sync crash
    app.run(debug=True, use_reloader=False)