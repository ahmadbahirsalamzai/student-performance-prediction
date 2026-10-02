from pathlib import Path

from flask import Flask, render_template, request, send_file

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dataset")
def download_dataset():
    return send_file(
        Path(app.root_path) / "data" / "stud.csv",
        mimetype="text/csv",
        as_attachment=True,
        download_name="stud.csv",
    )


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html", results=None)

    data = CustomData(
        gender=request.form.get("gender"),
        race_ethnicity=request.form.get("ethnicity"),
        parental_level_of_education=request.form.get("parental_level_of_education"),
        lunch=request.form.get("lunch"),
        test_preparation_course=request.form.get("test_preparation_course"),
        reading_score=float(request.form.get("reading_score")),
        writing_score=float(request.form.get("writing_score")),
    )

    results = PredictPipeline().predict(data.get_data_as_data_frame())
    return render_template("home.html", results=round(results[0], 2))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
