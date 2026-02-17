from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    css_class = ""

    if request.method == "POST":
        text = request.form.get("job_description", "").strip()

        if not text:
            result = "Please enter a job description!"
            css_class = "fake"
        else:
            vec = vectorizer.transform([text])
            prediction = model.predict(vec)[0]

            print("MODEL OUTPUT:", prediction)

            # 🔥 UPDATED LOGIC HERE
            if prediction == 1:
                result = "✅ This job looks FAKE."
                css_class = "fake"
            else:
                result = "❌ This job looks REAL."
                css_class = "real"

    return render_template("index.html", result=result, css_class=css_class)


if __name__ == "__main__":
    app.run(debug=True)
