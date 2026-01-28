Fake Job Detection – Machine Learning Web Application
📌 Project Overview

Fake job postings are a growing problem that misleads job seekers and causes financial and emotional harm.
This project uses Machine Learning and Natural Language Processing (NLP) to analyze job descriptions and predict whether a job posting is Real or Fraudulent.
The trained ML model is deployed using a Flask web application to provide a simple and user-friendly interface.

🚀 Features

Detects fraudulent job postings based on job description text

Uses TF-IDF Vectorization for text feature extraction

Machine Learning model trained using Logistic Regression

Simple and interactive web interface built with Flask

Fast and accurate prediction results

🛠️ Technologies Used

Python

Flask

Scikit-learn

Pandas & NumPy

HTML & CSS

Git & GitHub

📂 Project Structure
Project/
│
├── app.py                 # Flask application
├── train_model.py         # Model training script
├── model.pkl              # Trained ML model
├── vectorizer.pkl         # TF-IDF vectorizer
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
├── templates/
│   └── index.html         # Frontend HTML file
└── static/
    └── style.css          # CSS styling

▶️ How to Run the Project

Clone the repository:

git clone https://github.com/your-username/fake-job-detection.git


Navigate to the project folder:

cd fake-job-detection


Install required libraries:

pip install -r requirements.txt


Run the Flask application:

python app.py


Open your browser and visit:

http://127.0.0.1:5000

📊 Dataset

The dataset contains job postings labeled as real or fraudulent, including job descriptions and related details.
Text data is preprocessed and transformed using TF-IDF before training the model.

🧠 Machine Learning Model

Algorithm Used: Logistic Regression

Text Processing: TF-IDF Vectorization

Output:

REAL JOB ✅

FRAUDULENT JOB ❌

🔮 Future Enhancements

Improve accuracy using advanced NLP models

Add more ML algorithms for comparison

Deploy the application on cloud platforms (Heroku / Render)

Enhance UI for better user experience

👤 Author

Khaja
Final Year Engineering Student
Interested in Machine Learning, Data Analytics, and Web Development

⭐ If you like this project, don’t forget to star the repository!

