# 🧠 Depression Simple Android Application (API Model)

**DepressionSimple Android Application** is a mobile application designed to help assess and predict users' mental health conditions such as **Depression** and **Bipolar Disorder** using an AI model. The application connects to a Python-based backend through an API and was developed as a university project.

---

## 🎯 Key Features

- 🧠 **AI-Powered Prediction** – Predicts emotional conditions including Depression, Bipolar Type-1, and Type-2 using a trained ML model.
- 📝 **Detailed Questionnaire** – Users fill out a series of emotional and behavioral indicators.
- 📈 **Clear Results** – Returns a categorized result with an explanatory message.
- 🔗 **API Integration** – Connects the Kotlin mobile app to the backend through a secure POST request.

---

## ⚙️ System Architecture

| Component        | Description |
|------------------|-------------|
| **📱 Mobile Frontend (Kotlin)** | Developed using Kotlin for Android to collect user input and display prediction results. |
| **🧠 AI Backend (Python Flask)** | Provides REST API to serve predictions using a pre-trained Machine Learning model (`Depression_AI.pkl`). |
| **🔗 Communication** | Frontend and backend communicate via HTTP POST requests using a form-based data structure. |

---

## 🧰 Tech Stack

### 💻 Frontend (Mobile)
- Kotlin
- HTTP package for API communication
- Minimal, user-friendly UI design

### 🧠 Backend (AI Model API)
- Python
- Flask
- `joblib` to load the trained `.pkl` model
- Numpy
- RESTful API (deployed on port 3000)

---

## 🧪 How the AI Works

The backend receives 17 symptom-related inputs from the user, including:

- Sadness
- Euphoric behavior
- Suicidal thoughts
- Trouble sleeping
- Aggression
- Overthinking  
... and more.

The model then predicts the user's mental health state as follows:

| Prediction Code | Result |
|-----------------|---------|
| 0               | Normal  |
| 1               | Bipolar Type-1 |
| 2               | Bipolar Type-2 |
| 3               | Depression |
| Other           | Error |

---

## Required Form Data Parameters

| Parameter    | Description              |
|--------------|--------------------------|
| sadness      | Level of sadness         |
| euphoric     | Euphoric behavior        |
| exhausted    | Fatigue                  |
| sleep        | Sleep quality            |
| swing        | Mood swings              |
| suicidal     | Suicidal thoughts        |
| anorxia      | Appetite loss            |
| authority    | Authority conflict       |
| explanation  | Need for explanation     |
| aggressive   | Aggressiveness           |
| move_on     | Ability to move on       |
| break_down   | Breakdown tendency       |
| admit        | Acceptance               |
| overthinking | Overthinking             |
| sexual      | Sexual behavior changes  |
| concentration| Concentration ability    |
| optimisim    | Optimism level           |

---

## 🎓 Academic Context

This project was developed as a part of a **third-year university software project**, aiming to apply real AI models into mobile applications. It demonstrates:

- Application of Machine Learning to real-world scenarios  
- Mobile development using Kotlin  
- Backend integration using Flask  
- Psychological symptom analysis using AI  

---
