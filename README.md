# 👥 End-to-End HR Analytics Platform

A full-stack, data-driven HR Analytics application built using Streamlit that combines workforce analytics, employee management (CRUD), attrition insights, and an advanced AI-powered resume analyzer.

---

## 🚀 Overview

This project simulates a real-world HR system by integrating:

- 📊 Data Analytics
- 👥 Employee Management (CRUD)
- 🤖 Attrition Risk Insights
- 📄 AI Resume Analyzer (NLP + ATS Scoring)

It demonstrates how data science, business logic, and application development can be combined into a single interactive platform.

---

## ✨ Key Features

### 📊 Dashboard
- Real-time HR metrics:
  - Total Employees
  - Attrition Rate
  - Average Age
  - Salary Insights
- Interactive visualizations using Plotly

---

### 👥 Employee Management (CRUD)
- ➕ Create Employee  
- 📋 View Employee Records  
- ✏️ Update Employee Data  
- 🗑️ Delete Employee  
- Session-based state management

---

### 🔍 Smart Search & Filter
- Global keyword search across all fields  
- Department-based filtering  
- Handles mixed data types efficiently  

---

### 🤖 Attrition Analytics
- Department-wise attrition analysis  
- Age distribution insights  
- Salary vs Attrition visualization  
- Key risk factor identification  

---

### 📄 Resume Analyzer (Advanced NLP + ATS)

**Core Capabilities:**
- Extracts:
  - Name
  - Phone Number
  - Email
  - Education
  - College
  - Certifications
- Skill extraction using NLP (spaCy + PhraseMatcher)
- TF-IDF based keyword extraction
- Role matching using cosine similarity

**Advanced Features:**
- 🎯 Best Role Prediction
- 📊 Role-wise Match Score
- 🧠 ATS Score (0–100)
- 📉 Skill Gap Analysis
- 📌 Missing Skill Recommendations
- 📈 Role Fit Summary

---

### 📤 Export Feature
- Download full employee dataset as CSV

---

### 🎨 UI/UX
- Custom CSS styling
- Clean modern interface
- Gradient-based UI components
- Responsive layout

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **Language:** Python  
- **Data Handling:** pandas, numpy  
- **Visualization:** Plotly  
- **Machine Learning:** scikit-learn  
- **NLP:** spaCy, TF-IDF  
- **PDF Processing:** PyMuPDF  
- **Similarity Matching:** Cosine Similarity  
- **State Management:** Streamlit Session State  

---

## 📂 Dataset

- **IBM HR Analytics Employee Attrition Dataset**
- File: `HR-Employee-Attrition.csv`
- Place inside `data/` folder

---

## 📁 Project Structure

```

End-to-End-HR-Analytics-Platform/
│
├── app.py
├── data/
│   └── HR-Employee-Attrition.csv
├── requirements.txt
└── README.md

````

---

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Architab2005/End-to-End-HR-Analytics-Platform.git
cd End-to-End-HR-Analytics-Platform
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install spaCy model:

```bash
python -m spacy download en_core_web_sm
```

4. Add dataset:

* Place `HR-Employee-Attrition.csv` inside `data/`

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📊 Resume Analyzer Logic (Highlight for Recruiters)

* NLP-based skill extraction using **spaCy PhraseMatcher**
* TF-IDF for keyword importance scoring
* Cosine similarity for role matching
* Custom ATS scoring algorithm combining:

  * Role match score
  * Keywords
  * Skills
  * Resume length

---

## 📌 Key Highlights

* Combines **Data Analytics + NLP + ML + UI Development**
* Demonstrates **end-to-end product thinking**
* Real-world HR use case simulation
* Fully interactive and deployable

---

## ⚠️ Notes

* Resume Analyzer is heuristic + NLP based (not deep learning)
* Attrition prediction is analytical (can be extended to ML model)
* Designed for portfolio and learning purposes

---

## 👩‍💻 Author

**Archita B**
Final Year B.Tech Computer Science Engineering (2022–2026)
Data Science & Management Enthusiast

---

## 📄 License
This project is licensed under the MIT License.


