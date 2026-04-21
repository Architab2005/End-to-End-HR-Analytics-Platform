import streamlit as st
import pandas as pd
import plotly.express as px
import fitz
import re
import numpy as np
import spacy
from spacy.matcher import PhraseMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="End-to-End HR Analytics Platform",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .hero-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 35px 50px;
        border-radius: 25px;
        margin: 20px 0 40px 0;
        color: white;
        text-align: center;
        box-shadow: 0 12px 40px rgba(102,126,234,0.4);
        font-size: 3em;
        font-weight: 800;
    }
    .export-bar {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 25px;
        border-radius: 20px;
        margin: 40px 0;
        text-align: center;
        box-shadow: 0 8px 25px rgba(240,147,251,0.35);
    }
    .footer-section {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 30px;
        border-radius: 25px;
        margin-top: 60px;
        color: white;
        text-align: center;
        box-shadow: 0 -10px 40px rgba(79,172,254,0.4);
        font-size: 1.1em;
        font-weight: 600;
    }
    .page-title {
        color: #2c3e50;
        font-size: 2.2em;
        margin: 40px 0 25px 0;
        text-align: center;
        font-weight: 700;
    }
    .crud-tabs {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .skill-badge {
        background: linear-gradient(45deg, #28a745, #218838);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        margin: 2px;
        display: inline-block;
    }
    .missing-badge {
        background: linear-gradient(45deg, #dc3545, #b02a37);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        margin: 2px;
        display: inline-block;
    }
    .box {
        background: #f8f9fa;
        border-left: 5px solid #764ba2;
        padding: 15px;
        border-radius: 10px;
        margin: 12px 0;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
</style>
""", unsafe_allow_html=True)


st.sidebar.title("👥 HR Analytics Platform")
page = st.sidebar.selectbox(
    "Select Module",
    ["📊 Dashboard", "👥 Employees", "🔍 Search", "🤖 Predict", "📄 Resume Analyzer"],
    index=0
)


@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp


nlp = load_nlp()


@st.cache_data
def load_data():
    return pd.read_csv("data/HR-Employee-Attrition.csv")


if "df" not in st.session_state:
    st.session_state.df = load_data()
    st.session_state.employees_df = st.session_state.df[
        ["Age", "Department", "MonthlyIncome", "OverTime", "Attrition", "DistanceFromHome"]
    ].copy()
    st.session_state.employees_df["ID"] = range(1, len(st.session_state.employees_df) + 1)
    st.session_state.next_id = len(st.session_state.employees_df) + 1


st.markdown('<div class="hero-title">End-to-End HR-Analytics Platform</div>', unsafe_allow_html=True)
st.markdown("─" * 80)
st.markdown(f'<h2 class="page-title">{page}</h2>', unsafe_allow_html=True)


ROLE_SKILLS = {
    "Data Analyst": ["Python", "SQL", "Excel", "Power BI", "Tableau", "Statistics", "Data Visualization", "Pandas", "ETL", "Business Intelligence", "Machine Learning", "Reporting"],
    "Human Resources": ["Recruitment", "Talent Acquisition", "Employee Relations", "Onboarding", "Payroll", "Compliance", "Training", "Performance Management", "HR Analytics", "HRIS"],
    "Manager": ["Leadership", "Team Management", "Project Management", "Strategy", "Budgeting", "Stakeholder Management", "Performance Reviews", "Coaching", "Decision Making"],
    "Sales Executive": ["Sales", "Customer Relationship Management", "Negotiation", "Lead Generation", "Account Management", "Pipeline Management", "B2B Sales", "Closing Deals", "Revenue Growth"],
    "Research Scientist": ["Python", "Statistics", "Machine Learning", "Data Analysis", "Hypothesis Testing", "Experiment Design", "Research", "R"],
    "Laboratory Technician": ["Laboratory Techniques", "Quality Control", "Sampling", "Calibration", "Documentation", "GMP", "GLP", "Instrumentation"],
    "Research Director": ["Research Leadership", "Innovation", "R&D Strategy", "Grant Writing", "Patents", "Technology Transfer", "KPI Management", "Portfolio Management"],
    "Manufacturing Director": ["Operations Management", "Manufacturing", "Production Planning", "Quality Management", "Lean", "Six Sigma", "Supply Chain", "Continuous Improvement"],
    "Healthcare Representative": ["Healthcare Sales", "Clinical Knowledge", "Medical Devices", "Pharmaceuticals", "Regulatory Affairs", "Healthcare Compliance", "Communication", "Patient Support"],
    "Sales Representative": ["Sales", "CRM", "Cold Calling", "Lead Generation", "Upselling", "Cross Selling", "Sales Funnel", "Objection Handling", "Territory Management"]
}


ROLE_KEYWORDS = {
    "Data Analyst": ["Python", "SQL", "Excel", "Power BI", "Tableau", "Data Visualization", "Pandas", "Statistics", "ETL", "Reporting"],
    "Human Resources": ["HR", "Recruitment", "Onboarding", "Employee Relations", "Payroll", "Compliance", "Training", "Performance Management", "Talent Acquisition", "HR Analytics"],
    "Manager": ["Management", "Leadership", "Team", "Project Management", "Strategy", "Budgeting", "Supervision", "Coaching", "Stakeholder Management", "Performance"],
    "Sales Executive": ["Sales", "Customer", "Client", "Revenue", "Negotiation", "Account", "Pipeline", "Lead Generation", "Closing", "Relationship Management"],
    "Research Scientist": ["Research", "Python", "Statistics", "Analysis", "Machine Learning", "Hypothesis", "Experiment", "Data Modeling", "Publication", "R"],
    "Laboratory Technician": ["Laboratory", "Testing", "Quality", "Sample", "Calibration", "Protocol", "Documentation", "GMP", "GLP", "Instrumentation"],
    "Research Director": ["Research", "Leadership", "Innovation", "Strategy", "R&D", "Grant Writing", "Patents", "Portfolio", "Technology Transfer", "KPI"],
    "Manufacturing Director": ["Operations", "Manufacturing", "Production", "Quality", "Supply Chain", "Lean", "Six Sigma", "Capacity Planning", "Inventory", "Continuous Improvement"],
    "Healthcare Representative": ["Healthcare", "Medical", "Clinical", "Patient", "Pharmaceutical", "Compliance", "Sales", "Communication", "Regulatory", "Support"],
    "Sales Representative": ["Sales", "CRM", "Cold Calling", "Lead Generation", "Upselling", "Cross Selling", "Territory", "Sales Funnel", "Objection Handling", "Demo"]
}


CORE_SKILLS = ["Python", "SQL", "Management", "Excel", "Tableau", "Power BI", "Statistics", "Leadership", "C++", "Java", "HTML", "CSS", "Machine Learning", "Business Analysis"]
ALL_SKILLS = sorted(list(set(sum(ROLE_SKILLS.values(), []) + CORE_SKILLS)))


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\\s\\+\\#]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def extract_pdf_text(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def extract_phone_number(text):
    patterns = [
        r"\\b\\d{10}\\b",
        r"\\b\\d{3}[-.\\s]?\\d{3}[-.\\s]?\\d{4}\\b",
        r"\\b\\+?\\d{1,3}[-.\\s]?\\d{10}\\b"
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            digits = re.sub(r"\\D", "", m.group())
            if len(digits) >= 10:
                return digits[-10:]
    return ""


def extract_email(text):
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}", text)
    return m.group() if m else ""


def extract_name(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    first_lines = lines[:6]
    blacklist = {"resume", "curriculum vitae", "cv", "email", "mobile", "phone", "address"}
    for line in first_lines:
        low = line.lower()
        if any(b in low for b in blacklist):
            continue
        if len(line.split()) <= 5 and re.fullmatch(r"[A-Za-z .'-]+", line):
            return line.strip()
    return ""


def classify_education(text):
    t = text.lower()
    edu_keywords = [
        ("B.Tech Computer Science and Engineering", ["b.tech computer science and engineering", "btech computer science and engineering", "b.e computer science and engineering"]),
        ("B.Tech Computer Science", ["b.tech computer science", "btech computer science", "b.e computer science"]),
        ("B.Tech", ["b.tech", "btech", "b.e", "be"]),
        ("M.Tech", ["m.tech", "mtech"]),
        ("MBA", ["mba", "master of business administration"]),
        ("B.Sc", ["b.sc", "bsc", "bachelor of science"]),
        ("M.Sc", ["m.sc", "msc", "master of science"]),
        ("Diploma", ["diploma"]),
        ("12th", ["12th", "higher secondary", "hsc"]),
        ("10th", ["10th", "sslc", "secondary school"])
    ]
    found = []
    for label, pats in edu_keywords:
        if any(p in t for p in pats):
            found.append(label)
    return list(dict.fromkeys(found))


def extract_college(text):
    if "srm institute of science and technology" in text.lower():
        return "SRM Institute of Science and Technology"
    m = re.search(r"([A-Z][A-Za-z&\\-\\.\\s]{3,80}(institute|university|college)[A-Za-z&\\-\\.\\s]{0,60})", text, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def extract_certifications(text):
    cert_keywords = ["certification", "certificate", "certified", "coursera", "udemy", "nptel", "aws", "azure", "google cloud", "pmp", "scrum", "six sigma", "machine learning", "data science", "python", "sql"]
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    certs = []
    for line in lines:
        low = line.lower()
        if any(k in low for k in cert_keywords) and len(line) > 3:
            certs.append(line)
    return list(dict.fromkeys(certs))[:10]


def extract_skills(text):
    doc = nlp(text)
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in ALL_SKILLS]
    matcher.add("SKILLS", patterns)
    found = []
    for _, start, end in matcher(doc):
        span = doc[start:end].text.strip()
        if span not in found:
            found.append(span)
    return found


def extract_keywords_tfidf(text, top_n=12):
    doc = nlp(text)
    tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
            lemma = token.lemma_.lower().strip() if token.lemma_ != "-PRON-" else token.text.lower().strip()
            if lemma and len(lemma) > 2:
                tokens.append(lemma)
    if not tokens:
        return []
    chunks = [" ".join(tokens)]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(chunks)
    terms = vectorizer.get_feature_names_out()
    scores = np.asarray(tfidf.mean(axis=0)).ravel()
    top_idx = scores.argsort()[::-1][:top_n]
    return [terms[i] for i in top_idx if scores[i] > 0]


def tfidf_role_scores(resume_text, role_map):
    docs = [resume_text] + [" ".join(v) for v in role_map.values()]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(docs)
    sim = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    result = [(role, round(float(sim[i] * 100), 2)) for i, role in enumerate(role_map.keys())]
    return sorted(result, key=lambda x: x[1], reverse=True)


def ats_score(text, best_role_score, extracted_keywords, found_core, found_skills):
    keyword_bonus = min(len(extracted_keywords) * 1.5, 20)
    core_bonus = (len(found_core) / max(len(CORE_SKILLS), 1)) * 20
    skill_bonus = min(len(found_skills) * 2, 20)
    length_bonus = min(len(text.split()) / 400 * 10, 10)
    score = 0.4 * best_role_score + keyword_bonus + core_bonus + skill_bonus + length_bonus
    return round(min(score, 100), 1)


def ats_feedback(best_role, text):
    role_skills = ROLE_SKILLS[best_role]
    found = [s for s in role_skills if s.lower() in text.lower()]
    missing = [s for s in role_skills if s.lower() not in text.lower()]
    return found, missing[:5]


def score_label(score):
    if score >= 85:
        return "Excellent ATS fit"
    if score >= 70:
        return "Strong match"
    if score >= 50:
        return "Moderate match"
    return "Weak match"


def role_gap_summary(resume_text):
    rows = []
    text = clean_text(resume_text)
    for role, skills in ROLE_SKILLS.items():
        found = [s for s in skills if s.lower() in text]
        missing = [s for s in skills if s.lower() not in text]
        pct = round((len(found) / len(skills)) * 100, 1)
        rows.append({
            "Role": role,
            "Role Fit %": pct,
            "Matched Skills": len(found),
            "Missing Skills": ", ".join(missing[:5])
        })
    return pd.DataFrame(rows).sort_values("Role Fit %", ascending=False).reset_index(drop=True)


if page == "📊 Dashboard":
    df = st.session_state.df
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Employees", f"{len(df):,}")
    col2.metric("📉 Attrition Rate", f"{(df['Attrition'].value_counts().get('Yes', 0) / len(df) * 100):.1f}%")
    col3.metric("👴 Average Age", f"{df['Age'].mean():.0f}")
    col4.metric("💰 Avg Salary", f"${df['MonthlyIncome'].mean():,.0f}")


    c1, c2 = st.columns(2)
    with c1:
        dept_counts = df["Department"].value_counts().reset_index()
        dept_counts.columns = ["Department", "Count"]
        fig = px.bar(dept_counts, x="Department", y="Count", title="Employees by Department")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        attr = df["Attrition"].value_counts().reset_index()
        attr.columns = ["Attrition", "Count"]
        fig = px.pie(attr, names="Attrition", values="Count", title="Attrition Distribution")
        st.plotly_chart(fig, use_container_width=True)


elif page == "👥 Employees":
    st.markdown('<div class="crud-tabs">', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["📋 View", "➕ Create", "✏️ Update", "🗑️ Delete"])


    with tab1:
        st.subheader("Employee Directory")
        st.dataframe(st.session_state.employees_df, use_container_width=True)


    with tab2:
        st.subheader("➕ Create New Employee")
        with st.form("create_form"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age", 18, 65, 30)
                distance = st.number_input("Distance from Home", 1, 30, 5)
            with c2:
                # Fixed: Changed max salary from 20000 to 10000000 (10,00,00,000)
                salary = st.number_input("Monthly Income", 1000, 10000000, 6500, format="%d")
                overtime = st.selectbox("OverTime", ["Yes", "No"])
            department = st.selectbox("Department", st.session_state.df["Department"].unique())
            attrition = st.selectbox("Attrition", ["Yes", "No"])
            if st.form_submit_button("✅ Create Employee"):
                new_emp = pd.DataFrame([{
                    "ID": st.session_state.next_id,
                    "Age": age,
                    "DistanceFromHome": distance,
                    "MonthlyIncome": salary,
                    "OverTime": overtime,
                    "Department": department,
                    "Attrition": attrition
                }])
                st.session_state.employees_df = pd.concat([st.session_state.employees_df, new_emp], ignore_index=True)
                st.session_state.next_id += 1
                st.success(f"✅ Employee ID {new_emp.iloc[0]['ID']} created!")
                st.rerun()


    with tab3:
        st.subheader("✏️ Update Employee")
        if len(st.session_state.employees_df) > 0:
            idx = st.number_input("Row number to update", 0, len(st.session_state.employees_df) - 1, 0)
            emp = st.session_state.employees_df.iloc[int(idx)]
            with st.form("update_form"):
                c1, c2 = st.columns(2)
                with c1:
                    new_age = st.number_input("New Age", value=int(emp["Age"]))
                    new_distance = st.number_input("New Distance from Home", value=int(emp["DistanceFromHome"]))
                with c2:
                    # Fixed: Changed max salary to 10000000
                    new_salary = st.number_input("New Monthly Income", value=int(emp["MonthlyIncome"]), max_value=10000000, format="%d")
                    new_overtime = st.selectbox("New OverTime", ["Yes", "No"], index=0 if emp["OverTime"] == "Yes" else 1)
                new_department = st.selectbox("New Department", st.session_state.df["Department"].unique(), index=list(st.session_state.df["Department"].unique()).index(emp["Department"]))
                new_attrition = st.selectbox("New Attrition", ["Yes", "No"], index=0 if emp["Attrition"] == "Yes" else 1)
                if st.form_submit_button("💾 Update Employee"):
                    st.session_state.employees_df.loc[int(idx)] = [emp["ID"], new_age, new_department, new_salary, new_overtime, new_attrition, new_distance]
                    st.success(f"✅ Employee ID {emp['ID']} updated!")
                    st.rerun()


    with tab4:
        st.subheader("🗑️ Delete Employee")
        if len(st.session_state.employees_df) > 0:
            del_idx = st.number_input("Row number to delete", 0, len(st.session_state.employees_df) - 1, 0)
            st.write(st.session_state.employees_df.iloc[int(del_idx)])
            if st.button("🗑️ Confirm Delete", type="primary"):
                deleted_id = st.session_state.employees_df.iloc[int(del_idx)]["ID"]
                st.session_state.employees_df = st.session_state.employees_df.drop(int(del_idx)).reset_index(drop=True)
                st.success(f"✅ Employee ID {deleted_id} deleted!")
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


elif page == "🔍 Search":
    st.subheader("🔍 Smart Search")
    c1, c2 = st.columns(2)
    with c1:
        search = st.text_input("Search any value")
    with c2:
        dept_filter = st.selectbox("Filter by Department", ["All"] + sorted(st.session_state.df["Department"].unique()))
    
    filtered = st.session_state.employees_df.copy()
    
    # Fixed: Proper search implementation that handles mixed data types
    if search:
        # Convert all columns to string for searching, then check if any column contains search term
        mask = filtered.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False), axis=0).any(axis=1)
        filtered = filtered[mask]
    
    if dept_filter != "All":
        filtered = filtered[filtered["Department"] == dept_filter]
    
    st.dataframe(filtered, use_container_width=True)


elif page == "🤖 Predict":
    st.subheader("🤖 Attrition Risk Analytics")
    df = st.session_state.df
    c1, c2 = st.columns(2)
    with c1:
        dept_attrition = df.groupby(["Department", "Attrition"]).size().unstack(fill_value=0)
        fig1 = px.bar(dept_attrition, title="Attrition by Department", barmode="group")
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.histogram(df, x="Age", color="Attrition", nbins=20, title="Age Distribution by Attrition")
        st.plotly_chart(fig2, use_container_width=True)


    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.scatter(df, x="MonthlyIncome", y="Age", color="Attrition", title="Salary vs Age by Attrition")
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        risk_data = pd.DataFrame({"Factor": ["Overtime", "Distance > 10", "Low Salary", "Young Age"], "HighRisk": [35, 28, 22, 15]})
        fig4 = px.pie(risk_data, values="HighRisk", names="Factor", title="Top Attrition Risk Factors")
        st.plotly_chart(fig4, use_container_width=True)


elif page == "📄 Resume Analyzer":
    st.markdown("### 📄 Resume Analyzer")
    st.info("Upload one resume PDF and extract name, phone, education, college, skills, and certifications.")
    uploaded = st.file_uploader("Upload Resume (PDF)", type="pdf")


    if uploaded:
        raw_text = extract_pdf_text(uploaded)
        text = clean_text(raw_text)


        name = extract_name(raw_text)
        phone = extract_phone_number(raw_text)
        email = extract_email(raw_text)
        education = classify_education(raw_text)
        college = extract_college(raw_text)
        certifications = extract_certifications(raw_text)


        role_scores = tfidf_role_scores(text, ROLE_KEYWORDS)
        best_role, best_role_score = role_scores[0]


        extracted_skills = extract_skills(raw_text)
        extracted_keywords = extract_keywords_tfidf(raw_text, top_n=15)
        found_core = [s for s in CORE_SKILLS if s.lower() in text]
        score = ats_score(text, best_role_score, extracted_keywords, found_core, extracted_skills)
        found_role_skills, missing_role_skills = ats_feedback(best_role, raw_text)
        gap_df = role_gap_summary(raw_text)


        c1, c2, c3 = st.columns(3)
        c1.metric("Resume Score", f"{score}/100")
        c2.metric("Best Role Match", best_role, f"{best_role_score:.1f}%")
        c3.metric("ATS Status", score_label(score))


        st.subheader("Personal Details")
        info_df = pd.DataFrame([{
            "Name": name,
            "Phone Number": phone,
            "Email": email,
            "College Name": college
        }])
        st.dataframe(info_df, use_container_width=True)


        st.subheader("Education")
        st.write(", ".join(education) if education else "No education keywords found.")


        st.subheader("Certifications")
        st.write(", ".join(certifications) if certifications else "No certifications found.")


        st.subheader("Role Ranking for Your Resume")
        rank_df = pd.DataFrame(role_scores, columns=["Role", "Match Score"])
        fig = px.bar(rank_df, x="Role", y="Match Score", text="Match Score", title="All Role Matches")
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)


        st.subheader("Extracted Skills")
        if extracted_skills:
            st.markdown(" ".join([f'<span class="skill-badge">{s}</span>' for s in extracted_skills]), unsafe_allow_html=True)
        else:
            st.warning("No skills detected.")


        st.subheader("NLP Keyword Extraction")
        if extracted_keywords:
            st.markdown(" ".join([f'<span class="skill-badge">{kw.title()}</span>' for kw in extracted_keywords]), unsafe_allow_html=True)


        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Skills Found")
            if found_core:
                st.markdown(" ".join([f'<span class="skill-badge">{s}</span>' for s in found_core]), unsafe_allow_html=True)
            else:
                st.info("No core skills found.")
        with c2:
            st.subheader("Skills to Add")
            if missing_role_skills:
                st.markdown(" ".join([f'<span class="missing-badge">{s}</span>' for s in missing_role_skills]), unsafe_allow_html=True)


        st.subheader(f"ATS Feedback for {best_role}")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Skills Present**")
            if found_role_skills:
                st.markdown(" ".join([f'<span class="skill-badge">{s}</span>' for s in found_role_skills]), unsafe_allow_html=True)
        with col_b:
            st.markdown("**Skills Missing**")
            if missing_role_skills:
                st.markdown(" ".join([f'<span class="missing-badge">{s}</span>' for s in missing_role_skills]), unsafe_allow_html=True)


        st.subheader("Role Fit Summary")
        st.dataframe(gap_df, use_container_width=True)


        st.markdown("### Resume Feedback")
        st.success(f"{score_label(score)}. Resume score: {score}/100.")
        st.markdown("""
        <div class="box">
            <b>Recommendations:</b><br>
            • Add role-specific keywords for the target role.<br>
            • Use metrics and action verbs like led, built, optimized, increased.<br>
            • Mention tools, certifications, and measurable impact.<br>
            • Include strong skills such as Python, SQL, C++, Java, HTML, CSS, Business Analysis, and Machine Learning where relevant.<br>
        </div>
        """, unsafe_allow_html=True)


    else:
        st.markdown("""
        Upload a PDF resume to get:
        - Resume score out of 100
        - Name, phone number, email, college, education, certifications
        - Skill extraction
        - Ranking for all roles from one resume
        - ATS feedback
        - Missing skills suggestions
        """)


st.markdown('<div class="export-bar">', unsafe_allow_html=True)
if st.button("💾 Download All Data"):
    csv = st.session_state.employees_df.to_csv(index=False)
    st.download_button("📥 CSV", csv, "hr_data.csv", "text/csv")
st.markdown('</div>', unsafe_allow_html=True)


st.markdown("""
<div class="footer-section">
    Built by Archita B | Final Year B.Tech CSE'26 | Data Science & Management Enthusiast
</div>
""", unsafe_allow_html=True)