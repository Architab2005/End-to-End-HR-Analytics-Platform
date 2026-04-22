import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import spacy
from spacy.matcher import PhraseMatcher
import fitz
import re
import os

# Page config
st.set_page_config(
    page_title="End-to-End HR Analytics Platform",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 2rem;}
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white; text-align: center;}
    .metric-value {font-size: 2.5rem; font-weight: bold; margin: 0;}
    .footer-section {position: fixed; bottom: 0; width: 100%; text-align: center; padding: 1rem; background: #f8f9fa; color: #6c757d; font-size: 0.9rem; z-index: 1000;}
</style>
""", unsafe_allow_html=True)

# All 35 columns
COLS = [
    'Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department', 
    'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount', 
    'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 
    'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 
    'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 
    'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 
    'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel', 
    'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 
    'YearsWithCurrManager'
]

@st.cache_data
def load_data():
    """Load CSV from data folder"""
    csv_path = "data/HR-Employee-Attrition.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Ensure all columns exist
        for col in COLS:
            if col not in df.columns:
                df[col] = df['Age'].iloc[0] if col in ['Age', 'EmployeeCount', 'StandardHours'] else 'No' if col in ['Over18'] else 0
        return df[COLS]
    else:
        st.error("❌ Please place HR-Employee-Attrition.csv in `data/` folder")
        st.stop()

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = load_data()
if 'model' not in st.session_state:
    st.session_state.model = None
if 'le_dict' not in st.session_state:
    st.session_state.le_dict = {}

df = st.session_state.df.copy()

# Sidebar - Select Module
st.sidebar.markdown("## 📋 Select Module")
selected_module = st.sidebar.selectbox(
    "",
    ["📊 Dashboard", "👥 Employees", "🔍 Search", "🤖 Predict", "📄 Resume Analyzer"]
)

# 1. DASHBOARD
if selected_module == "📊 Dashboard":
    st.markdown('<h1 class="main-header">📊 HR Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <h3>Total Employees</h3>
            <p class="metric-value">{len(df):,}</p>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        attrition_rate = (df['Attrition'] == 'Yes').sum() / len(df) * 100
        st.markdown(f'''
        <div class="metric-card">
            <h3>Attrition Rate</h3>
            <p class="metric-value">{attrition_rate:.1f}%</p>
        </div>
        ''', unsafe_allow_html=True)
    with col3:
        avg_salary = df['MonthlyIncome'].mean()
        st.markdown(f'''
        <div class="metric-card">
            <h3>Avg Salary</h3>
            <p class="metric-value">₹{avg_salary:,.0f}</p>
        </div>
        ''', unsafe_allow_html=True)
    with col4:
        avg_age = df['Age'].mean()
        st.markdown(f'''
        <div class="metric-card">
            <h3>Avg Age</h3>
            <p class="metric-value">{avg_age:.0f} yrs</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Charts - FIXED PIE CHART
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👥 Attrition Distribution")
        attrition_counts = df['Attrition'].value_counts().reset_index()
        attrition_counts.columns = ['Attrition', 'count']
        fig = px.pie(attrition_counts, values='count', names='Attrition', 
                    title="Attrition Distribution", hole=0.3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏢 Department Wise")
        fig = px.histogram(df, x='Department', title="Employees by Department")
        st.plotly_chart(fig, use_container_width=True)

# 2. EMPLOYEES (CRUD)
elif selected_module == "👥 Employees":
    st.markdown('<h1 class="main-header">👥 Employee Management (CRUD)</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["➕ Create", "📝 Update", "🗑️ Delete", "👀 View All"])
    
    with tab1:  # CREATE
        st.subheader("➕ Create New Employee")
        with st.form("create_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", 18, 65, 30)
                department = st.selectbox("Department", sorted(df['Department'].unique()))
                job_role = st.selectbox("Job Role", sorted(df['JobRole'].unique()))
                gender = st.selectbox("Gender", ['Male', 'Female'])
            with col2:
                monthly_income = st.number_input("Monthly Income (₹)", 1000, 10000000, 5000, 
                                               help="Max ₹1 Crore allowed")
                attrition = st.selectbox("Attrition", ['Yes', 'No'])
                job_level = st.selectbox("Job Level", sorted(df['JobLevel'].unique()))
            
            submitted = st.form_submit_button("➕ Add Employee", use_container_width=True)
            if submitted:
                # Create new row with default values
                new_row = pd.DataFrame([{col: (df[col].mode()[0] if col not in 
                    ['Age', 'Department', 'JobRole', 'Gender', 'Attrition', 'JobLevel', 'MonthlyIncome'] 
                    else {'Age': age, 'Department': department, 'JobRole': job_role, 
                          'Gender': gender, 'Attrition': attrition, 'JobLevel': job_level, 
                          'MonthlyIncome': monthly_income}[col]) for col in COLS}])
                
                st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
                st.session_state.df.to_csv("data/HR-Employee-Attrition.csv", index=False)
                st.success("✅ Employee created successfully!")
                st.rerun()
    
    with tab2:  # UPDATE
        st.subheader("📝 Update Employee")
        emp_id = st.number_input("Employee Number", df['EmployeeNumber'].min(), 
                                df['EmployeeNumber'].max() + 1)
        emp_data = df[df['EmployeeNumber'] == emp_id]
        
        if not emp_data.empty:
            with st.form("update_form"):
                col1, col2 = st.columns(2)
                with col1:
                    new_age = st.number_input("New Age", value=int(emp_data['Age'].iloc[0]))
                    new_dept = st.selectbox("New Department", sorted(df['Department'].unique()), 
                                          index=list(df['Department'].unique()).index(emp_data['Department'].iloc[0]))
                with col2:
                    new_salary = st.number_input("New Salary (₹)", value=int(emp_data['MonthlyIncome'].iloc[0]), 
                                               min_value=1000, max_value=10000000)
                    new_role = st.selectbox("New Job Role", sorted(df['JobRole'].unique()), 
                                          index=list(df['JobRole'].unique()).index(emp_data['JobRole'].iloc[0]))
                
                submitted = st.form_submit_button("💾 Update", use_container_width=True)
                if submitted:
                    idx = df[df['EmployeeNumber'] == emp_id].index[0]
                    st.session_state.df.loc[idx, ['Age', 'Department', 'MonthlyIncome', 'JobRole']] = [
                        new_age, new_dept, new_salary, new_role]
                    st.session_state.df.to_csv("data/HR-Employee-Attrition.csv", index=False)
                    st.success("✅ Employee updated!")
                    st.rerun()
        else:
            st.warning("⚠️ Employee not found!")
    
    with tab3:  # DELETE
        st.subheader("🗑️ Delete Employee")
        emp_to_delete = st.number_input("Employee Number to Delete", df['EmployeeNumber'].min(), 
                                      df['EmployeeNumber'].max())
        if st.button("🗑️ Confirm Delete", type="primary", use_container_width=True):
            before_count = len(st.session_state.df)
            st.session_state.df = st.session_state.df[st.session_state.df['EmployeeNumber'] != emp_to_delete]
            st.session_state.df.to_csv("data/HR-Employee-Attrition.csv", index=False)
            st.success(f"✅ Employee deleted! ({before_count} → {len(st.session_state.df)})")
            st.rerun()
    
    with tab4:  # VIEW
        st.subheader("👀 View All Employees (35 Columns)")
        st.dataframe(df[COLS], use_container_width=True, height=600)

# 3. SEARCH
elif selected_module == "🔍 Search":
    st.markdown('<h1 class="main-header">🔍 Smart Employee Search</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        search_col = st.selectbox("Search Column", COLS)
    with col2:
        search_value = st.text_input(f"Search in {search_col}")
    
    if st.button("🔍 Search Employees", use_container_width=True):
        if search_value:
            results = df[df[search_col].astype(str).str.contains(search_value, case=False, na=False)]
            if not results.empty:
                st.success(f"✅ Found {len(results)} matching employees")
                st.dataframe(results[COLS], use_container_width=True, height=500)
            else:
                st.warning("⚠️ No employees found matching your criteria")

# 4. PREDICT ATTRITION
elif selected_module == "🤖 Predict":
    st.markdown('<h1 class="main-header">🤖 AI Attrition Prediction</h1>', unsafe_allow_html=True)
    
    # Train model if not exists
    @st.cache_data
    def train_model():
        df_model = df.copy()
        le_dict = {}
        for col in df_model.select_dtypes(include=['object']).columns:
            if col != 'Attrition':
                le = LabelEncoder()
                df_model[col] = le.fit_transform(df_model[col].astype(str))
                le_dict[col] = le
        
        X = df_model.drop('Attrition', axis=1)
        y = (df_model['Attrition'] == 'Yes').astype(int)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model, le_dict
    
    if st.session_state.model is None:
        with st.spinner("Training AI model..."):
            st.session_state.model, st.session_state.le_dict = train_model()
        st.success("✅ Model trained!")
    
    st.subheader("🔮 Predict Single Employee Attrition")
    col1, col2 = st.columns(2)
    with col1:
        pred_age = st.slider("Age", 18, 65, 35)
        pred_dept = st.selectbox("Department", sorted(df['Department'].unique()))
        pred_role = st.selectbox("Job Role", sorted(df['JobRole'].unique()))
    with col2:
        pred_salary = st.slider("Monthly Income (₹)", 1000, 10000000, 5000)
        pred_overtime = st.selectbox("Overtime", ['Yes', 'No'])
    
    if st.button("🤖 Predict Risk", use_container_width=True):
        pred_data = df.iloc[0:1].copy()
        pred_data['Age'] = pred_age
        pred_data['MonthlyIncome'] = pred_salary
        pred_data['Department'] = pred_dept
        pred_data['JobRole'] = pred_role
        pred_data['OverTime'] = pred_overtime
        
        # Encode
        for col in pred_data.select_dtypes(include=['object']).columns:
            if col in st.session_state.le_dict:
                pred_data[col] = st.session_state.le_dict[col].transform(pred_data[col].astype(str))
        
        risk_prob = st.session_state.model.predict_proba(pred_data.drop('Attrition', axis=1))[0][1] * 100
        st.metric("🎯 Attrition Risk", f"{risk_prob:.1f}%", delta=f"{risk_prob:.1f}%")

# 5. RESUME ANALYZER
elif selected_module == "📄 Resume Analyzer":
    st.markdown('<h1 class="main-header">📄 AI Resume Analyzer & ATS</h1>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=['pdf'])
    
    if uploaded_file is not None:
        # Extract text
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        
        # Job roles and keywords from dataset
        roles_keywords = {
            'Sales Executive': ['sales', 'client', 'customer', 'revenue', 'account'],
            'Research Scientist': ['research', 'data', 'analysis', 'experiment', 'model'],
            'Laboratory Technician': ['lab', 'technician', 'experiment', 'sample', 'test'],
            'Manager': ['manager', 'team', 'leadership', 'supervise', 'project'],
            'Sales Representative': ['sales', 'representative', 'client', 'lead'],
            'Research Director': ['research', 'director', 'lead', 'strategy']
        }
        
        # Calculate ATS scores
        doc_lower = text.lower()
        ats_scores = {}
        for role, keywords in roles_keywords.items():
            matches = sum(1 for kw in keywords if kw in doc_lower)
            score = (matches / len(keywords)) * 100
            ats_scores[role] = min(score, 100)
        
        best_fit = max(ats_scores, key=ats_scores.get)
        
        # Extract candidate details
        name_match = re.search(r'([A-Z][a-z]+ [A-Z][a-z]+)', text)
        name = name_match.group(1) if name_match else "Not Found"
        
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        email = email_match.group() if email_match else "Not Found"
        
        phone_match = re.search(r'[\d\s\-\+\(\)]{10,15}', text)
        phone = phone_match.group() if phone_match else "Not Found"
        
        # Experience
        years_match = re.search(r'(\d+)\s*(?:years?|yrs?|experience)', text, re.IGNORECASE)
        experience = "Fresher" if not years_match else f"{years_match.group(1)} years"
        
        # Courses & Certificates
        courses = ["B.Tech"] if "b.tech" in text.lower() or "btech" in text.lower() else []
        certificates = "Yes" if any(word in text.lower() for word in ["certificate", "certified"]) else "No"
        
        # Display Results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 ATS Score", f"{ats_scores[best_fit]:.0f}%")
        with col2:
            st.metric("💼 Best Fit", best_fit)
        with col3:
            st.metric("📚 Experience", experience)
        
        st.subheader("📊 All Role Fit Scores")
        fit_df = pd.DataFrame([
            {'Role': role, 'Fit Score %': score} for role, score in ats_scores.items()
        ]).sort_values('Fit Score %', ascending=False)
        st.bar_chart(fit_df.set_index('Role'))
        
        st.subheader("👤 Candidate Profile")
        st.json({
            "Name": name,
            "Email": email,
            "Phone": phone,
            "Experience": experience,
            "Courses": courses,
            "Certificates": certificates,
            "Missing Skills": [r for r, s in ats_scores.items() if s < 50]
        })

# Footer
st.markdown("""
<div class="footer-section">
    Built by Archita B | Final Year B.Tech CSE'26 | Data Science & Management Enthusiast
</div>
""", unsafe_allow_html=True)