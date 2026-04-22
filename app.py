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
import json

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
    .success-box {background-color: #d4edda; padding: 1rem; border-radius: 10px; border-left: 5px solid #28a745;}
    .error-box {background-color: #f8d7da; padding: 1rem; border-radius: 10px; border-left: 5px solid #dc3545;}
    .footer-section {position: fixed; bottom: 0; width: 100%; text-align: center; padding: 1rem; background: #f8f9fa; color: #6c757d; font-size: 0.9rem; z-index: 1000;}
</style>
""", unsafe_allow_html=True)

# All 35 columns from IBM dataset
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
        try:
            df = pd.read_csv(csv_path)
            # Ensure all 35 columns exist with defaults
            for col in COLS:
                if col not in df.columns:
                    if col in ['Age', 'EmployeeCount', 'StandardHours']:
                        df[col] = 30  # Default age-like values
                    elif col == 'Over18':
                        df[col] = 'Y'
                    else:
                        df[col] = 0
            return df[COLS].copy()
        except Exception as e:
            st.error(f"❌ Error loading CSV: {str(e)}")
            st.stop()
    else:
        # Create sample data if no CSV
        st.warning("⚠️ No CSV found. Creating sample data...")
        sample_data = {
            'Age': [29, 35, 28, 42],
            'Attrition': ['No', 'No', 'Yes', 'No'],
            'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Travel_Rarely', 'Non-Travel'],
            'DailyRate': [500, 800, 600, 1200],
            'Department': ['Sales', 'Research & Development', 'Sales', 'Human Resources'],
            'DistanceFromHome': [1, 10, 2, 5],
            'Education': [3, 4, 2, 3],
            'EducationField': ['Life Sciences', 'Technical Degree', 'Medical', 'Human Resources'],
            'EmployeeCount': [1, 1, 1, 1],
            'EmployeeNumber': [1, 2, 3, 4],
            'EnvironmentSatisfaction': [3, 4, 2, 3],
            'Gender': ['Male', 'Female', 'Male', 'Female'],
            'HourlyRate': [65, 75, 60, 70],
            'JobInvolvement': [3, 4, 2, 3],
            'JobLevel': [2, 3, 1, 2],
            'JobRole': ['Sales Executive', 'Research Scientist', 'Sales Representative', 'HR Analyst'],
            'JobSatisfaction': [3, 4, 2, 3],
            'MaritalStatus': ['Single', 'Married', 'Single', 'Divorced'],
            'MonthlyIncome': [5000, 8000, 4500, 6500],
            'MonthlyRate': [12000, 18000, 11000, 15000],
            'NumCompaniesWorked': [1, 2, 0, 3],
            'Over18': ['Y', 'Y', 'Y', 'Y'],
            'OverTime': ['No', 'Yes', 'No', 'No'],
            'PercentSalaryHike': [10, 15, 12, 11],
            'PerformanceRating': [3, 4, 3, 3],
            'RelationshipSatisfaction': [3, 4, 2, 3],
            'StandardHours': [40, 40, 40, 40],
            'StockOptionLevel': [0, 1, 0, 0],
            'TotalWorkingYears': [5, 8, 3, 6],
            'TrainingTimesLastYear': [2, 3, 1, 2],
            'WorkLifeBalance': [3, 3, 2, 3],
            'YearsAtCompany': [3, 5, 2, 4],
            'YearsInCurrentRole': [2, 3, 1, 2],
            'YearsSinceLastPromotion': [1, 2, 0, 1],
            'YearsWithCurrManager': [2, 3, 1, 2]
        }
        df = pd.DataFrame(sample_data)
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/HR-Employee-Attrition.csv", index=False)
        return df

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.df = load_data()
    st.session_state.employees_df = st.session_state.df[
        ["Age", "Department", "MonthlyIncome", "OverTime", "Attrition", "DistanceFromHome"]
    ].copy()
    st.session_state.employees_df["ID"] = range(1, len(st.session_state.employees_df) + 1)
    st.session_state.next_id = int(st.session_state.df['EmployeeNumber'].max() + 1)
    st.session_state.model = None
    st.session_state.le_dict = {}
    st.session_state.initialized = True

df = st.session_state.df.copy()

def save_data():
    """Save data to CSV with confirmation"""
    try:
        os.makedirs("data", exist_ok=True)
        csv_path = "data/HR-Employee-Attrition.csv"
        st.session_state.df.to_csv(csv_path, index=False)
        
        # Verify save
        if os.path.exists(csv_path):
            saved_df = pd.read_csv(csv_path)
            if len(saved_df) == len(st.session_state.df):
                return True, f"✅ Saved {len(saved_df)} records to {csv_path}"
            else:
                return False, "❌ Save failed - record count mismatch"
        return False, "❌ CSV file not found after save"
    except PermissionError:
        return False, "❌ Permission denied! Run as administrator or fix folder permissions"
    except Exception as e:
        return False, f"❌ Save error: {str(e)}"

# Sidebar
st.sidebar.markdown("## 📋 Select Module")
selected_module = st.sidebar.selectbox("", ["📊 Dashboard", "👥 Employees", "🔍 Search", "🤖 Predict", "📄 Resume Analyzer"])

# DASHBOARD
if selected_module == "📊 Dashboard":
    st.markdown('<h1 class="main-header">📊 HR Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><h3>Total Employees</h3><p class="metric-value">{len(df):,}</p></div>', unsafe_allow_html=True)
    with col2:
        attrition_rate = (df['Attrition'] == 'Yes').sum() / len(df) * 100
        st.markdown(f'<div class="metric-card"><h3>Attrition Rate</h3><p class="metric-value">{attrition_rate:.1f}%</p></div>', unsafe_allow_html=True)
    with col3:
        avg_salary = df['MonthlyIncome'].mean()
        st.markdown(f'<div class="metric-card"><h3>Avg Salary</h3><p class="metric-value">₹{avg_salary:,.0f}</p></div>', unsafe_allow_html=True)
    with col4:
        avg_age = df['Age'].mean()
        st.markdown(f'<div class="metric-card"><h3>Avg Age</h3><p class="metric-value">{avg_age:.0f} yrs</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👥 Attrition by Department")
        fig = px.histogram(df, x='Department', color='Attrition', title="Attrition Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("💰 Salary by Job Role")
        fig = px.box(df, x='JobRole', y='MonthlyIncome', title="Salary Distribution")
        st.plotly_chart(fig, use_container_width=True)

# EMPLOYEES CRUD
elif selected_module == "👥 Employees":
    st.markdown('<h1 class="main-header">👥 Employee Management (CRUD)</h1>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["➕ Create", "📝 Update", "🗑️ Delete", "👀 View All"])
    
    with tab1:
        st.subheader("➕ Create New Employee")
        with st.form("create_form"):
            col1, col2, col3, col4 = st.columns(4)
            with col1: age = st.number_input("Age", 18, 65, 30)
            with col2: department = st.selectbox("Department", sorted(df['Department'].unique()))
            with col3: distance = st.number_input("DistanceFromHome", 1, 30, 5)
            with col4: education = st.number_input("Education", 1, 5, 3)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: education_field = st.selectbox("EducationField", sorted(df['EducationField'].unique()))
            with col2: job_role = st.selectbox("JobRole", sorted(df['JobRole'].unique()))
            with col3: gender = st.selectbox("Gender", ["Male", "Female"])
            with col4: marital_status = st.selectbox("MaritalStatus", sorted(df['MaritalStatus'].unique()))
            
            col1, col2 = st.columns(2)
            with col1: monthly_income = st.number_input("MonthlyIncome", 1000, 20000, 6000, format="%d")
            with col2: overtime = st.selectbox("OverTime", ["Yes", "No"])
            
            submitted = st.form_submit_button("✅ Create Employee", use_container_width=True)
            if submitted:
                new_employee = {
                    'Age': age, 'Attrition': 'No', 'BusinessTravel': 'Travel_Rarely',
                    'DailyRate': 800, 'Department': department, 'DistanceFromHome': distance,
                    'Education': education, 'EducationField': education_field,
                    'EmployeeCount': 1, 'EmployeeNumber': st.session_state.next_id,
                    'EnvironmentSatisfaction': 3, 'Gender': gender,
                    'HourlyRate': 65, 'JobInvolvement': 3, 'JobLevel': 2,
                    'JobRole': job_role, 'JobSatisfaction': 3, 'MaritalStatus': marital_status,
                    'MonthlyIncome': monthly_income, 'MonthlyRate': 15000,
                    'NumCompaniesWorked': 1, 'Over18': 'Y', 'OverTime': overtime,
                    'PercentSalaryHike': 12, 'PerformanceRating': 3,
                    'RelationshipSatisfaction': 3, 'StandardHours': 40,
                    'StockOptionLevel': 0, 'TotalWorkingYears': 5,
                    'TrainingTimesLastYear': 2, 'WorkLifeBalance': 3,
                    'YearsAtCompany': 3, 'YearsInCurrentRole': 2,
                    'YearsSinceLastPromotion': 1, 'YearsWithCurrManager': 2
                }
                
                # Add new employee
                new_df = pd.DataFrame([new_employee])
                st.session_state.df = pd.concat([st.session_state.df, new_df], ignore_index=True)
                st.session_state.next_id += 1
                
                # Update employees_df
                st.session_state.employees_df = st.session_state.df[["Age", "Department", "MonthlyIncome", "OverTime", "Attrition", "DistanceFromHome"]].copy()
                st.session_state.employees_df["ID"] = range(1, len(st.session_state.employees_df) + 1)
                
                # Save and verify
                success, message = save_data()
                st.markdown(f'<div class="success-box">{message}</div>', unsafe_allow_html=True)
                st.balloons()
                st.rerun()
    
    with tab2:
        st.subheader("📝 Update Employee")
        emp_id = st.number_input("Employee ID", 1, int(df['EmployeeNumber'].max()), step=1)
        emp_data = df[df['EmployeeNumber'] == emp_id]
        
        if not emp_data.empty:
            with st.form("update_form"):
                col1, col2 = st.columns(2)
                with col1:
                    new_age = st.number_input("New Age", value=int(emp_data['Age'].iloc[0]))
                    new_dept = st.selectbox("New Department", sorted(df['Department'].unique()), 
                                          index=list(df['Department'].unique()).index(emp_data['Department'].iloc[0]))
                with col2:
                    new_salary = st.number_input("New Salary", value=int(emp_data['MonthlyIncome'].iloc[0]))
                    new_role = st.selectbox("New Job Role", sorted(df['JobRole'].unique()), 
                                          index=list(df['JobRole'].unique()).index(emp_data['JobRole'].iloc[0]))
                
                if st.form_submit_button("💾 Update Employee"):
                    idx = df[df['EmployeeNumber'] == emp_id].index[0]
                    st.session_state.df.loc[idx, ['Age', 'Department', 'MonthlyIncome', 'JobRole']] = [new_age, new_dept, new_salary, new_role]
                    
                    # Update employees_df
                    st.session_state.employees_df = st.session_state.df[["Age", "Department", "MonthlyIncome", "OverTime", "Attrition", "DistanceFromHome"]].copy()
                    st.session_state.employees_df["ID"] = range(1, len(st.session_state.employees_df) + 1)
                    
                    success, message = save_data()
                    if success:
                        st.markdown(f'<div class="success-box">{message}</div>', unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.markdown(f'<div class="error-box">{message}</div>', unsafe_allow_html=True)
                    st.rerun()
        else:
            st.warning("⚠️ Employee not found!")
    
    with tab3:
        st.subheader("🗑️ Delete Employee")
        emp_id = st.number_input("Employee ID to Delete", 1, int(df['EmployeeNumber'].max()), step=1)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ CONFIRM DELETE", type="primary"):
                before_count = len(st.session_state.df)
                st.session_state.df = st.session_state.df[st.session_state.df['EmployeeNumber'] != emp_id]
                
                st.session_state.employees_df = st.session_state.df[["Age", "Department", "MonthlyIncome", "OverTime", "Attrition", "DistanceFromHome"]].copy()
                st.session_state.employees_df["ID"] = range(1, len(st.session_state.employees_df) + 1)
                
                success, message = save_data()
                if success:
                    st.markdown(f'<div class="success-box">✅ Deleted! {before_count} → {len(st.session_state.df)} records</div>', unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f'<div class="error-box">{message}</div>', unsafe_allow_html=True)
                st.rerun()
        with col2:
            st.info("💡 Check Employee ID in View All tab")
    
    with tab4:
        st.subheader("👀 All Employees")
        st.dataframe(df, use_container_width=True, height=600)

# SEARCH
elif selected_module == "🔍 Search":
    st.markdown('<h1 class="main-header">🔍 Smart Search</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1,3])
    with col1:
        search_type = st.selectbox("Search by", ["Department", "JobRole", "Gender", "Attrition", "All"])
    with col2:
        search_term = st.text_input("Enter search term")
    
    if st.button("🔍 Search", use_container_width=True):
        if search_term:
            if search_type == "All":
                results = df[df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False).any(), axis=1)]
            else:
                results = df[df[search_type].astype(str).str.contains(search_term, case=False, na=False)]
            
            if not results.empty:
                st.success(f"✅ Found {len(results)} matches!")
                st.dataframe(results, use_container_width=True)
            else:
                st.warning("⚠️ No matches found")

# PREDICT ATTRITION
elif selected_module == "🤖 Predict":
    st.markdown('<h1 class="main-header">🤖 AI Attrition Predictor</h1>', unsafe_allow_html=True)
    
    @st.cache_data
    def train_model(_df):
        df_model = _df.copy()
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
        with st.spinner("🚀 Training AI Model..."):
            st.session_state.model, st.session_state.le_dict = train_model(df)
    
    st.subheader("🔮 Predict Employee Risk")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 65, 35)
        department = st.selectbox("Department", sorted(df['Department'].unique()))
        job_role = st.selectbox("Job Role", sorted(df['JobRole'].unique()))
    with col2:
        salary = st.slider("Monthly Salary", 1000, 20000, 6000)
        overtime = st.selectbox("Overtime", ["Yes", "No"])
        distance = st.slider("Distance from Home", 1, 30, 5)
    
    if st.button("🎯 Predict Risk", use_container_width=True):
        pred_data = df.iloc[0:1].copy()
        pred_data['Age'] = age
        pred_data['MonthlyIncome'] = salary
        pred_data['Department'] = department
        pred_data['JobRole'] = job_role
        pred_data['OverTime'] = overtime
        pred_data['DistanceFromHome'] = distance
        
        for col in pred_data.select_dtypes(include=['object']).columns:
            if col in st.session_state.le_dict:
                pred_data[col] = st.session_state.le_dict[col].transform(pred_data[col].astype(str))
        
        prob = st.session_state.model.predict_proba(pred_data.drop('Attrition', axis=1))[0][1] * 100
        st.metric("🎯 Attrition Risk", f"{prob:.1f}%")
        
        if prob > 50:
            st.error("🚨 HIGH RISK - Immediate attention needed!")
        elif prob > 25:
            st.warning("⚠️ Medium Risk")
        else:
            st.success("✅ Low Risk")

# RESUME ANALYZER - ENHANCED NLP
elif selected_module == "📄 Resume Analyzer":
    st.markdown('<h1 class="main-header">📄 AI Resume Analyzer & ATS</h1>', unsafe_allow_html=True)
    
    # Job roles and keywords from dataset
    JOB_KEYWORDS = {
        'Sales Executive': ['sales', 'client', 'customer', 'revenue', 'account', 'business development'],
        'Research Scientist': ['research', 'data', 'analysis', 'experiment', 'model', 'statistics'],
        'Laboratory Technician': ['lab', 'technician', 'experiment', 'sample', 'test', 'equipment'],
        'Sales Representative': ['sales', 'representative', 'lead', 'prospecting', 'negotiation'],
        'Research Director': ['research', 'director', 'strategy', 'leadership', 'publication'],
        'Manufacturing Director': ['manufacturing', 'production', 'operations', 'supply chain'],
        'Healthcare Representative': ['healthcare', 'medical', 'patient', 'clinical'],
        'Manager': ['manager', 'team', 'leadership', 'supervise', 'project management']
    }
    
    uploaded_file = st.file_uploader("📄 Upload PDF Resume", type=['pdf'])
    
    if uploaded_file:
        try:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            
            text_lower = text.lower()
            
            # ATS Scores
            ats_scores = {}
            for role, keywords in JOB_KEYWORDS.items():
                matches = sum(1 for kw in keywords if kw in text_lower)
                score = min((matches / len(keywords)) * 100, 100)
                ats_scores[role] = score
            
            best_fit = max(ats_scores, key=ats_scores.get)
            
            # Extract personal details with better regex
            name_match = re.search(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)', text)
            name = name_match.group(1).strip() if name_match else "Not Found"
            
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
            email = email_match.group() if email_match else "Not Found"
            
            phone_match = re.search(r'[\+]?[1-9][\d]{0,15}', text)
            phone = phone_match.group() if phone_match else "Not Found"
            
            exp_match = re.search(r'(\d+)\s*(?:year|yr)s?\s*(?:of\s+)?experience', text_lower, re.IGNORECASE)
            experience = f"{exp_match.group(1)} years" if exp_match else "Fresher"
            
            # Skills extraction
            skills = []
            for role, keywords in JOB_KEYWORDS.items():
                for skill in keywords:
                    if skill in text_lower:
                        skills.append(skill)
            
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("🎯 Best Fit Score", f"{ats_scores[best_fit]:.0f}%", f"for {best_fit}")
            with col2: st.metric("💼 Best Role", best_fit)
            with col3: st.metric("📚 Experience", experience)
            
            # All scores table
            st.subheader("📊 Role Fit Scores")
            scores_df = pd.DataFrame([
                {'Role': role, 'ATS Score': f"{score:.0f}%"} 
                for role, score in sorted(ats_scores.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(scores_df, use_container_width=True)
            
            # Personal details
            st.subheader("👤 Extracted Profile")
            st.json({
                "Name": name,
                "Email": email,
                "Phone": phone,
                "Experience": experience,
                "Found Skills": list(set(skills))[:10],
                "Missing Skills": [kw for role, kws in JOB_KEYWORDS.items() for kw in kws if kw not in text_lower][:10],
                "Recommendation": best_fit
            })
            
        except Exception as e:
            st.error(f"❌ Error processing resume: {str(e)}")


# Footer
st.markdown("""
<div class="footer-section">
    Built by Archita B | Final Year B.Tech CSE'26 | Data Science & Management Enthusiast
</div>
""", unsafe_allow_html=True)