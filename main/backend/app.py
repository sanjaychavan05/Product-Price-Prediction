import streamlit as st
import fitz  # PyMuPDF
import pdfplumber
import docx2txt
import sqlite3
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process
import re
import io
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Resume Relevance Checker",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'candidates' not in st.session_state:
    st.session_state.candidates = []
if 'job_description' not in st.session_state:
    st.session_state.job_description = ""
if 'model' not in st.session_state:
    st.session_state.model = None

# Database setup
def init_db():
    conn = sqlite3.connect('resume_scores.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            resume_text TEXT,
            score REAL,
            verdict TEXT,
            feedback TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Load sentence transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Resume parsing functions
def extract_text_from_pdf(file):
    """Extract text from PDF using PyMuPDF"""
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        st.error(f"Error extracting PDF text: {str(e)}")
        return ""

def extract_text_from_docx(file):
    """Extract text from DOCX using docx2txt"""
    try:
        text = docx2txt.process(file)
        return text
    except Exception as e:
        st.error(f"Error extracting DOCX text: {str(e)}")
        return ""

def parse_resume_sections(text):
    """Parse resume into key sections"""
    sections = {
        'skills': [],
        'education': [],
        'experience': [],
        'projects': [],
        'certifications': []
    }
    
    # Convert to lowercase for pattern matching
    text_lower = text.lower()
    
    # Extract skills (common patterns)
    skill_patterns = [
        r'python|java|javascript|react|angular|vue|node\.?js|sql|mysql|postgresql|mongodb',
        r'aws|azure|gcp|docker|kubernetes|jenkins|git|github|gitlab',
        r'machine learning|ml|ai|artificial intelligence|deep learning|nlp',
        r'data science|pandas|numpy|scikit-learn|tensorflow|pytorch',
        r'html|css|bootstrap|jquery|php|django|flask|spring|express'
    ]
    
    for pattern in skill_patterns:
        matches = re.findall(pattern, text_lower)
        sections['skills'].extend(matches)
    
    # Extract education
    edu_patterns = [
        r'bachelor[s]?\s+(?:of\s+)?(?:science|arts|engineering|technology|computer science)',
        r'master[s]?\s+(?:of\s+)?(?:science|arts|engineering|technology|computer science)',
        r'phd|doctorate|mba|btech|mtech|be|me'
    ]
    
    for pattern in edu_patterns:
        matches = re.findall(pattern, text_lower)
        sections['education'].extend(matches)
    
    # Extract experience (years)
    exp_patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
        r'(\d+)\+?\s*years?\s*(?:in|of)'
    ]
    
    for pattern in exp_patterns:
        matches = re.findall(pattern, text_lower)
        sections['experience'].extend(matches)
    
    # Remove duplicates and clean
    for key in sections:
        sections[key] = list(set(sections[key]))
    
    return sections

# Job description parsing
def parse_job_description(jd_text):
    """Parse job description to extract requirements"""
    requirements = {
        'skills': [],
        'education': [],
        'experience': [],
        'tools': [],
        'keywords': []
    }
    
    # Convert to lowercase
    jd_lower = jd_text.lower()
    
    # Extract skills
    skill_keywords = [
        'python', 'java', 'javascript', 'react', 'angular', 'vue', 'nodejs', 'sql',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'machine learning',
        'data science', 'html', 'css', 'bootstrap', 'django', 'flask', 'spring'
    ]
    
    for skill in skill_keywords:
        if skill in jd_lower:
            requirements['skills'].append(skill)
    
    # Extract education requirements
    edu_keywords = [
        'bachelor', 'master', 'phd', 'degree', 'computer science', 'engineering',
        'btech', 'mtech', 'be', 'me', 'mba'
    ]
    
    for edu in edu_keywords:
        if edu in jd_lower:
            requirements['education'].append(edu)
    
    # Extract experience requirements
    exp_matches = re.findall(r'(\d+)\+?\s*years?', jd_lower)
    requirements['experience'] = exp_matches
    
    # Extract tools/technologies
    tool_keywords = [
        'jira', 'confluence', 'slack', 'teams', 'excel', 'powerpoint', 'word',
        'linux', 'windows', 'macos', 'ubuntu', 'centos'
    ]
    
    for tool in tool_keywords:
        if tool in jd_lower:
            requirements['tools'].append(tool)
    
    # Extract general keywords
    general_keywords = [
        'leadership', 'communication', 'teamwork', 'problem solving', 'analytical',
        'creative', 'innovative', 'collaborative', 'detail oriented', 'self motivated'
    ]
    
    for keyword in general_keywords:
        if keyword in jd_lower:
            requirements['keywords'].append(keyword)
    
    return requirements

# Scoring functions
def hard_match_score(resume_sections, jd_requirements):
    """Calculate hard match score based on keyword matching"""
    total_score = 0
    max_score = 0
    
    # Skills matching (40% weight)
    resume_skills = set([skill.lower() for skill in resume_sections['skills']])
    jd_skills = set([skill.lower() for skill in jd_requirements['skills']])
    
    if jd_skills:
        skill_matches = len(resume_skills.intersection(jd_skills))
        skill_score = (skill_matches / len(jd_skills)) * 40
        total_score += skill_score
    max_score += 40
    
    # Education matching (20% weight)
    resume_edu = set([edu.lower() for edu in resume_sections['education']])
    jd_edu = set([edu.lower() for edu in jd_requirements['education']])
    
    if jd_edu:
        edu_matches = len(resume_edu.intersection(jd_edu))
        edu_score = (edu_matches / len(jd_edu)) * 20
        total_score += edu_score
    max_score += 20
    
    # Experience matching (20% weight)
    if jd_requirements['experience']:
        jd_exp = max([int(exp) for exp in jd_requirements['experience']])
        resume_exp = 0
        
        for exp in resume_sections['experience']:
            try:
                exp_years = int(exp)
                resume_exp = max(resume_exp, exp_years)
            except:
                continue
        
        if resume_exp >= jd_exp:
            exp_score = 20
        else:
            exp_score = (resume_exp / jd_exp) * 20
        total_score += exp_score
    max_score += 20
    
    # Tools matching (10% weight)
    resume_tools = set([tool.lower() for tool in resume_sections.get('tools', [])])
    jd_tools = set([tool.lower() for tool in jd_requirements['tools']])
    
    if jd_tools:
        tool_matches = len(resume_tools.intersection(jd_tools))
        tool_score = (tool_matches / len(jd_tools)) * 10
        total_score += tool_score
    max_score += 10
    
    # Keywords matching (10% weight)
    resume_text = ' '.join([str(section) for section in resume_sections.values()]).lower()
    jd_keywords = set([keyword.lower() for keyword in jd_requirements['keywords']])
    
    keyword_matches = 0
    for keyword in jd_keywords:
        if keyword in resume_text:
            keyword_matches += 1
    
    if jd_keywords:
        keyword_score = (keyword_matches / len(jd_keywords)) * 10
        total_score += keyword_score
    max_score += 10
    
    return (total_score / max_score * 100) if max_score > 0 else 0

def soft_match_score(resume_text, jd_text, model):
    """Calculate soft match score using sentence transformers"""
    try:
        # Generate embeddings
        resume_embedding = model.encode([resume_text])
        jd_embedding = model.encode([jd_text])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(resume_embedding, jd_embedding)[0][0]
        
        # Convert to percentage
        return similarity * 100
    except Exception as e:
        st.error(f"Error in soft match scoring: {str(e)}")
        return 0

def generate_feedback(resume_sections, jd_requirements):
    """Generate feedback on missing skills/requirements"""
    feedback = []
    
    # Check missing skills
    resume_skills = set([skill.lower() for skill in resume_sections['skills']])
    jd_skills = set([skill.lower() for skill in jd_requirements['skills']])
    missing_skills = jd_skills - resume_skills
    
    if missing_skills:
        feedback.append(f"Missing skills: {', '.join(missing_skills)}")
    
    # Check missing education
    resume_edu = set([edu.lower() for edu in resume_sections['education']])
    jd_edu = set([edu.lower() for edu in jd_requirements['education']])
    missing_edu = jd_edu - resume_edu
    
    if missing_edu:
        feedback.append(f"Missing education: {', '.join(missing_edu)}")
    
    # Check experience gap
    if jd_requirements['experience']:
        jd_exp = max([int(exp) for exp in jd_requirements['experience']])
        resume_exp = 0
        
        for exp in resume_sections['experience']:
            try:
                exp_years = int(exp)
                resume_exp = max(resume_exp, exp_years)
            except:
                continue
        
        if resume_exp < jd_exp:
            feedback.append(f"Experience gap: Need {jd_exp} years, have {resume_exp} years")
    
    # Check missing tools
    resume_tools = set([tool.lower() for tool in resume_sections.get('tools', [])])
    jd_tools = set([tool.lower() for tool in jd_requirements['tools']])
    missing_tools = jd_tools - resume_tools
    
    if missing_tools:
        feedback.append(f"Missing tools: {', '.join(missing_tools)}")
    
    if not feedback:
        feedback.append("Good match! All major requirements are covered.")
    
    return feedback

def calculate_final_score(hard_score, soft_score, hard_weight=0.7, soft_weight=0.3):
    """Calculate final weighted score"""
    return (hard_score * hard_weight) + (soft_score * soft_weight)

def get_verdict(score):
    """Get verdict based on score"""
    if score >= 80:
        return "High"
    elif score >= 60:
        return "Medium"
    else:
        return "Low"

# Database functions
def save_candidate(name, resume_text, score, verdict, feedback):
    """Save candidate data to database"""
    conn = sqlite3.connect('resume_scores.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO candidates (name, resume_text, score, verdict, feedback)
        VALUES (?, ?, ?, ?, ?)
    ''', (name, resume_text, score, verdict, '; '.join(feedback)))
    conn.commit()
    conn.close()

def get_candidates():
    """Get all candidates from database"""
    conn = sqlite3.connect('resume_scores.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM candidates ORDER BY score DESC')
    candidates = cursor.fetchall()
    conn.close()
    return candidates

# Main Streamlit app
def main():
    st.title("📄 Automated Resume Relevance Check System")
    st.markdown("---")
    
    # Sidebar for job description
    with st.sidebar:
        st.header("📋 Job Description")
        
        # Job description input
        jd_option = st.radio("Choose input method:", ["Text Input", "File Upload"])
        
        if jd_option == "Text Input":
            job_description = st.text_area(
                "Enter Job Description:",
                height=300,
                placeholder="Paste the job description here..."
            )
        else:
            jd_file = st.file_uploader("Upload Job Description", type=['txt', 'pdf', 'docx'])
            if jd_file:
                if jd_file.type == "text/plain":
                    job_description = str(jd_file.read(), "utf-8")
                elif jd_file.type == "application/pdf":
                    job_description = extract_text_from_pdf(jd_file)
                elif jd_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    job_description = extract_text_from_docx(jd_file)
                else:
                    job_description = ""
            else:
                job_description = ""
        
        st.session_state.job_description = job_description
        
        if job_description:
            st.success("✅ Job description loaded!")
            with st.expander("Preview Job Description"):
                st.text(job_description[:500] + "..." if len(job_description) > 500 else job_description)
    
    # Main content area
    if not st.session_state.job_description:
        st.warning("⚠️ Please enter or upload a job description in the sidebar first.")
        return
    
    # Load model
    if st.session_state.model is None:
        with st.spinner("Loading AI model..."):
            st.session_state.model = load_model()
    
    # Resume upload section
    st.header("📁 Upload Resumes")
    
    uploaded_files = st.file_uploader(
        "Choose resume files",
        type=['pdf', 'docx'],
        accept_multiple_files=True,
        help="Upload multiple resumes to compare against the job description"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} resume(s) uploaded!")
        
        # Process resumes
        if st.button("🚀 Analyze Resumes", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            candidates_data = []
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                
                # Extract text based on file type
                if file.type == "application/pdf":
                    resume_text = extract_text_from_pdf(file)
                elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    resume_text = extract_text_from_docx(file)
                else:
                    continue
                
                if not resume_text:
                    st.error(f"Could not extract text from {file.name}")
                    continue
                
                # Parse resume sections
                resume_sections = parse_resume_sections(resume_text)
                
                # Parse job description
                jd_requirements = parse_job_description(st.session_state.job_description)
                
                # Calculate scores
                hard_score = hard_match_score(resume_sections, jd_requirements)
                soft_score = soft_match_score(resume_text, st.session_state.job_description, st.session_state.model)
                final_score = calculate_final_score(hard_score, soft_score)
                verdict = get_verdict(final_score)
                
                # Generate feedback
                feedback = generate_feedback(resume_sections, jd_requirements)
                
                # Store candidate data
                candidate_data = {
                    'name': file.name,
                    'resume_text': resume_text,
                    'resume_sections': resume_sections,
                    'hard_score': hard_score,
                    'soft_score': soft_score,
                    'final_score': final_score,
                    'verdict': verdict,
                    'feedback': feedback
                }
                
                candidates_data.append(candidate_data)
                
                # Save to database
                save_candidate(file.name, resume_text, final_score, verdict, feedback)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("✅ Analysis complete!")
            st.session_state.candidates = candidates_data
    
    # Display results
    if st.session_state.candidates:
        st.header("📊 Results Dashboard")
        
        # Create results dataframe
        results_data = []
        for candidate in st.session_state.candidates:
            results_data.append({
                'Candidate': candidate['name'],
                'Score': f"{candidate['final_score']:.1f}",
                'Verdict': candidate['verdict'],
                'Hard Match': f"{candidate['hard_score']:.1f}",
                'Soft Match': f"{candidate['soft_score']:.1f}"
            })
        
        df = pd.DataFrame(results_data)
        
        # Display results table
        st.subheader("📋 Candidate Rankings")
        st.dataframe(df, use_container_width=True)
        
        # Top 5 candidates chart
        st.subheader("📈 Top 5 Candidates")
        top_5 = df.head(5)
        
        fig = px.bar(
            top_5,
            x='Candidate',
            y='Score',
            color='Verdict',
            color_discrete_map={'High': '#2E8B57', 'Medium': '#FFD700', 'Low': '#DC143C'},
            title="Top 5 Candidates by Relevance Score"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed view
        st.subheader("🔍 Detailed Candidate Analysis")
        
        selected_candidate = st.selectbox(
            "Select a candidate for detailed analysis:",
            [candidate['name'] for candidate in st.session_state.candidates]
        )
        
        if selected_candidate:
            candidate = next(c for c in st.session_state.candidates if c['name'] == selected_candidate)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Final Score", f"{candidate['final_score']:.1f}")
            
            with col2:
                st.metric("Hard Match", f"{candidate['hard_score']:.1f}")
            
            with col3:
                st.metric("Soft Match", f"{candidate['soft_score']:.1f}")
            
            # Verdict with color coding
            verdict_color = {'High': '🟢', 'Medium': '🟡', 'Low': '🔴'}
            st.markdown(f"**Verdict:** {verdict_color[candidate['verdict']]} {candidate['verdict']}")
            
            # Feedback
            st.subheader("💡 Feedback & Recommendations")
            for feedback_item in candidate['feedback']:
                st.write(f"• {feedback_item}")
            
            # Resume sections
            st.subheader("📄 Extracted Resume Information")
            
            sections = candidate['resume_sections']
            
            col1, col2 = st.columns(2)
            
            with col1:
                if sections['skills']:
                    st.write("**Skills Found:**")
                    st.write(", ".join(sections['skills']))
                
                if sections['education']:
                    st.write("**Education:**")
                    st.write(", ".join(sections['education']))
            
            with col2:
                if sections['experience']:
                    st.write("**Experience:**")
                    st.write(", ".join(sections['experience']))
                
                if sections['projects']:
                    st.write("**Projects:**")
                    st.write(", ".join(sections['projects']))
            
            # Raw resume text (collapsible)
            with st.expander("📝 View Raw Resume Text"):
                st.text(candidate['resume_text'][:2000] + "..." if len(candidate['resume_text']) > 2000 else candidate['resume_text'])
    
    # Database view
    st.header("💾 Stored Results")
    
    if st.button("🔄 Refresh Database View"):
        candidates_db = get_candidates()
        
        if candidates_db:
            db_df = pd.DataFrame(candidates_db, columns=['ID', 'Name', 'Resume Text', 'Score', 'Verdict', 'Feedback', 'Timestamp'])
            db_df = db_df.drop('Resume Text', axis=1)  # Don't show full text in table
            st.dataframe(db_df, use_container_width=True)
        else:
            st.info("No candidates in database yet.")

if __name__ == "__main__":
    main()