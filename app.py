import os
os.environ["HF_HOME"] = "./hf_cache"
import re
try:
    import pdfplumber
except:
    pdfplumber = None
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
import numpy as np

import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import PyPDF2
import docx
import time

# ---------------- CONFIG ----------------
st.set_page_config(page_title="HireSense AI", layout="wide")

# ---------------- DARK MODE TOGGLE ----------------
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

# ---------------- CSS ----------------
if dark_mode:
    bg = "#0B0F17"
    text = "#F8FAFC"
    sidebar_bg = "rgba(15, 23, 42, 0.72)"
    input_bg = "rgba(30, 41, 59, 0.62)"
    glass_bg = "rgba(15, 23, 42, 0.58)"
    border = "rgba(255, 255, 255, 0.14)"
    shadow = "0 24px 70px rgba(0, 0, 0, 0.45)"
else:
    bg = "#F5F7FB"
    text = "#111827"
    sidebar_bg = "rgba(255, 255, 255, 0.62)"
    input_bg = "rgba(255, 255, 255, 0.70)"
    glass_bg = "rgba(255, 255, 255, 0.58)"
    border = "rgba(255, 255, 255, 0.65)"
    shadow = "0 24px 70px rgba(31, 38, 135, 0.14)"

st.markdown(f"""
<style>
.stApp {{
    background:
        radial-gradient(circle at top left, rgba(108, 99, 255, 0.20), transparent 32%),
        radial-gradient(circle at top right, rgba(0, 201, 167, 0.18), transparent 30%),
        linear-gradient(135deg, {bg}, {bg});
    color: {text};
}}

.block-container {{
    padding-top: 3rem;
    max-width: 1180px;
}}

section[data-testid="stSidebar"] {{
    background: {sidebar_bg};
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border-right: 1px solid {border};
}}

section[data-testid="stSidebar"] * {{
    color: {text};
}}

h1, h2, h3, h4, h5, h6, p, label, span, div {{
    color: {text};
}}

textarea, input {{
    background: {input_bg} !important;
    color: {text} !important;
    border: 1px solid {border} !important;
    border-radius: 16px !important;
}}

div[data-baseweb="select"] > div {{
    background: {input_bg} !important;
    color: {text} !important;
    border: 1px solid {border} !important;
    border-radius: 16px !important;
}}

.stButton>button, .stDownloadButton>button {{
    background: linear-gradient(135deg, rgba(108, 99, 255, 0.95), rgba(0, 201, 167, 0.95)) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.35) !important;
    border-radius: 999px !important;
    height: 3.2em;
    width: 100%;
    font-size: 16px;
    font-weight: 700;
    box-shadow: 0 14px 35px rgba(0, 201, 167, 0.22);
}}

.stButton>button:hover, .stDownloadButton>button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 18px 45px rgba(108, 99, 255, 0.28);
}}

div[data-testid="stFileUploader"], div[data-testid="stDataFrame"], div[data-testid="stAlert"] {{
    background: {glass_bg};
    border: 1px solid {border};
    border-radius: 24px;
    box-shadow: {shadow};
    backdrop-filter: blur(26px);
    -webkit-backdrop-filter: blur(26px);
    padding: 16px;
}}

div[data-testid="stFileUploader"] section {{
    background: {input_bg} !important;
    border: 1px dashed {border} !important;
    border-radius: 20px !important;
}}

div[role="radiogroup"] {{
    background: {glass_bg};
    border: 1px solid {border};
    border-radius: 18px;
    padding: 12px;
    backdrop-filter: blur(20px);
}}

</style>
""", unsafe_allow_html=True)
# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:center;
background: linear-gradient(to right, #6C63FF, #00C9A7);
-webkit-background-clip: text;
color: transparent;'>
🚀 HireSense AI
</h1>
""", unsafe_allow_html=True)

st.subheader("Advanced Resume Intelligence System")

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L3-v2', cache_folder="./hf_cache")

with st.spinner("Loading AI model... Please wait"):
    model = load_model()


# ---------------- JOB ROLE DATA ----------------
job_roles = {
    "Python Developer": "Python, APIs, backend, SQL required",
    "Data Scientist": "Python, ML, statistics, NLP, data analysis",
    "Frontend Developer": "HTML, CSS, JavaScript, React",
    "Backend Developer": "Node.js, databases, APIs, system design",
    "Full Stack Developer": "Frontend + backend, MERN/MEAN stack",
    "AI/ML Engineer": "Machine Learning, Deep Learning, TensorFlow, PyTorch",
    "Data Analyst": "Excel, SQL, Power BI, data visualization",
    "Cloud Engineer": "AWS, Azure, Docker, Kubernetes",
    "DevOps Engineer": "CI/CD, Docker, Kubernetes, automation",
    "Cybersecurity Analyst": "Network security, ethical hacking, SIEM",
    "Mobile App Developer": "Android, iOS, Flutter, React Native",
    "UI/UX Designer": "Figma, prototyping, user research",
    "Blockchain Developer": "Solidity, Web3, smart contracts",
    "Software Tester (QA)": "Manual testing, automation, Selenium",
    "Product Manager": "Agile, roadmap planning, stakeholder management",
    "Business Analyst": "Requirement analysis, data insights, communication",
    "Game Developer": "Unity, Unreal Engine, C#",
    "AR/VR Developer": "Unity, 3D modeling, immersive tech"
}

skill_suggestions = {
    "Python Developer": ["Python", "Django", "Flask", "FastAPI", "REST APIs", "SQL", "PostgreSQL", "Redis", "Celery", "Docker", "Git", "Unit Testing", "Async Programming"],
    "Data Scientist": ["Python", "Pandas", "NumPy", "Scikit-learn", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "NLP", "Feature Engineering", "Model Deployment", "Statistics", "Data Visualization", "Power BI", "Tableau"],
    "Frontend Developer": ["HTML", "CSS", "JavaScript", "TypeScript", "React", "Next.js", "Tailwind CSS", "Redux", "Responsive Design", "Web Performance Optimization", "API Integration", "UI/UX Principles"],
    "Backend Developer": ["Node.js", "Express.js", "Python", "Django", "REST APIs", "GraphQL", "Microservices", "MongoDB", "PostgreSQL", "Authentication (JWT, OAuth)", "Docker", "System Design", "Caching (Redis)"],
    "Full Stack Developer": ["React", "Node.js", "MongoDB", "Express.js", "Next.js", "TypeScript", "REST APIs", "GraphQL", "Authentication", "Docker", "CI/CD", "System Design"],
    "AI/ML Engineer": ["Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Transformers", "LLMs", "Hugging Face", "Computer Vision", "NLP", "MLOps", "Model Deployment", "Data Pipelines", "Python"],
    "Data Analyst": ["Excel", "SQL", "Power BI", "Tableau", "Python", "Pandas", "Data Cleaning", "Data Visualization", "Statistics", "Dashboarding", "Business Insights"],
    "Cloud Engineer": ["AWS", "Azure", "Google Cloud", "Docker", "Kubernetes", "Terraform", "CI/CD", "Cloud Security", "Load Balancing", "Monitoring (Prometheus, Grafana)"],
    "DevOps Engineer": ["Docker", "Kubernetes", "Jenkins", "CI/CD Pipelines", "Terraform", "Linux", "Shell Scripting", "Monitoring", "AWS", "Infrastructure as Code"],
    "Cybersecurity Analyst": ["Network Security", "Ethical Hacking", "Penetration Testing", "SIEM", "Threat Analysis", "Vulnerability Assessment", "Firewalls", "Encryption", "OWASP Top 10"],
    "Mobile App Developer": ["Flutter", "React Native", "Kotlin", "Swift", "Firebase", "REST APIs", "UI/UX Design", "App Deployment", "State Management"],
    "UI/UX Designer": ["Figma", "Adobe XD", "Wireframing", "Prototyping", "User Research", "Interaction Design", "Usability Testing", "Design Systems"],
    "Blockchain Developer": ["Solidity", "Smart Contracts", "Ethereum", "Web3.js", "Hardhat", "DeFi", "Cryptography", "NFT Development"],
    "Software Tester (QA)": ["Manual Testing", "Automation Testing", "Selenium", "Cypress", "JUnit", "API Testing", "Performance Testing", "Bug Tracking Tools"],
    "Product Manager": ["Agile", "Scrum", "Roadmapping", "Stakeholder Management", "User Stories", "Market Research", "Data-driven Decisions"],
    "Business Analyst": ["Requirement Gathering", "SQL", "Data Analysis", "Process Modeling", "Stakeholder Communication", "Documentation", "Business Intelligence"],
    "Game Developer": ["Unity", "Unreal Engine", "C#", "C++", "Game Physics", "3D Modeling", "Animation", "AR/VR Integration"],
    "AR/VR Developer": ["Unity", "Unreal Engine", "C#", "3D Modeling", "XR Development", "Spatial Computing", "VR Interaction Design"]
}

# ---------------- FUNCTIONS ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text(file):
    text = ""
    try:
        file_name = file.name.lower()

        if file_name.endswith(".pdf"):
            if pdfplumber:
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + " "
            else:
                reader = PyPDF2.PdfReader(file)
                for p in reader.pages:
                    page_text = p.extract_text()
                    if page_text:
                        text += page_text + " "

        elif file_name.endswith(".docx"):
            doc = docx.Document(file)
            for para in doc.paragraphs:
                text += para.text + " "

        elif file_name.endswith(".txt"):
            text = file.read().decode("utf-8", errors="ignore")

    except Exception as e:
        st.error(f"Error reading file: {e}")

    return clean_text(text)
def is_valid_resume(text):
    if not text or len(text.split()) < 80:
        return False

    resume_keywords = [
        "experience", "education", "skills", "projects",
        "project", "certification", "certifications", "summary",
        "objective", "work experience", "employment",
        "internship", "degree", "university", "college",
        "email", "phone", "linkedin", "github", "profile",
        "technical skills", "professional summary"
    ]

    matches = 0
    text = text.lower()

    for keyword in resume_keywords:
        if keyword in text:
            matches += 1

    return matches >= 3
def extract_skills_from_text(text, role):
    if not text:
        return []

    text = clean_text(text)
    skills = skill_suggestions.get(role, [])

    found = []
    for skill in skills:
        skill_clean = clean_text(skill)
        if skill_clean in text:
            found.append(skill)

    return found

def calculate_ats_score(resume_text, job_desc, role):
    required_skills = skill_suggestions.get(role, [])
    resume_skills = extract_skills_from_text(resume_text, role)

    emb1 = model.encode(resume_text)
    emb2 = model.encode(job_desc)

    semantic_score = util.cos_sim(emb1, emb2).item() * 100
    semantic_score = max(0, min(semantic_score, 100))

    if required_skills:
        skill_score = (len(resume_skills) / len(required_skills)) * 100
    else:
        skill_score = 0

    ats = (semantic_score * 0.60) + (skill_score * 0.40)
    ats = round(ats, 2)

    missing_skills = [s for s in required_skills if s not in resume_skills]

    return ats, resume_skills, missing_skills, semantic_score, skill_score
def generate_ai_suggestions(resume_text, job_desc, role, missing_skills, semantic_score, skill_score):
    suggestions = []

    if semantic_score < 50:
        suggestions.append(
            "Your resume does not strongly match the job description. Add more role-specific keywords and rewrite your summary according to the selected job role."
        )

    if skill_score < 60:
        suggestions.append(
            "Your resume is missing important skills for this role. Add relevant projects, tools, or certifications for the missing skills."
        )

    if missing_skills:
        top_missing = ", ".join(missing_skills[:5])
        suggestions.append(
            f"Focus on these missing skills first: {top_missing}."
        )

    if len(resume_text.split()) < 250:
        suggestions.append(
            "Your resume text looks short. Add more measurable achievements, project details, and responsibilities."
        )

    if "project" not in resume_text.lower() and "projects" not in resume_text.lower():
        suggestions.append(
            "Add a Projects section because ATS systems value practical proof of skills."
        )

    if "certification" not in resume_text.lower() and "certifications" not in resume_text.lower():
        suggestions.append(
            "Add certifications related to the selected role to improve credibility."
        )

    if not suggestions:
        suggestions.append(
            "Your resume matches this role well. Improve it further by adding measurable achievements, numbers, and stronger action verbs."
        )

    return suggestions


def create_pdf_report(role, ats, semantic_score, skill_score, resume_skills, missing_skills, ai_suggestions):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)

    width, height = letter
    y = height - 50

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, y, "HireSense AI - Resume Screening Report")

    y -= 40
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, f"Selected Role: {role}")

    y -= 25
    pdf.drawString(50, y, f"ATS Score: {ats}%")

    y -= 25
    pdf.drawString(50, y, f"Semantic Match: {round(semantic_score, 2)}%")

    y -= 25
    pdf.drawString(50, y, f"Skill Match: {round(skill_score, 2)}%")

    y -= 40
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, y, "Skills Found:")
    y -= 20

    pdf.setFont("Helvetica", 11)
    found_text = ", ".join(resume_skills) if resume_skills else "No known skills detected"
    for line in [found_text[i:i+90] for i in range(0, len(found_text), 90)]:
        pdf.drawString(50, y, line)
        y -= 15

    y -= 20
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, y, "Missing Skills:")
    y -= 20

    pdf.setFont("Helvetica", 11)
    missing_text = ", ".join(missing_skills) if missing_skills else "No major missing skills"
    for line in [missing_text[i:i+90] for i in range(0, len(missing_text), 90)]:
        pdf.drawString(50, y, line)
        y -= 15

    y -= 20
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, y, "AI Suggestions:")
    y -= 20

    pdf.setFont("Helvetica", 11)
    for suggestion in ai_suggestions:
        text = f"- {suggestion}"
        for line in [text[i:i+90] for i in range(0, len(text), 90)]:
            pdf.drawString(50, y, line)
            y -= 15
        y -= 5

        if y < 80:
            pdf.showPage()
            y = height - 50
            pdf.setFont("Helvetica", 11)

    pdf.save()
    buffer.seek(0)
    return buffer


# ---------------- ROLE SELECT FUNCTION (SIMPLIFIED) ----------------
def role_selector():
    return st.selectbox("Select Job Role", list(job_roles.keys()))


# ---------------- MENU ----------------
menu = st.sidebar.selectbox("switch tool",
    ["Resume Analyzer", "Skill Gap", "Bulk Screening", "About"])


# ---------------- RESUME ANALYZER ----------------
if menu == "Resume Analyzer":
    st.header("📄 Resume Analyzer")

    col1, col2 = st.columns(2)

    with col1:
        resume = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

    with col2:
        role = role_selector()
        job_desc = st.text_area("Job Description", job_roles[role])

    if st.button("🚀 Analyze Resume"):
        if not resume:
            st.error("Upload resume")
        else:
            with st.spinner("🔍 AI analyzing..."):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i + 1)

                text = extract_text(resume)

                if not text.strip():
                    st.error("Could not extract text from the resume. Please upload a readable PDF, DOCX, or TXT file.")
                    st.stop()

                if not is_valid_resume(text):
                    st.error("This file does not look like a valid resume. Please upload an actual resume with sections like Skills, Education, Experience, Projects, or Certifications.")
                    st.stop()

                ats, resume_skills, missing_skills, semantic_score, skill_score = calculate_ats_score(text, job_desc, role)
                        # ATS CARD
            st.markdown(f"""
            <div style="
            background:{glass_bg};
            border:1px solid {border};
            box-shadow:{shadow};
            backdrop-filter:blur(26px);
            -webkit-backdrop-filter:blur(26px);
            padding:28px;
            border-radius:28px;
            text-align:center;
            font-size:34px;
            font-weight:800;">
            ATS Score: {ats}%
            </div>
            """, unsafe_allow_html=True)
            st.info(f"Semantic Match: {round(semantic_score, 2)}% | Skill Match: {round(skill_score, 2)}%")
                                    # Circular ATS Meter
            st.markdown(f"""
            <div style="display:flex; justify-content:center; margin:28px 0;">
                <div style="
                    width:190px;
                    height:190px;
                    border-radius:50%;
                    background:conic-gradient(#00C9A7 {ats * 3.6}deg, rgba(148,163,184,0.28) 0deg);
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    box-shadow:{shadow};
                    border:1px solid {border};">
                    <div style="
                        width:138px;
                        height:138px;
                        border-radius:50%;
                        background:{glass_bg};
                        backdrop-filter:blur(26px);
                        -webkit-backdrop-filter:blur(26px);
                        border:1px solid {border};
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        flex-direction:column;">
                        <div style="font-size:34px; font-weight:800; color:{text};">{ats}%</div>
                        <div style="font-size:14px; color:{text}; opacity:0.75;">ATS Match</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            # 🔥 SHOW DETECTED SKILLS
            st.subheader("✅ Skills Found in Resume")
            if resume_skills:
                st.success(", ".join(resume_skills))
            else:
                st.warning("No known skills detected")

            # 🎯 SHOW ONLY MISSING SKILLS
            st.subheader("📌 Suggested Skills to Learn")

            if missing_skills:
                cols = st.columns(min(len(missing_skills), 4))
                for i, skill in enumerate(missing_skills):
                    cols[i % len(cols)].markdown(f"""
                    <div style="
                    background:#FF4B4B;
                    padding:10px;
                    border-radius:10px;
                    text-align:center;
                    color:white;
                    font-weight:bold;">
                    {skill}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("🎉 Your resume already matches most required skills!")


# ---------------- SKILL GAP ----------------
elif menu == "Skill Gap":
    st.header("Skill Gap Analyzer")

    col1, col2 = st.columns(2)

    with col1:
        resume = st.file_uploader("Upload Resume for Skill Gap", type=["pdf", "docx", "txt"])

    with col2:
        role = role_selector()

    if st.button("🔍 Analyze Skill Gap"):
        if not resume:
            st.error("Upload resume")
        else:
            with st.spinner("Analyzing skill gap..."):
                text = extract_text(resume)

                if not text.strip():
                    st.error("Could not extract text from the resume. Please upload a readable PDF, DOCX, or TXT file.")
                    st.stop()

                if not is_valid_resume(text):
                    st.error("This file does not look like a valid resume. Please upload an actual resume with sections like Skills, Education, Experience, Projects, or Certifications.")
                    st.stop()

                required_skills = skill_suggestions.get(role, [])

                resume_skills = extract_skills_from_text(text, role)
                missing_skills = [s for s in required_skills if s not in resume_skills]

                if required_skills:
                    skill_gap_score = round((len(resume_skills) / len(required_skills)) * 100, 2)
                else:
                    skill_gap_score = 0

            st.markdown(f"""
            <div style="
            background: linear-gradient(to right, #6C63FF, #00C9A7);
            padding:25px;
            border-radius:15px;
            text-align:center;
            font-size:30px;
            font-weight:bold;
            color:white;">
            Skill Match: {skill_gap_score}%
            </div>
            """, unsafe_allow_html=True)

            st.subheader("✅ Skills You Already Have")
            if resume_skills:
                st.success(", ".join(resume_skills))
            else:
                st.warning("No matching skills found for this role.")

            st.subheader("📌 Skills You Should Add / Learn")
            if missing_skills:
                cols = st.columns(min(len(missing_skills), 4))
                for i, skill in enumerate(missing_skills):
                    cols[i % len(cols)].markdown(f"""
                    <div style="
                    background:#FF4B4B;
                    padding:10px;
                    border-radius:10px;
                    text-align:center;
                    color:white;
                    font-weight:bold;
                    margin-bottom:8px;">
                    {skill}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("🎉 No major skill gaps found for this role!")

            suggestions = generate_ai_suggestions(
                text,
                job_roles[role],
                role,
                missing_skills,
                skill_gap_score,
                skill_gap_score
            )

            st.subheader("🤖 Improvement Suggestions")
            for suggestion in suggestions:
                st.info(suggestion)

# ---------------- BULK SCREENING ----------------
elif menu == "Bulk Screening":
    st.header("📂 Bulk Resume Screening")

    role = role_selector()
    job_desc = st.text_area("Job Description", job_roles[role])

    screening_mode = st.radio(
        "Choose Screening Input",
        ["Upload Resume Files", "Upload CSV Dataset"]
    )

    top_n = st.number_input(
        "How many top resumes do you want?",
        min_value=1,
        value=5,
        step=1
    )

    results = []

    if screening_mode == "Upload Resume Files":
        resumes = st.file_uploader(
            "Upload Multiple Resumes",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )

        if st.button("🚀 Screen Resumes"):
            if not resumes:
                st.error("Upload at least one resume")
            else:
                with st.spinner("Screening resumes..."):
                    for resume_file in resumes:
                        text = extract_text(resume_file)

                        if text.strip() and is_valid_resume(text):
                            ats, resume_skills, missing_skills, semantic_score, skill_score = calculate_ats_score(text, job_desc, role)

                            results.append({
                                "Resume": resume_file.name,
                                "ATS Score": ats,
                                "Semantic Match": round(semantic_score, 2),
                                "Skill Match": round(skill_score, 2),
                                "Skills Found": ", ".join(resume_skills),
                                "Missing Skills": ", ".join(missing_skills)
                            })
                        else:
                            results.append({
                                "Resume": resume_file.name,
                                "ATS Score": 0,
                                "Semantic Match": 0,
                                "Skill Match": 0,
                                "Skills Found": "",
                                "Missing Skills": "Invalid resume or could not extract text"

                            })

                df = pd.DataFrame(results)
                df = df.sort_values(by="ATS Score", ascending=False).reset_index(drop=True)
                df.index = df.index + 1

                top_df = df.head(top_n)

                st.subheader(f"🏆 Top {top_n} Ranked Resumes")
                st.dataframe(top_df)

                csv_data = top_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Top Ranked Resumes CSV",
                    data=csv_data,
                    file_name=f"top_{top_n}_ranked_resumes.csv",
                    mime="text/csv"
                )

    elif screening_mode == "Upload CSV Dataset":
        csv_file = st.file_uploader(
            "Upload CSV Dataset",
            type=["csv"]
        )

        if st.button("🚀 Screen CSV Dataset"):
            if not csv_file:
                st.error("Upload a CSV file")
            else:
                with st.spinner("Screening CSV dataset..."):
                    df_csv = pd.read_csv(csv_file)

                    possible_text_columns = [
                        "Resume Text",
                        "resume_text",
                        "Resume_str",
                        "resume_str",
                        "Resume_Text",
                        "cleaned_text",
                        "Text",
                        "text",
                        "Resume",
                        "resume"
                    ]
                    resume_text_column = None
                    for col in possible_text_columns:
                        if col in df_csv.columns:
                            resume_text_column = col
                            break

                    if resume_text_column is None:
                        st.error("CSV must contain a resume text column like 'Resume_str', 'Resume Text', 'Resume_Text', 'cleaned_text', 'Text', or 'Resume'.")

                        st.stop()

                    for index, row in df_csv.iterrows():
                        resume_text = str(row[resume_text_column])

                        if "Name" in df_csv.columns:
                            resume_name = row["Name"]
                        elif "name" in df_csv.columns:
                            resume_name = row["name"]
                        elif "Candidate" in df_csv.columns:
                            resume_name = row["Candidate"]
                        elif "candidate" in df_csv.columns:
                            resume_name = row["candidate"]
                        else:
                            resume_name = f"Candidate {index + 1}"

                        if resume_text.strip() and is_valid_resume(resume_text):
                            ats, resume_skills, missing_skills, semantic_score, skill_score = calculate_ats_score(resume_text, job_desc, role)

                            results.append({
                                "Candidate": resume_name,
                                "ATS Score": ats,
                                "Semantic Match": round(semantic_score, 2),
                                "Skill Match": round(skill_score, 2),
                                "Skills Found": ", ".join(resume_skills),
                                "Missing Skills": ", ".join(missing_skills)
                            })
                        else:
                            results.append({
                                "Candidate": resume_name,
                                "ATS Score": 0,
                                "Semantic Match": 0,
                                "Skill Match": 0,
                                "Skills Found": "",
                                "Missing Skills": "Invalid resume or empty resume text"
                            })

                df = pd.DataFrame(results)
                df = df.sort_values(by="ATS Score", ascending=False).reset_index(drop=True)
                df.index = df.index + 1

                top_df = df.head(top_n)

                st.subheader(f"🏆 Top {top_n} Ranked Candidates")
                st.dataframe(top_df)

                csv_data = top_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Top Ranked Candidates CSV",
                    data=csv_data,
                    file_name=f"top_{top_n}_ranked_candidates.csv",
                    mime="text/csv"
                )
# ---------------- ABOUT ----------------
else:
    st.write("""
    ### HireSense AI (Advanced)
    - ATS Resume Scoring
    - Skill Gap Detection
    - Bulk Resume Screening
    - AI Suggestions
    """)

    # ✅ Added credit (no removal)
    st.markdown("### 👨‍💻 Made by Abhay")

    # ✅ WhatsApp Button
    whatsapp_number = "918808528969" 

    st.markdown(f"""
    <a href="https://wa.me/{whatsapp_number}" target="_blank">
        <button style="
            background-color:#25D366;
            color:white;
            padding:12px 20px;
            border:none;
            border-radius:10px;
            font-size:16px;
            cursor:pointer;">
            💬 Contact on WhatsApp
        </button>
    </a>
    """, unsafe_allow_html=True)
