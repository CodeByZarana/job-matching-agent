import os
import logging
import re
import PyPDF2
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='job_agent.log'
)
logger = logging.getLogger('job_agent')

class JobMatchingAgent:
    common_skills = [
        "java", "springboot", "python", "javascript", "typescript", "reactjs", "nodejs",
        "csharp", "aspdotnet", "aws", "azure", "docker", "kubernetes", "cicd", "restapi",
        "microservices", "mysql", "postgresql", "mongodb", "firebase", "git",
        "agile", "scrum", "jira", "reactnative", "graphql", "devops", "unittesting",
        "integrationtesting", "jenkins", "powerautomate", "lambda", "s3", "ec2", "rds"
    ]

    def __init__(self, resume_path, threshold=0.6):
        self.resume_path = resume_path
        self.threshold = threshold
        self.resume_text = self._extract_text_from_resume()
        self.processed_resume = self._preprocess_text(self.resume_text)
        self.skills = self._extract_skills()

    def _extract_text_from_resume(self):
        file_extension = os.path.splitext(self.resume_path)[1].lower()
        text = ""
        try:
            if file_extension == '.pdf':
                with open(self.resume_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
            elif file_extension == '.docx':
                text = docx2txt.process(self.resume_path)
            else:
                with open(self.resume_path, 'r', encoding='utf-8') as file:
                    text = file.read()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from resume: {e}")
            return ""

    def _extract_skills(self):
        return [skill for skill in self.common_skills if skill in self.processed_resume]

    def _preprocess_text(self, text):
        replacements = [
            ("asp.net", "aspdotnet"), (".net", "dotnet"), ("c#", "csharp"),
            ("node.js", "nodejs"), ("react.js", "reactjs"), ("spring boot", "springboot"),
            ("react native", "reactnative"), ("restful api", "restapi"), ("rest api", "restapi"),
            ("unit testing", "unittesting"), ("integration testing", "integrationtesting"), ("ci/cd", "cicd")
        ]
        text = text.lower()
        for old, new in replacements:
            text = text.replace(old, new)

        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def calculate_similarity(self, job_description):
        processed_job = self._preprocess_text(job_description)

        # Contextual similarity (TF-IDF)
        vectorizer = TfidfVectorizer()
        corpus = [self.processed_resume, processed_job]
        tfidf_matrix = vectorizer.fit_transform(corpus)
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        # Skill match score
        skill_match_score = self._calculate_skill_match(processed_job)

        # Combined weighted scoring
        combined_score = (cosine_sim * 0.4) + (skill_match_score * 0.6)

        print(f"\nContextual (TF-IDF) Similarity: {cosine_sim * 100:.2f}%")
        print(f"Skill Match Percentage: {skill_match_score * 100:.2f}%")
        print(f"Combined Matching Score: {combined_score * 100:.2f}%")

        return combined_score

    def _calculate_skill_match(self, processed_job):
        job_skills = [skill for skill in self.common_skills if skill in processed_job]
        matching_skills = [skill for skill in job_skills if skill in self.skills]

        print("Job skills requested:", job_skills)
        print("Your matching skills:", matching_skills)

        return len(matching_skills) / len(job_skills) if job_skills else 0

if __name__ == "__main__":
    resume_path = "C:\\Users\\zarana\\Downloads\\Zarana BMO Resume.pdf"
    agent = JobMatchingAgent(resume_path, threshold=0.6)

    print("\nExtracted Resume Skills:")
    print(agent.skills)

    # Clearly defined test cases:

    test_jobs = [
        {
            "title": "Full Stack Developer",
            "description": """
                Looking for Full Stack Developer with experience in React.js, Spring Boot, AWS, Microservices, RESTful APIs, Docker, and Kubernetes.
            """
        },
        {
            "title": "Frontend Developer",
            "description": """
                We are hiring a Frontend Developer experienced in React.js, JavaScript, TypeScript, HTML, CSS, Redux, Docker, AWS, and GraphQL.
            """
        },
        {
            "title": "Backend Developer",
            "description": """
                Looking for Backend Developer experienced in Java Spring Boot, Microservices architecture, PostgreSQL, MongoDB, Docker, Kubernetes, AWS cloud solutions, and CI/CD pipelines.
            """
        },
        {
            "title": "Mobile Developer",
            "description": """
                Seeking Mobile Developer skilled with React Native, JavaScript, RESTful APIs, Firebase backend services, and Agile methodologies.
            """
        }
    ]

    for job in test_jobs:
        print(f"\n--- Testing for {job['title']} ---")
        similarity_score = agent.calculate_similarity(job["description"])
        
        if similarity_score >= agent.threshold:
            print("✅ This job matches your profile!")
        else:
            print("❌ This job does not meet your matching threshold.")
