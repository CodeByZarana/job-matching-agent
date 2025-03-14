import os
import logging
import re
import PyPDF2
import docx2txt
import json
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='job_agent.log'
)
logger = logging.getLogger('job_agent')

class EnhancedJobMatchingAgent:
    # Categorized skills list for better analysis
    skill_categories = {
        "languages": [
            "java", "python", "javascript", "typescript", "csharp", "c++", "php", "ruby", 
            "golang", "swift", "kotlin", "scala", "rust", "shell", "bash", "sql", "r"
        ],
        "frontend": [
            "html", "css", "sass", "less", "reactjs", "vuejs", "angular", "svelte", "jquery", 
            "bootstrap", "tailwind", "materialui", "redux", "nextjs", "webpack"
        ],
        "backend": [
            "nodejs", "springboot", "django", "flask", "laravel", "express", 
            "rails", "aspnetcore", "aspdotnet", "dotnet", "fastapi", "spring", "hibernate"
        ],
        "databases": [
            "mysql", "postgresql", "mongodb", "sql", "oracle", "sqlserver", "sqlite", 
            "firebase", "dynamodb", "elasticsearch", "redis", "cassandra", "neo4j", "graphql"
        ],
        "devops": [
            "git", "github", "gitlab", "bitbucket", "docker", "kubernetes", "jenkins", 
            "circleci", "travisci", "ansible", "terraform", "puppet", "chef", "cicd", 
            "aws", "azure", "gcp", "lambda", "ec2", "s3", "rds", "heroku", "prometheus", "grafana"
        ],
        "architecture": [
            "microservices", "restapi", "serverless", "monolith", "eventdriven", 
            "mvc", "mvvm", "grpc", "soap", "oauth", "jwt"
        ],
        "testing": [
            "unittesting", "integrationtesting", "selenium", "cypress", "jest", "mocha", 
            "chai", "puppeteer", "pytest", "junit", "cucumber", "jasmine"
        ],
        "methodology": [
            "agile", "scrum", "kanban", "waterfall", "lean", "jira", "confluence", 
            "trello", "asana", "devops"
        ],
        "mobile": [
            "android", "ios", "reactnative", "flutter", "xamarin", "ionic", "cordova", 
            "swift", "kotlin"
        ],
        "ai_data": [
            "machinelearning", "deeplearning", "tensorflow", "pytorch", "keras", 
            "opencv", "sklearn", "pandas", "numpy", "hadoop", "spark", "tableau", 
            "powerbi", "jupyter", "nlp"
        ]
    }
    
    # Create a flattened list of all skills (for backward compatibility)
    common_skills = [skill for category in skill_categories.values() for skill in category]
    
    # Skill replacements for normalization
    skill_replacements = [
        ("asp.net", "aspdotnet"), (".net", "dotnet"), ("c#", "csharp"),
        ("node.js", "nodejs"), ("react.js", "reactjs"), ("vue.js", "vuejs"),
        ("spring boot", "springboot"), ("react native", "reactnative"),
        ("restful api", "restapi"), ("rest api", "restapi"),
        ("unit testing", "unittesting"), ("integration testing", "integrationtesting"),
        ("ci/cd", "cicd"), ("machine learning", "machinelearning"), 
        ("deep learning", "deeplearning"), ("sql server", "sqlserver")
    ]

    def __init__(self, resume_path, threshold=0.6, weights=None):
        # Keep the original resume reading functionality
        self.resume_path = resume_path
        self.threshold = threshold
        
        # Custom weights for different components of the match score
        self.weights = weights or {
            "tfidf": 0.4,        # Contextual similarity weight 
            "skills": 0.4,       # Skills match weight
            "categories": 0.2    # Category match weight (new)
        }
        
        # Extract and process resume
        self.resume_text = self._extract_text_from_resume()
        self.processed_resume = self._preprocess_text(self.resume_text)
        self.skills = self._extract_skills()
        self.categorized_skills = self._categorize_skills()
        
        # Print extracted resume info
        print("\nExtracted Resume Skills:")
        for category, skills in self.categorized_skills.items():
            if skills:
                print(f"{category.capitalize()}: {', '.join(skills)}")
        
        # Calculate resume stats
        self.resume_stats = self._calculate_resume_stats()
        
    def _extract_text_from_resume(self):
        """Extract text from the resume file - same as original"""
        file_extension = os.path.splitext(self.resume_path)[1].lower()
        text = ""
        try:
            if file_extension == '.pdf':
                with open(self.resume_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() or ""
            elif file_extension == '.docx':
                text = docx2txt.process(self.resume_path)
            else:
                with open(self.resume_path, 'r', encoding='utf-8') as file:
                    text = file.read()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from resume: {e}")
            return ""
            
    def _preprocess_text(self, text):
        """Preprocess text for better matching - enhanced version"""
        if not text:
            return ""
            
        # Apply skill replacements
        text = text.lower()
        for old, new in self.skill_replacements:
            text = text.replace(old, new)

        # Remove punctuation and normalize spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
        
    def _extract_skills(self):
        """Extract skills from resume by matching against known skills list"""
        found_skills = []
        for skill in self.common_skills:
            # Look for the skill as a whole word
            if re.search(r'\b' + re.escape(skill) + r'\b', self.processed_resume):
                found_skills.append(skill)
        return found_skills
        
    def _categorize_skills(self):
        """Group extracted skills by category"""
        categorized = {}
        for category, skills in self.skill_categories.items():
            matching_skills = [skill for skill in skills if skill in self.skills]
            if matching_skills:
                categorized[category] = matching_skills
        return categorized
        
    def _calculate_resume_stats(self):
        """Calculate statistics about the resume"""
        # Count skills by category
        category_counts = {category: len(skills) for category, skills in self.categorized_skills.items()}
        
        # Identify primary and secondary skill areas
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        primary_categories = [category for category, count in sorted_categories[:3] if count > 0]
        
        # Check for experience indicators
        years_pattern = r'(\d+)\+?\s*(?:years|year|yr|yrs)(?:\s+of\s+experience|\s+experience)'
        experience_matches = re.findall(years_pattern, self.resume_text.lower())
        years_of_experience = max([int(y) for y in experience_matches]) if experience_matches else None
        
        # Check for seniority indicators
        seniority_keywords = {
            "senior": ["senior", "sr", "lead", "principal", "staff", "architect"],
            "mid": ["mid", "intermediate", "associate"],
            "junior": ["junior", "jr", "entry", "intern", "graduate"]
        }
        
        seniority = None
        resume_lower = self.resume_text.lower()
        for level, keywords in seniority_keywords.items():
            if any(re.search(r'\b' + re.escape(keyword) + r'\b', resume_lower) for keyword in keywords):
                seniority = level
                break
                
        # Inference if seniority not explicitly found
        if not seniority and years_of_experience:
            if years_of_experience >= 8:
                seniority = "senior"
            elif years_of_experience >= 3:
                seniority = "mid"
            else:
                seniority = "junior"
                
        return {
            "total_skills": len(self.skills),
            "category_counts": category_counts,
            "primary_categories": primary_categories,
            "years_of_experience": years_of_experience,
            "seniority": seniority
        }
        
    def _extract_job_skills(self, processed_job):
        """Extract skills mentioned in the job description"""
        return [skill for skill in self.common_skills if re.search(r'\b' + re.escape(skill) + r'\b', processed_job)]
        
    def _categorize_job_skills(self, job_skills):
        """Group job skills by category"""
        categorized = {}
        for category, skills in self.skill_categories.items():
            matching_skills = [skill for skill in skills if skill in job_skills]
            if matching_skills:
                categorized[category] = matching_skills
        return categorized
    
    def _calculate_skill_match(self, job_skills):
        """Calculate skill match percentage"""
        if not job_skills:
            return 0
            
        matching_skills = [skill for skill in job_skills if skill in self.skills]
        return len(matching_skills) / len(job_skills)
    
    def _calculate_category_match(self, job_categories):
        """Calculate category match with weighting for important categories"""
        if not job_categories:
            return 0
            
        # For each category, calculate match percentage
        category_scores = []
        
        for category, skills in job_categories.items():
            # Get resume skills in this category
            resume_skills_in_category = self.categorized_skills.get(category, [])
            
            # Calculate match for this category
            if skills:
                score = len([s for s in skills if s in resume_skills_in_category]) / len(skills)
                
                # Weight by importance (more skills in a category = more important)
                weight = len(skills) / sum(len(s) for s in job_categories.values())
                category_scores.append(score * weight)
        
        return sum(category_scores)
    
    def _extract_job_requirements(self, job_description):
        """Extract key requirements from the job description"""
        # Experience level required
        years_pattern = r'(\d+)[\+]?\s*(?:years|year|yr|yrs)(?:\s+of\s+experience|\s+experience)'
        experience_matches = re.findall(years_pattern, job_description.lower())
        required_years = max([int(y) for y in experience_matches]) if experience_matches else None
        
        # Check for seniority keywords
        seniority_keywords = {
            "senior": ["senior", "sr", "lead", "principal", "staff", "architect"],
            "mid": ["mid", "intermediate", "associate"],
            "junior": ["junior", "jr", "entry", "intern", "graduate"]
        }
        
        job_level = None
        job_lower = job_description.lower()
        for level, keywords in seniority_keywords.items():
            if any(re.search(r'\b' + re.escape(keyword) + r'\b', job_lower) for keyword in keywords):
                job_level = level
                break
                
        # If level not found but years mentioned, infer level
        if not job_level and required_years:
            if required_years >= 7:
                job_level = "senior"
            elif required_years >= 3:
                job_level = "mid"
            else:
                job_level = "junior"
                
        # Check for education requirements
        education_keywords = {
            "bachelor": ["bachelor", "bs", "b.s.", "undergraduate", "college degree"],
            "master": ["master", "ms", "m.s.", "graduate degree"],
            "phd": ["phd", "ph.d", "doctorate"]
        }
        
        required_education = None
        for level, keywords in education_keywords.items():
            if any(re.search(r'\b' + re.escape(keyword) + r'\b', job_lower) for keyword in keywords):
                required_education = level
                break
                
        return {
            "required_years": required_years,
            "job_level": job_level,
            "required_education": required_education
        }
        
    def _calculate_experience_match(self, requirements):
        """Calculate match score for experience level"""
        # If no experience requirements specified, assume match
        if not requirements.get("job_level") and not requirements.get("required_years"):
            return 1.0
            
        # Years of experience match
        years_score = 0.0
        if requirements.get("required_years") and self.resume_stats.get("years_of_experience"):
            resume_years = self.resume_stats["years_of_experience"]
            required_years = requirements["required_years"]
            
            if resume_years >= required_years:
                years_score = 1.0
            else:
                # Partial match if close
                years_score = resume_years / required_years if required_years > 0 else 0
                
        # Seniority level match
        level_score = 0.0
        if requirements.get("job_level") and self.resume_stats.get("seniority"):
            level_mapping = {"junior": 1, "mid": 2, "senior": 3}
            
            job_level_value = level_mapping.get(requirements["job_level"], 0)
            resume_level_value = level_mapping.get(self.resume_stats["seniority"], 0)
            
            if resume_level_value >= job_level_value:
                level_score = 1.0
            else:
                # Partial match if close
                level_score = resume_level_value / job_level_value if job_level_value > 0 else 0
        
        # Use the better of the two scores
        return max(years_score, level_score) if (years_score > 0 or level_score > 0) else 0.5
    
    def calculate_similarity(self, job_description):
        """Calculate similarity between resume and job description - enhanced version"""
        processed_job = self._preprocess_text(job_description)
        
        # Get job skills and requirements
        job_skills = self._extract_job_skills(processed_job)
        job_categories = self._categorize_job_skills(job_skills)
        job_requirements = self._extract_job_requirements(job_description)
        
        # Calculate different components of the match
        tfidf_score = self._calculate_tfidf_similarity(processed_job)
        skill_match_score = self._calculate_skill_match(job_skills)
        category_match_score = self._calculate_category_match(job_categories)
        experience_match_score = self._calculate_experience_match(job_requirements)
        
        # Apply weights to get final score
        combined_score = (
            (tfidf_score * self.weights["tfidf"]) +
            (skill_match_score * self.weights["skills"]) +
            (category_match_score * self.weights["categories"])
        )
        
        # Print detailed results
        self._print_match_details(
            job_skills, job_categories, job_requirements,
            tfidf_score, skill_match_score, category_match_score, 
            experience_match_score, combined_score
        )
        
        return combined_score, {
            "tfidf_score": tfidf_score,
            "skill_match": skill_match_score,
            "category_match": category_match_score,
            "experience_match": experience_match_score,
            "combined_score": combined_score,
            "job_skills": job_skills,
            "matching_skills": [skill for skill in job_skills if skill in self.skills],
            "missing_skills": [skill for skill in job_skills if skill not in self.skills],
            "job_requirements": job_requirements
        }
    
    def _calculate_tfidf_similarity(self, processed_job):
        """Calculate TF-IDF similarity between resume and job"""
        try:
            vectorizer = TfidfVectorizer()
            corpus = [self.processed_resume, processed_job]
            tfidf_matrix = vectorizer.fit_transform(corpus)
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except Exception as e:
            logger.error(f"Error calculating TF-IDF similarity: {e}")
            return 0.0
    
    def _print_match_details(self, job_skills, job_categories, job_requirements, 
                           tfidf_score, skill_match_score, category_match_score, 
                           experience_match_score, combined_score):
        """Print detailed matching information"""
        print("\n" + "="*70)
        print("JOB MATCH ANALYSIS")
        print("="*70)
        
        print(f"\nContextual (TF-IDF) Similarity: {tfidf_score * 100:.2f}%")
        print(f"Skill Match Percentage: {skill_match_score * 100:.2f}%")
        print(f"Category Match Percentage: {category_match_score * 100:.2f}%")
        print(f"Experience Match Percentage: {experience_match_score * 100:.2f}%")
        print(f"Combined Matching Score: {combined_score * 100:.2f}%")
        
        print("\nSKILL MATCHING DETAILS:")
        print(f"Job skills requested: {len(job_skills)}")
        matching_skills = [skill for skill in job_skills if skill in self.skills]
        missing_skills = [skill for skill in job_skills if skill not in self.skills]
        print(f"Your matching skills ({len(matching_skills)}): {', '.join(matching_skills)}")
        print(f"Missing skills ({len(missing_skills)}): {', '.join(missing_skills)}")
        
        # Show job categories
        print("\nJOB SKILL CATEGORIES:")
        for category, skills in job_categories.items():
            resume_skills = self.categorized_skills.get(category, [])
            match_percent = len([s for s in skills if s in resume_skills]) / len(skills) * 100 if skills else 0
            status = "✅" if match_percent >= 70 else "⚠️" if match_percent >= 40 else "❌"
            print(f"{status} {category.capitalize()}: {match_percent:.1f}% match ({len([s for s in skills if s in resume_skills])}/{len(skills)})")
        
        # Show experience level match
        print("\nEXPERIENCE REQUIREMENTS:")
        if job_requirements.get("required_years"):
            resume_years = self.resume_stats.get("years_of_experience", "Unknown")
            print(f"Years Required: {job_requirements['required_years']}+ years (You: {resume_years} years)")
            
        if job_requirements.get("job_level"):
            print(f"Level Required: {job_requirements['job_level'].capitalize()} (You: {self.resume_stats.get('seniority', 'Unknown').capitalize()})")
            
        print("\n" + "="*70)
    
    def analyze_job(self, job_title, job_description):
        """Analyze a job description and determine if it's a good match"""
        print(f"\n--- Analysis for {job_title} ---")
        score, details = self.calculate_similarity(job_description)
        
        if score >= self.threshold:
            print("✅ This job matches your profile!")
            self._suggest_improvements(details)
            return True
        else:
            print("❌ This job does not meet your matching threshold.")
            self._suggest_improvements(details)
            return False
    
    def _suggest_improvements(self, match_details):
        """Suggest improvements to increase match score"""
        if len(match_details["missing_skills"]) > 0:
            print("\nSUGGESTED IMPROVEMENTS:")
            # Suggest top 3 missing skills to learn
            top_missing = match_details["missing_skills"][:3]
            if top_missing:
                print("Consider learning these skills to improve your match:")
                for skill in top_missing:
                    print(f"- {skill}")
    
    def batch_analyze_jobs(self, jobs_list):
        """Analyze multiple jobs and rank them by match score"""
        results = []
        
        for job in jobs_list:
            title = job.get("title", "Untitled Position")
            company = job.get("company", "Unknown")
            description = job.get("description", "")
            
            score, details = self.calculate_similarity(description)
            
            results.append({
                "title": title,
                "company": company,
                "score": score,
                "matches_threshold": score >= self.threshold,
                "details": details
            })
        
        # Rank jobs by score
        ranked_jobs = sorted(results, key=lambda x: x["score"], reverse=True)
        
        # Print ranking summary
        print("\n" + "="*70)
        print("JOB RANKINGS")
        print("="*70)
        
        for i, job in enumerate(ranked_jobs, 1):
            match_icon = "✅" if job["matches_threshold"] else "❌"
            print(f"{i}. {match_icon} {job['title']} - Match: {job['score']*100:.2f}%")
            
        return ranked_jobs
    
    def export_report(self, job_title, job_description, output_file=None):
        """Generate a comprehensive job match report and export to JSON"""
        score, details = self.calculate_similarity(job_description)
        
        report = {
            "job_title": job_title,
            "match_score": score * 100,
            "matches_threshold": score >= self.threshold,
            "resume_stats": self.resume_stats,
            "match_details": details,
            "suggested_improvements": [skill for skill in details["missing_skills"][:5]]
        }
        
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2)
                print(f"\nReport saved to {output_file}")
            except Exception as e:
                logger.error(f"Error saving report: {e}")
        
        return report


if __name__ == "__main__":
    # Use the same resume path as original
    resume_path = "C:\\Users\\zarana\\Downloads\\Zarana BMO Resume.pdf"
    agent = EnhancedJobMatchingAgent(resume_path, threshold=0.6)
    
    # Example job descriptions to test against
    test_jobs = [
        {
            "title": "Full Stack Developer",
            "company": "Tech Solutions Inc.",
            "description": """
                Looking for Full Stack Developer with experience in React.js, Spring Boot, AWS, Microservices, RESTful APIs, Docker, and Kubernetes.
                
                Requirements:
                - 2+ years of experience in full stack development
                - Strong knowledge of React.js and Spring Boot
                - Experience with AWS and containerization
                - Bachelor's degree in Computer Science or related field
            """
        },
        {
            "title": "Frontend Developer",
            "company": "Creative Web Agency",
            "description": """
                We are hiring a Frontend Developer experienced in React.js, JavaScript, TypeScript, HTML, CSS, Redux, Docker, AWS, and GraphQL.
                
                Requirements:
                - 2+ years of experience with React.js and modern JavaScript
                - Strong knowledge of HTML5, CSS3 and responsive design
                - Experience with state management using Redux
                - Ability to work in an Agile environment
            """
        },
        {
            "title": "Backend Developer",
            "company": "Enterprise Solutions",
            "description": """
                Looking for Backend Developer experienced in Java Spring Boot, Microservices architecture, PostgreSQL, MongoDB, Docker, Kubernetes, AWS cloud solutions, and CI/CD pipelines.
                
                Key Requirements:
                - 4+ years of experience in backend development
                - Strong knowledge of Java and Spring Boot framework
                - Experience designing and implementing microservices
                - Familiarity with both SQL and NoSQL databases
                - DevOps experience is a plus
            """
        },
        {
            "title": "Senior Full Stack Developer",
            "company": "TechInnovate Solutions",
            "description": """
                We're looking for a talented Senior Full Stack Developer to join our growing team. You'll be responsible for building and maintaining our cloud-based enterprise applications, working across the entire stack from frontend to backend.

                Key Responsibilities:
                • Design, develop, and deploy scalable web applications using React.js and Node.js
                • Write clean, maintainable, and efficient code in TypeScript
                • Work with our MongoDB and PostgreSQL databases to optimize queries and data structures
                • Implement RESTful APIs and GraphQL endpoints
                • Collaborate with the DevOps team to deploy applications using Docker and Kubernetes
                • Participate in code reviews and mentor junior developers
                • Implement CI/CD pipelines using Jenkins
                • Contribute to architectural decisions and technology selection

                Requirements:
                • 5+ years of experience in full stack development
                • Proficiency in React.js, Node.js, and TypeScript
                • Experience with MongoDB and PostgreSQL
                • Familiarity with Docker, Kubernetes, and AWS or Azure cloud services
                • Knowledge of microservices architecture and RESTful API design
                • Experience with CI/CD pipelines and Git version control
                • Strong problem-solving skills and attention to detail
                • Excellent communication and collaboration skills
            """
        },
        {
            "title": "DevOps Engineer",
            "company": "Cloud Innovations",
            "description": """
                We're seeking a DevOps Engineer to help us build and maintain our cloud infrastructure and CI/CD pipelines.
                
                Responsibilities:
                • Design and implement CI/CD pipelines using Jenkins, GitHub Actions, or similar tools
                • Manage cloud infrastructure on AWS using Terraform or CloudFormation
                • Containerize applications using Docker and orchestrate with Kubernetes
                • Implement monitoring and alerting solutions
                • Automate deployment and infrastructure management
                
                Requirements:
                • 3+ years of experience in DevOps or similar role
                • Strong experience with AWS services
                • Experience with containerization and orchestration
                • Knowledge of infrastructure as code and configuration management
                • Experience with monitoring and logging tools
            """
        }
    ]
    
    # Run batch analysis
    ranked_jobs = agent.batch_analyze_jobs(test_jobs)
    
    # Generate a detailed report for the top match
    if ranked_jobs:
        top_job = ranked_jobs[0]
        top_job_data = next((j for j in test_jobs if j["title"] == top_job["title"]), None)
        
        if top_job_data:
            agent.export_report(
                top_job["title"], 
                top_job_data["description"],
                "top_job_match_report.json"
            )