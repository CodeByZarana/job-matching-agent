import os
import logging
import re
# We'll add these imports for PDF and DOCX processing
import PyPDF2
import docx2txt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='job_agent.log'
)
logger = logging.getLogger('job_agent')

class JobMatchingAgent:
    def __init__(self, resume_path, threshold=0.6):
        """
        Initialize the job matching agent
        
        Args:
            resume_path (str): Path to resume file (PDF or DOCX)
            threshold (float): Minimum similarity score to consider a match (0-1)
        """
        self.resume_path = resume_path
        self.threshold = threshold
        # Now let's extract the text from the resume
        self.resume_text = self._extract_text_from_resume()
        print(f"Agent initialized with resume at {resume_path}")
        print(f"Extracted {len(self.resume_text)} characters from resume")
        
    def _extract_text_from_resume(self):
        """Extract text from resume file (PDF or DOCX)"""
        file_extension = os.path.splitext(self.resume_path)[1].lower()
        text = ""
        
        try:
            if file_extension == '.pdf':
                with open(self.resume_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(pdf_reader.pages)):
                        text += pdf_reader.pages[page_num].extract_text()
            elif file_extension == '.docx':
                text = docx2txt.process(self.resume_path)
            else:
                with open(self.resume_path, 'r', encoding='utf-8') as file:
                    text = file.read()
            
            logger.info(f"Successfully extracted text from resume: {self.resume_path}")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from resume: {e}")
            return ""
    
    def display_resume_preview(self):
        """Display a preview of the extracted resume text"""
        if not self.resume_text:
            return "No text was extracted from the resume."
        
        # Show the first 500 characters as a preview
        preview = self.resume_text[:500]
        return f"Resume preview:\n{preview}...\n\n(Total length: {len(self.resume_text)} characters)"
        
if __name__ == "__main__":
    agent = JobMatchingAgent(
        resume_path="C:\\Users\\zarana\\Downloads\\Zarana BMO Resume.pdf",  # Make sure this path is correct
        threshold=0.6
    )
    
    # Display a preview of the extracted resume text
    print("\nResume Text Preview:")
    print(agent.display_resume_preview())