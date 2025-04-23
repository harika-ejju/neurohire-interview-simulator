import os
import re
import json
import requests
import PyPDF2
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Any, Optional

# Load environment variables
load_dotenv()
GEMINI_API_URL = os.getenv("GEMINI_API_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# PDF processing functions
def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract text content from an uploaded PDF file.
    
    Args:
        pdf_file: The uploaded PDF file object
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def clean_resume_text(text: str) -> str:
    """
    Clean and preprocess the extracted resume text.
    
    Args:
        text: Raw text extracted from resume
        
    Returns:
        str: Cleaned text
    """
    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_resume_sections(text: str) -> Dict[str, str]:
    """
    Attempt to parse resume into sections.
    
    Args:
        text: Cleaned resume text
        
    Returns:
        Dict[str, str]: Dictionary with resume sections
    """
    # This is a simple implementation - a more robust parser would be needed for production
    sections = {}
    
    # Common section headers in resumes
    section_headers = [
        "EDUCATION", "WORK EXPERIENCE", "SKILLS", "PROJECTS",
        "CERTIFICATIONS", "PUBLICATIONS", "CONTACT"
    ]
    
    # Simple regex-based section parsing
    current_section = "OVERVIEW"
    sections[current_section] = ""
    
    lines = text.split("\n")
    for line in lines:
        line_upper = line.upper()
        
        # Check if line contains a section header
        found_section = False
        for header in section_headers:
            if header in line_upper:
                current_section = header
                sections[current_section] = ""
                found_section = True
                break
                
        if not found_section:
            sections[current_section] += line + "\n"
    
    return sections

# Gemini API interaction functions
def query_gemini_api(prompt: str, model_context: Optional[str] = None) -> Dict[str, Any]:
    """
    Send a prompt to the Gemini API and get the response.
    
    Args:
        prompt: The prompt text to send to the API
        model_context: Optional context (like resume text) to include
        
    Returns:
        Dict[str, Any]: JSON response from the API
    """
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }
    
    # Build the API request
    content_parts = []
    
    # Add context if provided
    if model_context:
        content_parts.append({
            "text": f"CONTEXT: {model_context}\n\n"
        })
    
    # Add the main prompt
    content_parts.append({
        "text": prompt
    })
    
    payload = {
        "contents": [{
            "parts": content_parts
        }],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 800
        }
    }
    
    try:
        response = requests.post(
            GEMINI_API_URL,
            headers=headers,
            json=payload
        )
        return response.json()
    except Exception as e:
        print(f"Error querying Gemini API: {e}")
        return {"error": str(e)}

def process_gemini_response(response: Dict[str, Any]) -> str:
    """
    Process the raw Gemini API response and extract the text.
    
    Args:
        response: Raw JSON response from Gemini API
        
    Returns:
        str: Extracted text from the response
    """
    try:
        if "error" in response:
            return f"Error: {response['error']}"
        
        if "candidates" in response and len(response["candidates"]) > 0:
            candidate = response["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if len(parts) > 0 and "text" in parts[0]:
                    return parts[0]["text"]
        
        return "No valid response found in the API result."
    except Exception as e:
        print(f"Error processing Gemini response: {e}")
        return f"Error processing response: {str(e)}"

# Interview and evaluation functions
def generate_technical_questions(resume_text: str, n: int = 3) -> List[str]:
    """
    Generate technical interview questions based on the resume content.
    
    Args:
        resume_text: The extracted and cleaned resume text
        n: Number of questions to generate
        
    Returns:
        List[str]: List of generated questions
    """
    prompt = f"""
    Based on the following resume, generate {n} challenging technical interview questions that 
    assess the candidate's knowledge and experience. The questions should be directly related 
    to the technologies and skills mentioned in the resume.
    
    Make the questions specific, technical in nature, and designed to test depth of knowledge.
    Format the output as a numbered list of questions only.
    """
    
    response = query_gemini_api(prompt, resume_text)
    response_text = process_gemini_response(response)
    
    # Extract questions using regex (looking for numbered list items)
    questions = re.findall(r'\d+\.\s+(.*?)(?=\n\d+\.|\Z)', response_text, re.DOTALL)
    
    # Clean up whitespace
    questions = [q.strip() for q in questions]
    
    # If we couldn't parse questions properly, just split by newlines as fallback
    if len(questions) == 0:
        questions = [line.strip() for line in response_text.split('\n') if line.strip()]
    
    return questions[:n]  # Ensure we return at most n questions

def evaluate_answer(question: str, answer: str, resume_text: str) -> Tuple[int, str, Dict[str, Any]]:
    """
    Evaluate a user's answer to an interview question.
    
    Args:
        question: The interview question
        answer: The user's answer
        resume_text: The resume text for context
        
    Returns:
        Tuple[int, str, Dict]: Score (0-10), feedback explanation, and full evaluation data
    """
    # Classify the question type to provide more relevant evaluation
    question_type = classify_question_type(question)
    
    prompt = f"""
    You are an expert technical interviewer evaluating a candidate's response. 
    
    QUESTION TYPE: {question_type}
    
    Evaluate the following answer to a technical interview question. Rate it on a scale of 1-10 
    based on accuracy, completeness, depth of knowledge, and relevance. Provide specific, 
    actionable feedback that reflects the strengths and weaknesses of this particular answer.
    
    Question: {question}
    
    Answer: {answer}
    
    Be specific in your feedback. Mention:
    1. What was good about the answer
    2. What specific knowledge gaps were identified
    3. What the candidate should study to improve
    4. How the answer relates to industry best practices
    
    Your evaluation MUST be in the following JSON format with no extra text before or after:
    {{
        "score": <integer between 1 and 10>,
        "feedback": "<detailed constructive feedback explaining the score>",
        "strengths": ["<specific strength 1>", "<specific strength 2>", ...],
        "areas_to_improve": ["<specific area 1>", "<specific area 2>", ...],
        "suggested_resources": ["<relevant resource or topic to study>", ...]
    }}
    """
    
    response = query_gemini_api(prompt, resume_text)
    response_text = process_gemini_response(response)
    
    # Enhanced JSON extraction
    try:
        # Try to find a JSON block with various patterns
        json_patterns = [
            r'```json\s*({.*?})\s*```',  # JSON in code blocks
            r'({(?:"score"|"feedback"|"strengths"|"areas_to_improve"|"suggested_resources")[^}]+(?:"score"|"feedback"|"strengths"|"areas_to_improve"|"suggested_resources")[^}]+})',  # Partial matching of expected fields
            r'({.*})',  # Any JSON-like structure as fallback
        ]
        
        found_json = None
        for pattern in json_patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                found_json = match.group(1).strip()
                break
        
        if found_json:
            # Clean the JSON string - replace single quotes with double quotes if needed
            found_json = found_json.replace("'", "\"")
            # Remove any trailing commas before closing brackets (common JSON parsing issue)
            found_json = re.sub(r',\s*}', '}', found_json)
            found_json = re.sub(r',\s*]', ']', found_json)
            
            evaluation = json.loads(found_json)
            
            # Ensure all expected fields are present
            score = int(evaluation.get("score", 5))
            # Clamp score between 1 and 10
            score = max(1, min(10, score))
            
            feedback = evaluation.get("feedback", "")
            if not feedback:
                feedback = "The answer shows some understanding, but could be more comprehensive."
            
            # Ensure the evaluation has all required fields
            if "strengths" not in evaluation:
                evaluation["strengths"] = []
            if "areas_to_improve" not in evaluation:
                evaluation["areas_to_improve"] = []
            if "suggested_resources" not in evaluation:
                evaluation["suggested_resources"] = []
                
            return score, feedback, evaluation
        else:
            # Construct a basic evaluation from the text if JSON parsing fails
            return extract_evaluation_from_text(response_text, question_type)
    except Exception as e:
        print(f"Error parsing evaluation response: {e}")
        print(f"Response text: {response_text}")
        return construct_fallback_evaluation(question, answer, question_type)

def classify_question_type(question: str) -> str:
    """
    Classify the type of technical question to provide more relevant evaluation criteria.
    
    Args:
        question: The interview question
        
    Returns:
        str: The classified question type
    """
    question_lower = question.lower()
    
    if any(keyword in question_lower for keyword in ["algorithm", "complexity", "big o", "time complexity", "space complexity"]):
        return "ALGORITHM"
    elif any(keyword in question_lower for keyword in ["design", "architecture", "system", "scale"]):
        return "SYSTEM_DESIGN"
    elif any(keyword in question_lower for keyword in ["database", "sql", "query", "index", "nosql"]):
        return "DATABASE"
    elif any(keyword in question_lower for keyword in ["react", "angular", "vue", "dom", "component", "frontend", "ui"]):
        return "FRONTEND"
    elif any(keyword in question_lower for keyword in ["api", "rest", "graphql", "endpoint", "http", "backend"]):
        return "BACKEND"
    elif any(keyword in question_lower for keyword in ["devops", "ci/cd", "pipeline", "deployment", "kubernetes", "docker"]):
        return "DEVOPS"
    elif any(keyword in question_lower for keyword in ["machine learning", "ml", "ai", "model", "training", "neural"]):
        return "MACHINE_LEARNING"
    else:
        return "GENERAL_TECHNICAL"

def extract_evaluation_from_text(text: str, question_type: str) -> Tuple[int, str, Dict[str, Any]]:
    """
    Extract evaluation components from unstructured text when JSON parsing fails.
    
    Args:
        text: The response text
        question_type: The type of question being evaluated
        
    Returns:
        Tuple[int, str, Dict]: Score, feedback, and structured evaluation
    """
    # Look for score patterns
    score_match = re.search(r'(?:score|rating|grade)[^\d]*?(\d+)[^\d]*?(?:\/|\s*out\s*of\s*)?\s*10', text, re.IGNORECASE)
    score = int(score_match.group(1)) if score_match else 5
    score = max(1, min(10, score))  # Clamp between 1 and 10
    
    # Extract feedback - look for specific sections
    feedback_patterns = [
        r'(?:feedback|evaluation|assessment):\s*([^\n]+(?:\n[^\n]+)*)',
        r'(?:strengths|positives|pros):\s*([^\n]+(?:\n[^\n]+)*)',
        r'(?:weaknesses|areas\s+to\s+improve|cons):\s*([^\n]+(?:\n[^\n]+)*)'
    ]
    
    feedback_parts = []
    for pattern in feedback_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            feedback_parts.append(match.group(1).strip())
    
    # Combine extracted parts or use a default
    feedback = " ".join(feedback_parts) if feedback_parts else "Based on your answer, there's room for improvement in clarity and technical depth."
    
    # Extract bullet points for strengths and areas to improve
    strengths = re.findall(r'(?:strengths|positives|pros)[^\n]*(?:\n|\:)\s*(?:[-•*]\s*([^\n]+))+', text, re.IGNORECASE | re.MULTILINE)
    areas_to_improve = re.findall(r'(?:weaknesses|areas\s+to\s+improve|cons)[^\n]*(?:\n|\:)\s*(?:[-•*]\s*([^\n]+))+', text, re.IGNORECASE | re.MULTILINE)
    
    # Create structured evaluation
    evaluation = {
        "score": score,
        "feedback": feedback,
        "strengths": strengths if strengths else ["Showed some understanding of the topic"],
        "areas_to_improve": areas_to_improve if areas_to_improve else ["Could provide more comprehensive explanation"],
        "suggested_resources": get_default_resources(question_type)
    }
    
    return score, feedback, evaluation

def construct_fallback_evaluation(question: str, answer: str, question_type: str) -> Tuple[int, str, Dict[str, Any]]:
    """
    Construct a fallback evaluation when all parsing methods fail.
    
    Args:
        question: The interview question
        answer: The user's answer
        question_type: The type of question
        
    Returns:
        Tuple[int, str, Dict]: Score, feedback, and structured evaluation
    """
    # Determine score based on answer length and complexity
    words = answer.split()
    score = 5  # Default middle score
    
    # Adjust score based on answer length (very basic heuristic)
    if len(words) < 20:
        score = 3  # Very short answers are penalized
    elif len(words) > 100:
        score = 6  # Longer answers get a small boost
        
    # Check for technical terms based on question type
    technical_terms = {
        "ALGORITHM": ["complexity", "big o", "time", "space", "efficiency", "optimal"],
        "SYSTEM_DESIGN": ["scalability", "reliability", "availability", "performance", "latency"],
        "DATABASE": ["index", "query", "schema", "transaction", "normalize"],
        "FRONTEND": ["component", "state", "render", "event", "lifecycle"],
        "BACKEND": ["api", "endpoint", "request", "response", "middleware"],
        "DEVOPS": ["pipeline", "deployment", "container", "infrastructure", "monitoring"],
        "MACHINE_LEARNING": ["model", "training", "features", "accuracy", "validation"],
        "GENERAL_TECHNICAL": ["implementation", "pattern", "practice", "solution", "approach"]
    }
    
    relevant_terms = technical_terms.get(question_type, technical_terms["GENERAL_TECHNICAL"])
    term_count = sum(1 for term in relevant_terms if term.lower() in answer.lower())
    
    # Adjust score based on presence of technical terms
    score += min(2, term_count)  # Boost score by up to 2 points based on technical terms
    score = max(1, min(10, score))  # Ensure score is between 1 and 10
    
    # Generate generic but somewhat specific feedback
    feedback = f"Your answer to this {question_type.lower().replace('_', ' ')} question demonstrates some understanding, but could be more comprehensive and technically detailed."
    
    evaluation = {
        "score": score,
        "feedback": feedback,
        "strengths": ["Attempted to address the question"],
        "areas_to_improve": ["Provide more specific technical details", "Structure your answer more clearly"],
        "suggested_resources": get_default_resources(question_type)
    }
    
    return score, feedback, evaluation

def get_default_resources(question_type: str) -> List[str]:
    """
    Get default learning resources based on question type.
    
    Args:
        question_type: The type of question
        
    Returns:
        List[str]: List of suggested learning resources
    """
    resources = {
        "ALGORITHM": ["Introduction to Algorithms by CLRS", "AlgoExpert", "LeetCode practice problems"],
        "SYSTEM_DESIGN": ["System Design Interview by Alex Xu", "Designing Data-Intensive Applications by Martin Kleppmann"],
        "DATABASE": ["Database System Concepts by Silberschatz et al.", "MongoDB University courses"],
        "FRONTEND": ["React documentation", "JavaScript.info", "Frontend Masters courses"],
        "BACKEND": ["RESTful API Design Best Practices", "Node.js documentation", "Spring Framework documentation"],
        "DEVOPS": ["The DevOps Handbook", "Docker documentation", "Kubernetes Learning Path"],
        "MACHINE_LEARNING": ["Hands-On Machine Learning with Scikit-Learn and TensorFlow", "Fast.ai courses"],
        "GENERAL_TECHNICAL": ["Clean Code by Robert C. Martin", "Reddit r/programming", "Medium engineering blogs"]
    }
    
    return resources.get(question_type, resources["GENERAL_TECHNICAL"])

def generate_performance_summary(questions: List[str], answers: List[str], scores: List[int], 
                               feedbacks: List[str], resume_text: str) -> Dict[str, Any]:
    """
    Generate a comprehensive performance summary after all questions.
    
    Args:
        questions: List of questions asked
        answers: List of user answers
        scores: List of scores for each answer
        feedbacks: List of feedback for each answer
        resume_text: The resume text for context
        
    Returns:
        Dict[str, Any]: Performance summary with overall score, strengths, 
                        weaknesses, and development areas
    """
    # Prepare the QA pairs with scores for the prompt
    qa_summary = ""
    for i, (q, a, s, f) in enumerate(zip(questions, answers, scores, feedbacks)):
        qa_summary += f"Q{i+1}: {q}\nA{i+1}: {a}\nScore: {s}/10\nFeedback: {f}\n\n"
    
    prompt = f"""
    Based on the candidate's performance in the technical interview questions and their resume, 
    provide a comprehensive evaluation summary. Include:

    1. Overall score (average of individual scores)
    2. Key strengths demonstrated
    3. Areas needing improvement
    4. Specific technologies or skills to develop further
    5. General recommendation for the candidate

    Format the output as JSON:
    {{
        "overall_score": <average_score>,
        "strengths": ["strength1", "strength2", ...],
        "weaknesses": ["weakness1", "weakness2", ...],
        "development_areas": ["area1", "area2", ...],
        "recommendation": "summary recommendation"
    }}

    Interview Q&A Summary:
    {qa_summary}
    """
    
    response = query_gemini_api(prompt, resume_text)
    response_text = process_gemini_response(response)
    
    # Try to extract JSON from the response
    try:
        # Find JSON pattern in the response text
        json_match = re.search(r'({.*})', response_text.replace('\n', ''), re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            summary = json.loads(json_str)
        else:
            # Calculate average score manually as fallback
            avg_score = sum(scores) / len(scores) if scores else 0
            summary = {
                "overall_score": round(avg_score, 1),
                "strengths": ["Unable to parse strengths from model response"],
                "weaknesses": ["Unable to parse weaknesses from model response"],
                "development_areas": ["Unable to parse development areas from model response"],
                "recommendation": "Please review individual feedback for specific recommendations."
            }
    except Exception as e:
        print(f"Error parsing summary response: {e}")
        # Create fallback summary with at least the average score
        avg_score = sum(scores) / len(scores) if scores else 0
        summary = {
            "overall_score": round(avg_score, 1),
            "strengths": ["Error parsing model response"],
            "weaknesses": ["Error parsing model response"],
            "development_areas": ["Error parsing model response"],
            "recommendation": "An error occurred while generating the summary. Please review individual feedback."
        }
    
    return summary

