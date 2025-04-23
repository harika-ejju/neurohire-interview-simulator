import streamlit as st
import os
import requests
import json
import PyPDF2
import tempfile
import pandas as pd
import altair as alt
import plotly.graph_objects as go
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# API Config
GEMINI_API_URL = os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", )
# App configuration
st.set_page_config(page_title="Resume Interview Simulator", layout="wide")

# Custom CSS for better UI
CODING_QUESTIONS = {
    "Python": [
        "Implement a binary search algorithm",
        "Write a decorator to measure function execution time",
        "Create a generator for Fibonacci sequence",
        "Handle file I/O with context managers",
        "Implement a class with polymorphism"
    ],
    "Java": [
        "Implement a thread-safe singleton pattern",
        "Create a REST API using Spring Boot",
        "Handle checked exceptions properly",
        "Implement a custom comparator for sorting",
        "Demonstrate inheritance with abstract classes"
    ],
    "JavaScript": [
        "Implement promise chaining with error handling",
        "Create a React component with hooks",
        "Implement debounce function",
        "Handle async/await with API calls",
        "Demonstrate closure usage"
    ],
    "C++": [
        "Implement move semantics for a custom class",
        "Create template functions for generic programming",
        "Handle memory management with smart pointers",
        "Implement operator overloading",
        "Create a multi-threaded application"
    ]
}
st.markdown("""
<style>
    .chat-container {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        background-color: #f9f9f9;
        overflow-y: auto;
    }
    .user-message {
        background-color: #e6f7ff;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .bot-message {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .evaluation {
        background-color: #f5f5dc;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        border-left: 3px solid #ffd700;
    }
    .summary {
        background-color: #e6ffe6;
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
        border-left: 3px solid #28a745;
    }
    .contact-form {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .contact-info {
        background-color: #e9ecef;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = ""
if 'interview_started' not in st.session_state:
    st.session_state.interview_started = False
if 'interview_completed' not in st.session_state:
    st.session_state.interview_completed = False
if 'evaluations' not in st.session_state:
    st.session_state.evaluations = []
if 'total_score' not in st.session_state:
    st.session_state.total_score = 0
if 'num_questions' not in st.session_state:
    st.session_state.num_questions = 5
if 'current_page' not in st.session_state:
    st.session_state.current_page = "interview"  # Default to interview page

def generate_questions_from_resume(resume_text, num_questions=5):
    """Generate interview questions based on resume content using Gemini API"""
    try:
        prompt = f"""
        You are a technical interviewer preparing for an interview with a candidate. 
        
        I have the candidate's resume text below. Based on their experience, skills, and background,
        generate {num_questions} relevant technical interview questions that will help evaluate their knowledge and expertise.
        
        The questions should be specific to their background and technical skills mentioned in the resume.
        Include a mix of technical knowledge, problem-solving, and experience-based questions.
        
        Resume: {resume_text[:3000]}
        
        Output exactly {num_questions} questions as a JSON array of strings:
        ["Question 1", "Question 2", ..., "Question {num_questions}"]
        """
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        }
        
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        response = requests.post(GEMINI_API_URL, headers=headers, json=data)
        
        if response.status_code == 200:
            response_data = response.json()
            text_response = response_data['candidates'][0]['content']['parts'][0]['text']
            
            # Extract JSON array from response
            if '```json' in text_response:
                json_str = text_response.split('```json')[1].split('```')[0].strip()
            elif '```' in text_response:
                json_str = text_response.split('```')[1].strip()
            else:
                json_str = text_response
            
            # Clean up the JSON string if needed
            json_str = json_str.replace('`', '')
            
            # Parse the JSON to get the questions
            questions = json.loads(json_str)
            
            # Ensure we have exactly the requested number of questions
            if len(questions) > num_questions:
                questions = questions[:num_questions]
            elif len(questions) < num_questions:
                # Add generic questions if needed
                generic_questions = [
                    "Tell me about your background in software development.",
                    "Explain a challenging project you worked on and how you overcame obstacles.",
                    "How do you approach debugging a complex issue?",
                    "How do you stay updated with the latest technological trends?",
                    "Where do you see yourself in 5 years in terms of technical expertise?",
                    "What development methodologies are you familiar with?",
                    "Describe your experience with cloud platforms.",
                    "How do you handle code reviews?",
                    "What's your approach to continuous learning?",
                    "How do you prioritize tasks when working on multiple projects?",
                    "Describe your experience with performance optimization.",
                    "How do you ensure your code is secure?",
                    "What's your experience with containerization technologies?",
                    "How do you document your code?",
                    "Describe a time when you had to learn a new technology quickly."
                ]
                questions.extend(generic_questions[:(num_questions-len(questions))])
            
            return questions
        else:
            st.error(f"Failed to generate questions: API returned {response.status_code}")
            default_questions = [
                "Tell me about your background in software development.",
                "Explain your experience with Python and related frameworks.",
                "Describe a challenging project you worked on and how you overcame obstacles.",
                "How do you approach debugging a complex issue in a production environment?",
                "What's your experience with database systems and SQL?",
                "How do you stay updated with the latest technological trends?",
                "Explain your understanding of RESTful APIs and microservices.",
                "Describe your experience with version control systems like Git.",
                "How do you approach testing and ensuring code quality?",
                "Where do you see yourself in 5 years in terms of technical expertise?",
                "What development methodologies are you most comfortable with?",
                "How do you handle requirements that change during development?",
                "Tell me about your experience with cloud platforms.",
                "How do you ensure your code is maintainable?",
                "Describe your approach to code reviews.",
                "What strategies do you use for debugging complex issues?",
                "How do you stay current with technology trends?",
                "Describe your experience with performance optimization.",
                "How do you approach learning a new programming language or framework?",
                "What's your experience with containerization and orchestration?"
            ]
            return default_questions[:num_questions]
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        default_questions = [
            "Tell me about your background in software development.",
            "Explain your experience with Python and related frameworks.",
            "Describe a challenging project you worked on and how you overcame obstacles.",
            "How do you approach debugging a complex issue in a production environment?",
            "What's your experience with database systems and SQL?",
            "How do you stay updated with the latest technological trends?",
            "Explain your understanding of RESTful APIs and microservices.",
            "Describe your experience with version control systems like Git.",
            "How do you approach testing and ensuring code quality?",
            "Where do you see yourself in 5 years in terms of technical expertise?",
            "What development methodologies are you most comfortable with?",
            "How do you handle requirements that change during development?",
            "Tell me about your experience with cloud platforms.",
            "How do you ensure your code is maintainable?",
            "Describe your approach to code reviews.",
            "What strategies do you use for debugging complex issues?",
            "How do you stay current with technology trends?",
            "Describe your experience with performance optimization.",
            "How do you approach learning a new programming language or framework?",
            "What's your experience with containerization and orchestration?"
        ]
        return default_questions[:num_questions]

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    text = ""
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(pdf_file.getvalue())
        temp_file_path = temp_file.name
    
    with open(temp_file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    os.unlink(temp_file_path)
    return text

def evaluate_answer(question, answer, resume_text):
    """Use Gemini API to evaluate the answer"""
    try:
        prompt = f"""
        You are a technical interviewer evaluating a candidate's response.
        
        Resume context: {resume_text[:2000]}
        
        Question: {question}
        
        Candidate's Answer: {answer}
        
        Evaluate the answer and provide:
        1. A score out of 10
        2. Specific feedback on strengths
        3. Areas that could be improved
        4. Suggestions for further development
        
        Respond in JSON format:
        {{
            "score": [1-10],
            "feedback": "your detailed feedback",
            "strengths": "what was good about the answer",
            "improvements": "what could be improved"
        }}
        """
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        }
        
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        response = requests.post(GEMINI_API_URL, headers=headers, json=data)
        
        if response.status_code == 200:
            response_data = response.json()
            text_response = response_data['candidates'][0]['content']['parts'][0]['text']
            
            try:
                # Extract JSON from response (handling potential text before/after JSON)
                json_str = text_response
                if '```json' in text_response:
                    json_str = text_response.split('```json')[1].split('```')[0].strip()
                elif '```' in text_response:
                    json_str = text_response.split('```')[1].strip()
                
                evaluation = json.loads(json_str)
                return evaluation
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "score": 5,
                    "feedback": "Unable to parse evaluation. " + text_response[:200] + "...",
                    "strengths": "N/A",
                    "improvements": "N/A"
                }
        else:
            return {
                "score": 5,
                "feedback": f"API Error: {response.status_code}",
                "strengths": "N/A",
                "improvements": "N/A"
            }
    except Exception as e:
        return {
            "score": 5,
            "feedback": f"Error: {str(e)}",
            "strengths": "N/A",
            "improvements": "N/A"
        }

def display_messages():
    """Display chat messages in UI using st.chat_message"""
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            with st.chat_message("user"):
                st.write(content)
        elif role == "assistant":
            with st.chat_message("assistant"):
                st.write("Interviewer: " + content)
        elif role == "evaluation":
            with st.chat_message("assistant"):
                st.write("Evaluation: " + content)

def generate_interview_summary():
    """Generate a summary of the interview performance"""
    num_questions = len(st.session_state.evaluations)
    average_score = st.session_state.total_score / num_questions if num_questions > 0 else 0
    
    # Return structured data instead of HTML
    summary_data = {
        "total_score": st.session_state.total_score,
        "max_score": num_questions * 10,
        "average_score": average_score,
        "question_reviews": [],
        "skill_areas": {}  # For tracking skill areas for radar chart
    }
    
    for i, eval in enumerate(st.session_state.evaluations):
        question = st.session_state.interview_questions[i] if i < len(st.session_state.interview_questions) else f"Question {i+1}"
        summary_data["question_reviews"].append({
            "question_number": i+1,
            "question_text": question,
            "score": eval.get('score', 0),
            "strengths": eval.get('strengths', 'N/A'),
            "improvements": eval.get('improvements', 'N/A')
        })
    
    return summary_data

def next_question():
    """Proceed to the next interview question"""
    if st.session_state.current_question < len(st.session_state.interview_questions):
        question = st.session_state.interview_questions[st.session_state.current_question]
        # Add question number to the displayed question
        question_with_number = f"Question {st.session_state.current_question + 1}/{st.session_state.num_questions}: {question}"
        st.session_state.messages.append({"role": "assistant", "content": question_with_number})
        st.session_state.current_question += 1
    else:
        if not st.session_state.interview_completed:
            # Just mark the interview as completed, summary will be displayed outside chat
            st.session_state.interview_completed = True
            # Generate the summary and store it in session state
            st.session_state.summary_data = generate_interview_summary()

def start_interview():
    """Start the interview process"""
    st.session_state.interview_started = True
    st.session_state.messages = []
    st.session_state.current_question = 0
    st.session_state.evaluations = []
    st.session_state.total_score = 0
    st.session_state.interview_completed = False
    
    # Generate questions based on resume and selected number of questions
    with st.spinner("Analyzing your resume and generating personalized questions..."):
        st.session_state.interview_questions = generate_questions_from_resume(
            st.session_state.resume_text, 
            st.session_state.num_questions
        )
    
    welcome_message = "Welcome to your technical interview! I'll ask you personalized questions based on your resume and evaluate your responses. Let's begin!"
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})
    next_question()

def reset_interview():
    """Reset interview state and start a new interview"""
    st.session_state.interview_started = False
    start_interview()

def update_interview_settings():
    """Update the number of questions in the current session without restarting the interview"""
    # Get the original number of questions
    original_num = len(st.session_state.interview_questions)
    new_num = st.session_state.num_questions
    
    # If increasing the number of questions
    if new_num > original_num:
        # Generate additional questions
        with st.spinner("Generating additional questions..."):
            additional_questions = generate_questions_from_resume(
                st.session_state.resume_text,
                new_num - original_num
            )
            st.session_state.interview_questions.extend(additional_questions)
    
    # If decreasing the number of questions
    elif new_num < original_num:
        # Trim the questions list
        st.session_state.interview_questions = st.session_state.interview_questions[:new_num]
        
        # Adjust current_question if needed
        if st.session_state.current_question > new_num:
            st.session_state.current_question = new_num
            
            # If interview was already completed, mark it as not completed
            if st.session_state.interview_completed:
                st.session_state.interview_completed = False

    # Show success message
    st.success(f"Number of questions updated to {new_num}")

def create_score_distribution_pie_chart(summary_data):
    """Create a pie chart showing score distribution categories"""
    # Calculate score distribution
    score_distribution = {
        "Excellent (8-10)": 0,
        "Good (6-7.9)": 0, 
        "Average (4-5.9)": 0,
        "Needs Improvement (0-3.9)": 0
    }
    
    for review in summary_data["question_reviews"]:
        score = review['score']
        if score >= 8:
            score_distribution["Excellent (8-10)"] += 1
        elif score >= 6:
            score_distribution["Good (6-7.9)"] += 1
        elif score >= 4:
            score_distribution["Average (4-5.9)"] += 1
        else:
            score_distribution["Needs Improvement (0-3.9)"] += 1
    
    # Create pie chart data
    pie_data = pd.DataFrame({
        'Category': list(score_distribution.keys()),
        'Count': list(score_distribution.values())
    })
    
    # Define colors for each category
    colors = ['#4CAF50', '#8BC34A', '#FFC107', '#F44336']
    
    # Create the pie chart
    fig = go.Figure(data=[go.Pie(
        labels=pie_data['Category'],
        values=pie_data['Count'],
        hole=0.3,  # Creates a donut chart
        marker=dict(colors=colors),
        textinfo='label+percent',
        textposition='outside',
        pull=[0.1 if cat == "Excellent (8-10)" else 0 for cat in pie_data['Category']],  # Pull out the "Excellent" slice
        hoverinfo='label+percent+value',
        showlegend=True
    )])
    
    return fig

def show_contact_page():
    """Display the contact page with proper Web3Forms response handling"""
    st.title("Contact Us")
    
    with st.container():
        st.markdown("""
        ## Have questions or feedback?
        We'd love to hear from you! Please fill out the form below or reach out to us directly.
        """)
        
        # Web3Forms configuration
        WEB3FORMS_ACCESS_KEY = "1cbc4557-1623-421b-aa23-8a40bb5be0e0"
        WEB3FORMS_ENDPOINT = "https://api.web3forms.com/submit"
        
        # Contact form
        with st.form("contact_form"):
            st.markdown('<div class="contact-form">', unsafe_allow_html=True)
            
            # Hidden fields for Web3Forms
            st.markdown(f"""
                <input type="hidden" name="access_key" value="{WEB3FORMS_ACCESS_KEY}">
                <input type="hidden" name="redirect" value="false">
                <input type="checkbox" name="botcheck" style="display: none !important" value="">
            """, unsafe_allow_html=True)
            
            # Form fields
            name = st.text_input("Your Name*", placeholder="John Doe", key="name")
            email = st.text_input("Your Email*", placeholder="john@example.com", key="email")
            subject = st.selectbox("Subject", [
                "General Inquiry", 
                "Technical Support", 
                "Feedback", 
                "Feature Request",
                "Partnership Opportunities"
            ], key="subject")
            message = st.text_area("Your Message*", height=150, 
                                 placeholder="Type your message here...", key="message")
            
            submitted = st.form_submit_button("Send Message")
            
            if submitted:
                if name and email and message:
                    # Prepare form data
                    form_data = {
                        "access_key": WEB3FORMS_ACCESS_KEY,
                        "name": name,
                        "email": email,
                        "subject": subject,
                        "message": message,
                        "botcheck": "",
                        "redirect": "false"
                    }
                    
                    # Send to Web3Forms with proper response handling
                    with st.spinner("Sending your message..."):
                        try:
                            response = requests.post(
                                WEB3FORMS_ENDPOINT,
                                data=form_data,
                                timeout=10
                            )
                            
                            # Check for HTML success response
                            if response.status_code == 200 and "<title>Success!" in response.text:
                                st.success("Thank you for your message! We'll get back to you soon.")
                            else:
                                # Try to parse as JSON if not HTML
                                try:
                                    result = response.json()
                                    if result.get("success"):
                                        st.success("Thank you for your message! We'll get back to you soon.")
                                    else:
                                        error_msg = result.get("message", "Unknown error occurred")
                                        st.error(f"Failed to send message: {error_msg}")
                                except ValueError:
                                    st.error(f"Unexpected response. Please try again. (Status: {response.status_code})")
                                    st.text(f"Response content: {response.text[:200]}...")
                            
                        except requests.exceptions.RequestException as e:
                            st.error(f"Network error occurred: {str(e)}")
                        except Exception as e:
                            st.error(f"An unexpected error occurred: {str(e)}")
                else:
                    st.error("Please fill in all required fields (marked with *)")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Contact information
        st.markdown("""
        ## Alternative Contact Methods
        """)
        
        with st.container():
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="contact-info">', unsafe_allow_html=True)
                st.subheader("Email Us")
                st.write("📧 support@interviewsimulator.com")
                st.write("📧 feedback@interviewsimulator.com")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="contact-info">', unsafe_allow_html=True)
                st.subheader("Social Media")
                st.write("🐦 Twitter: [@InterviewSimApp](https://twitter.com/InterviewSimApp)")
                st.write("💼 LinkedIn: [Interview Simulator](https://linkedin.com/company/interview-simulator)")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # FAQ section
        st.subheader("Frequently Asked Questions")
        
        faq_items = {
            "How does the contact form work?": """
            Our contact form uses Web3Forms to securely deliver your messages to our team.
            Your information is protected and will only be used to respond to your inquiry.
            """,
            "When can I expect a response?": """
            We typically respond to inquiries within 1-2 business days. For urgent matters,
            please contact us directly via email.
            """,
            "Is my information secure?": """
            Yes! We use Web3Forms which encrypts your data in transit and doesn't store
            your information on their servers after delivering it to us.
            """
        }
        
        for question, answer in faq_items.items():
            with st.expander(question):
                st.write(answer)

def show_interview_page():
    """Display the main interview page"""
    # Create two columns
    col1, col2 = st.columns([1, 2])
    
    # Left column: Resume upload and interview settings
    with col1:
        st.header("Resume Upload")
        uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")
        
        # Interview settings section - always show regardless of interview state
        st.subheader("Interview Settings")
        
        # Number of questions slider - always show
        num_questions = st.slider(
            "Number of questions", 
            min_value=1, 
            max_value=20, 
            value=st.session_state.num_questions,
            help="Select how many questions you want in the interview"
        )
        
        # Update session state with selected number
        st.session_state.num_questions = num_questions
        
        # Add "Apply Changes" button if interview is in progress
        if st.session_state.interview_started and not st.session_state.interview_completed:
            if st.button("Apply Changes"):
                update_interview_settings()
        
        if uploaded_file is not None:
            st.success("Resume uploaded successfully!")
            
            # Display PDF preview
            with st.expander("Resume Preview", expanded=True):
                resume_text = extract_text_from_pdf(uploaded_file)
                st.text_area("Extracted Text", resume_text, height=300)
                st.session_state.resume_text = resume_text
            
            # Start interview button
            if not st.session_state.interview_started:
                if st.button("Start Interview"):
                    start_interview()
            
            # Reset button
            if st.session_state.interview_completed:
                if st.button("Start New Interview"):
                    reset_interview()
    
    # Right column: Chat interface
    with col2:
        st.header("Technical Interview")
        
        # Fixed-height chat container
        chat_container = st.container(height=550)
        
        with chat_container:
            display_messages()
        
        # Input for user's answer - This is the fixed version
        if st.session_state.interview_started and not st.session_state.interview_completed:
            with st.form(key=f"answer_form_{st.session_state.current_question}"):
                user_answer = st.text_area(
                    "Your answer",
                    key=f"user_input_{st.session_state.current_question}",
                    height=100
                )
                
                submitted = st.form_submit_button("Submit Answer")
                
                if submitted and user_answer:
                    # Process the answer
                    st.session_state.messages.append({"role": "user", "content": user_answer})
                    
                    current_q_index = st.session_state.current_question - 1
                    question_text = st.session_state.interview_questions[current_q_index]
                    
                    evaluation = evaluate_answer(
                        question_text, 
                        user_answer, 
                        st.session_state.resume_text
                    )
                    
                    st.session_state.evaluations.append(evaluation)
                    st.session_state.total_score += int(evaluation.get("score", 5))
                    
                    eval_text = f"Question {current_q_index + 1}/{st.session_state.num_questions} - Score: {evaluation.get('score')}/10"
                    st.session_state.messages.append({"role": "evaluation", "content": eval_text})
                    
                    next_question()
                    st.rerun()  # This will clear the form
    
    # Display performance metrics and detailed review outside chat window (below chat)
    if st.session_state.interview_completed and hasattr(st.session_state, 'summary_data'):
        st.markdown("---")
        st.header("Interview Performance Results")
        
        # Performance metrics section
        metrics_cols = st.columns(3)
        with metrics_cols[0]:
            st.metric(
                label="Total Score", 
                value=f"{st.session_state.summary_data['total_score']}/{st.session_state.summary_data['max_score']}"
            )
        with metrics_cols[1]:
            st.metric(
                label="Average Score", 
                value=f"{st.session_state.summary_data['average_score']:.1f}/10"
            )
        with metrics_cols[2]:
            performance_level = "Excellent" if st.session_state.summary_data['average_score'] >= 8 else \
                               "Good" if st.session_state.summary_data['average_score'] >= 6 else \
                               "Average" if st.session_state.summary_data['average_score'] >= 4 else "Needs Improvement"
            st.metric(label="Performance Level", value=performance_level)
        
        # Add visualizations for performance
        st.subheader("Performance Visualization")
        
        # Create DataFrame for bar chart of scores
        scores_data = pd.DataFrame({
            'Question': [f"Q{r['question_number']}" for r in st.session_state.summary_data["question_reviews"]],
            'Score': [r['score'] for r in st.session_state.summary_data["question_reviews"]]
        })
        
        # Create bar chart for scores
        score_chart = alt.Chart(scores_data).mark_bar().encode(
            x=alt.X('Question', sort=None),
            y=alt.Y('Score', scale=alt.Scale(domain=[0, 10])),
            color=alt.Color('Score:Q', scale=alt.Scale(
                domain=[0, 4, 7, 10],
                range=['red', 'orange', 'green', 'green']
            )),
            tooltip=['Question', 'Score']
        ).properties(
            title='Scores by Question',
            width=600
        )
        
        st.altair_chart(score_chart, use_container_width=True)
        
        # Add performance over time line chart
        st.subheader("Performance Progression")
        cumulative_scores = []
        running_total = 0
        
        for i, review in enumerate(st.session_state.summary_data["question_reviews"]):
            running_total += review['score']
            cumulative_scores.append({
                'Question Number': i + 1,
                'Average Score': running_total / (i + 1)
            })
        
        progress_data = pd.DataFrame(cumulative_scores)
        
        progress_chart = alt.Chart(progress_data).mark_line(point=True).encode(
            x='Question Number',
            y=alt.Y('Average Score', scale=alt.Scale(domain=[0, 10])),
            tooltip=['Question Number', 'Average Score']
        ).properties(
            title='Running Average Score',
            width=600
        )
        
        st.altair_chart(progress_chart, use_container_width=True)
        
        # Add pie chart for score distribution
        st.subheader("Score Distribution")
        pie_chart = create_score_distribution_pie_chart(st.session_state.summary_data)
        st.plotly_chart(pie_chart, use_container_width=True)
        
        # Question-by-question review in expander
        with st.expander("Question-by-Question Review", expanded=True):
            for review in st.session_state.summary_data["question_reviews"]:
                st.subheader(f"Question {review['question_number']}")
                st.write(f"**Question:** {review['question_text']}")
                st.write(f"**Score:** {review['score']}/10")
                
                # Use columns for strengths and improvements
                cols = st.columns(2)
                with cols[0]:
                    st.success(f"**Strengths:**\n{review['strengths']}")
                with cols[1]:
                    st.warning(f"**Areas to Improve:**\n{review['improvements']}")
                
                st.divider()
        
        # Next steps and recommendations
        with st.expander("Next Steps & Recommendations", expanded=True):
            st.write("""
            ## Next Steps
            Based on your performance, focus on the improvement areas mentioned above. 
            Consider reviewing relevant documentation and practice more coding problems to strengthen your skills.
            
            ## Learning Resources
            - Technical documentation for areas you need to improve
            - Practice coding exercises on platforms like LeetCode, HackerRank
            - Join developer communities related to your field
            - Consider online courses to fill knowledge gaps
            """)
def show_coding_results():
    coding_state = st.session_state.app_state['coding']
    st.header("Coding Interview Results")
    
    if not coding_state['evaluations']:
        st.warning("No evaluation data available")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Score", f"{coding_state['total_score']}/{len(coding_state['evaluations'])*10}")
    with col2:
        avg_score = coding_state['total_score']/len(coding_state['evaluations'])
        st.metric("Average Score", f"{avg_score:.1f}/10")
    with col3:
        st.metric("Problems Solved", len(coding_state['evaluations']))
    
    st.subheader("Code Quality Analysis")
    metrics = pd.DataFrame([{
        'Correctness': e.get('correctness', False),
        'Efficiency': e.get('efficiency', 'N/A'),
        'Best Practices': ', '.join(e.get('best_practices', [])) or 'N/A'
    } for e in coding_state['evaluations']])
    st.dataframe(metrics)
    
    st.subheader("Detailed Feedback")
    for idx, eval in enumerate(coding_state['evaluations']):
        with st.expander(f"Problem {idx+1} Feedback", expanded=False):
            st.markdown(f"**Score**: {eval.get('score', 'N/A')}/10")
            st.markdown(f"**Correctness**: {'Yes' if eval.get('correctness') else 'No'}")
            st.markdown(f"**Efficiency**: {eval.get('efficiency', 'N/A')}")
            st.markdown(f"**Best Practices**: {', '.join(eval.get('best_practices', [])) or 'N/A'}")
            st.markdown(f"**Errors**: {', '.join(eval.get('errors', [])) or 'None'}")
            st.markdown(f"**Feedback**: {eval.get('feedback', 'N/A')}")


def coding_interview_page():
    """Display the coding interview page"""
    st.header("Coding Interview Preparation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Coding Settings")
        st.session_state.coding_language = st.selectbox(
            "Programming Language",
            ["Python", "Java", "JavaScript","C++"],
            key="coding_lang"
        )
        st.session_state.num_coding_questions = st.slider(
            "Number of Questions", 
            min_value=1, 
            max_value=10, 
            value=5,
            key="coding_num_q"
        )
        
        if st.button("👨💻 Start Coding Interview"):
            st.session_state.coding_started = True
            st.session_state.current_coding_question = 0
            st.session_state.code_evaluations = []
            st.session_state.coding_score = 0
            st.session_state.coding_messages = []
            st.rerun()
    
    with col2:
        if st.session_state.get('coding_started', False):
            if not st.session_state.get('coding_completed', False):
                current_question = st.session_state.current_coding_question
                total_questions = st.session_state.num_coding_questions
                
                if current_question < total_questions:
                    st.markdown(f"### Question {current_question + 1}/{total_questions}")
                    question = CODING_QUESTIONS[st.session_state.coding_language][current_question]
                    st.markdown(f"<div class='code-container'>{question}</div>", unsafe_allow_html=True)
                    
                    with st.form(key=f"coding_form_{current_question}"):
                        code = st.text_area("Write Your Code", height=300)
                        if st.form_submit_button("Submit Code"):
                            evaluation = evaluate_code_question(question, code)
                            st.session_state.code_evaluations.append(evaluation)
                            st.session_state.coding_score += evaluation.get('score', 0)
                            st.session_state.current_coding_question += 1
                            
                            if st.session_state.current_coding_question >= st.session_state.num_coding_questions:
                                st.session_state.coding_completed = True
                            st.rerun()
                else:
                    st.session_state.coding_completed = True
                    st.rerun()
            else:
                show_coding_results()
        else:
            st.info("Please select programming language and number of questions, then click 'Start Coding Interview'")

def evaluate_code_question(question, code):
    """Evaluate coding question using Gemini API"""
    try:
        prompt = f"""
        Evaluate this coding solution:
        Problem: {question}
        Code: {code}
        
        Provide JSON response with:
        - score (1-10)
        - feedback
        - correctness (boolean)
        - efficiency
        - best_practices
        - errors
        """
        
        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        
        response = requests.post(GEMINI_API_URL, headers=headers, json=data)
        
        if response.status_code == 200:
            response_data = response.json()
            text_response = response_data['candidates'][0]['content']['parts'][0]['text']
            
            if '```json' in text_response:
                json_str = text_response.split('```json')[1].split('```')[0].strip()
            else:
                json_str = text_response
            
            return json.loads(json_str)
        return default_code_evaluation()
    except Exception as e:
        return default_code_evaluation()

def show_coding_results():
    """Display coding interview results"""
    st.header("Coding Interview Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Score", f"{st.session_state.coding_score}/{st.session_state.num_coding_questions * 10}")
    with col2:
        avg_score = st.session_state.coding_score / st.session_state.num_coding_questions
        st.metric("Average Score", f"{avg_score:.1f}/10")
    with col3:
        st.metric("Completed Questions", st.session_state.num_coding_questions)
    
    st.subheader("Performance Analysis")
    
    # Create score distribution chart
    scores = [e.get('score', 0) for e in st.session_state.code_evaluations]
    chart_data = pd.DataFrame({
        "Question": [f"Q{i+1}" for i in range(len(scores))],
        "Score": scores
    })
    
    bar_chart = alt.Chart(chart_data).mark_bar().encode(
    x='Question',
    y='Score',
    color=alt.Color('Score', scale=alt.Scale(domain=[0, 10], range=['lightblue', 'darkblue']))
)
    
    st.altair_chart(bar_chart, use_container_width=True)
    
    st.subheader("Detailed Feedback")
    for i, eval in enumerate(st.session_state.code_evaluations):
        with st.expander(f"Question {i+1} Feedback", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Correctness**")
                st.write("✅ Correct" if eval.get('correctness') else "❌ Incorrect")
                st.markdown("**Efficiency**")
                st.write(eval.get('efficiency', 'N/A'))
            with col2:
                st.markdown("**Best Practices**")
                st.write(eval.get('best_practices', 'N/A'))
                st.markdown("**Errors**")
                st.write(eval.get('errors', 'None'))
            
            st.markdown("**Score:** " + str(eval.get('score', 'N/A')) + "/10")
            st.markdown("**Feedback:** " + eval.get('feedback', 'No feedback available'))

def default_code_evaluation():
    return {
        "score": 5,
        "feedback": "Evaluation failed",
        "correctness": False,
        "efficiency": "N/A",
        "best_practices": [],
        "errors": ["Evaluation error"]
    }
def main():
    """Main application function"""
    # Initialize current_page if not exists
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "interview"
    
    # Navigation sidebar
    with st.sidebar:
        st.title("Navigation")
        if st.button("Interview Simulator"):
            st.session_state.current_page = "interview"
        if st.button("Coding Interview"):
            st.session_state.current_page = "coding_interview"
        if st.button("Contact Us"):
            st.session_state.current_page = "contact"
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        Resume Interview Simulator helps you prepare for technical interviews by:
        - Analyzing your resume
        - Generating personalized questions
        - Evaluating your responses
        - Providing detailed feedback
        """)
    
    # Display the appropriate page based on navigation
    if st.session_state.current_page == "interview":
        show_interview_page()
    elif st.session_state.current_page == "coding_interview":
        coding_interview_page()
    elif st.session_state.current_page == "contact":
        show_contact_page()

if __name__ == "__main__":
    main()