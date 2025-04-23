# Resume Analysis & Technical Interview Simulator

This application is a Streamlit-based technical interview simulator that analyzes your resume and conducts a personalized technical interview, providing feedback and scoring your performance.

## Features

- Upload and analyze your resume (PDF format)
- Interactive chat interface for technical interview simulation
- Question generation based on your resume content
- Real-time evaluation of your answers
- Performance assessment after 10 questions
- Skill development recommendations based on your performance

## Prerequisites

- Python 3.8+
- Gemini API key (for LLM functionality)

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd resume-interview-simulator
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   - Create a `.env` file in the project root (or rename `.env.example` to `.env`)
   - Add your Gemini API credentials:
     ```
     GEMINI_API_URL="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
     GEMINI_API_KEY="your_api_key_here"
     ```

## Usage

1. Start the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Access the application in your web browser (typically at http://localhost:8501)

3. Upload your resume (PDF format) using the upload area on the left side of the screen

4. Once your resume is uploaded and processed, the interview will begin in the chat interface on the right side

5. Answer the technical questions to the best of your ability

6. After 10 questions, the system will provide:
   - Your overall score
   - Areas of strength
   - Areas for improvement
   - Technical skills that need development

## Docker Setup

### Running with Docker

1. Build the Docker image:
   ```
   docker build -t interview-bot .
   ```

2. Run the container with your API key:
   ```
   docker run -p 8501:8501 -e GEMINI_API_KEY="your_api_key_here" interview-bot
   ```

The GEMINI_API_URL is hardcoded in the Docker image, so you only need to provide your API key.

### Environment Configuration

You can provide the API key in two ways:

1. **Setting the environment variable directly:**
   ```
   docker run -p 8501:8501 -e GEMINI_API_KEY="your_api_key_here" interview-bot
   ```

2. **Using an .env file:**
   - Create a `.env` file with your API key:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```
   - Pass it to Docker with the `--env-file` flag:
     ```
     docker run -p 8501:8501 --env-file .env interview-bot
     ```

### Accessing the Application

Once running in Docker, access the application at:
- http://localhost:8501

If you're running Docker on a remote machine, replace `localhost` with the appropriate IP address or hostname.

### Docker Troubleshooting

- **Port Conflicts:** If port 8501 is already in use, you can map to a different port:
  ```
  docker run -p 8000:8501 --env-file .env interview-bot
  ```
  Then access the application at http://localhost:8000

- **Environment Variables:** If the application can't access the Gemini API, verify your environment variables are correctly passed to the container

- **Container Crashes:** Check the container logs for error messages:
  ```
  docker logs <container_id>
  ```

## Application Structure

- `app.py`: Main Streamlit application entry point
- `utils.py`: Helper functions for PDF processing, Gemini API integration, and evaluation logic
- `requirements.txt`: List of required Python packages
- `.env`: Environment variables configuration file

## Troubleshooting

- **PDF Upload Issues**: Ensure your PDF is not password-protected and is properly formatted
- **API Connection Errors**: Verify your Gemini API key is correctly set in the `.env` file
- **Dependency Issues**: Make sure all packages in `requirements.txt` are properly installed

## License

[Add your license information here]

## Contributors

[Add contributor information here]

# neurohire-interview-simulator
