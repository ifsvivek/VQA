# Visual Question Answering (VQA) System

A web application that enables users to ask and receive answers to questions about images. This system takes an image and a text-based question as inputs and generates a natural language answer as output, leveraging the powerful Llama-3.2-11b-vision model via the Groq API.

## About Visual Question Answering (VQA)

Visual Question Answering is an AI task that combines computer vision and natural language processing. A VQA system interprets visual content (images) and natural language questions to generate relevant textual answers. The system must:

1. Understand the content of the image (objects, relationships, activities, etc.)
2. Comprehend the natural language question
3. Reason about the relationship between the question and the image
4. Generate an appropriate answer in natural language

This implementation uses state-of-the-art multimodal AI to seamlessly connect visual understanding with language comprehension and generation.

## Features

- Upload custom images for analysis
- Natural language interface for image-based questions
- Contextual conversation with follow-up questions
- Multimodal understanding (combining vision and language)
- Pre-loaded example images for demonstration
- Interactive chat interface with message history
- One-click conversation reset

## Technical Overview

This VQA system leverages:

- **Llama-3.2-11b-vision**: A large multimodal model capable of understanding both images and text
- **Groq API**: For fast and efficient model inference
- **LangChain**: For maintaining conversation context and history
- **Gradio**: For building the interactive web interface
- **PIL (Python Imaging Library)**: For image processing

## Requirements

- Python 3.8+
- Groq API key (Get one at https://console.groq.com/)
- Internet connection for API access

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd VQA
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage

1. Launch the application:
   ```bash
   python app.py
   ```

2. Open the provided URL in your web browser (typically http://127.0.0.1:7860)

3. Use the application in one of two ways:
   - Upload your own image using the upload button
   - Select one of the example images

4. Type your question in the text box and press Enter or click "Send"

5. Review the AI's response and continue the conversation with follow-up questions

### Example Questions

Here are some examples of questions you can ask about an uploaded image:

- "What objects can you see in this image?"
- "What time of day does this appear to be?"
- "Is there any text visible in this image? If so, what does it say?"
- "What colors are most prominent in this scene?"
- "How many people are in this image and what are they doing?"
- "Describe the architectural style of the building in this image."

## How It Works

1. **Image Processing**: The uploaded image is temporarily stored and converted to a base64 encoding for API submission.

2. **Question Analysis**: The system processes the user's natural language question.

3. **Context Management**: Previous conversation turns are tracked to maintain context for follow-up questions.

4. **API Interaction**: The image and question (along with any relevant conversation history) are sent to the Groq API utilizing the Llama-3.2-11b-vision model.

5. **Multimodal Processing**: The model analyzes both the visual content of the image and the linguistic content of the question.

6. **Response Generation**: The model generates a natural language response that answers the question based on the image content.

7. **User Interface**: The response is displayed to the user in an intuitive chat interface.

## Limitations

- The system requires internet access to communicate with the Groq API.
- Very complex visual scenes may receive simplified interpretations.
- The quality of answers depends on the clarity and resolution of the uploaded images.
- There may be usage limits based on the Groq API's rate limiting policies.

## Credits

This application uses:
- [Groq API](https://groq.com/) for LLM inference
- [Gradio](https://gradio.app/) for the web interface
- [LangChain](https://www.langchain.com/) for conversation management
- [Llama-3.2-11b-vision](https://www.meta.ai/llama/) model from Meta AI
