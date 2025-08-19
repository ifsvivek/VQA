# Visual Question Answering (VQA) System

A web application that enables users to ask and receive answers to questions about images. This system takes an image and a text-based question as inputs and generates a natural language answer as output, leveraging the powerful Gemma-3n local model.

## About Visual Question Answering (VQA)

Visual Question Answering is an AI task that combines computer vision and natural language processing. A VQA system interprets visual content (images) and natural language questions to generate relevant textual answers. The system must:

1. Understand the content of the image (objects, relationships, activities, etc.)
2. Comprehend the natural language question
3. Reason about the relationship between the question and the image
4. Generate an appropriate answer in natural language

This implementation uses the state-of-the-art Gemma-3n multimodal model running locally to seamlessly connect visual understanding with language comprehension and generation.

## Features

-   Upload custom images for analysis
-   Natural language interface for image-based questions
-   Contextual conversation with follow-up questions
-   Multimodal understanding (combining vision and language)
-   Pre-loaded example images for demonstration
-   Interactive chat interface with message history
-   One-click conversation reset
-   **Local model execution** - no internet connection required for inference
-   **Privacy-focused** - your images and conversations stay on your device

## Technical Overview

This VQA system leverages:

-   **Gemma-3n-e4b-it**: Google's powerful multimodal model capable of understanding both images and text
-   **Transformers**: For loading and running the model locally
-   **PyTorch**: For efficient model inference
-   **LangChain**: For maintaining conversation context and history
-   **Gradio**: For building the interactive web interface
-   **PIL (Python Imaging Library)**: For image processing

## Requirements

-   Python 3.8+
-   GPU with at least 8GB VRAM (recommended for optimal performance)
-   Sufficient disk space for model download (~6GB)
-   CUDA-compatible GPU (optional but recommended)

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/ifsvivek/VQA
    cd VQA
    ```

2. Install required packages:

    ```bash
    pip install -r requirements.txt
    ```

    **Note**: The first time you run the application, the Gemma model will be downloaded automatically. This may take some time and requires an internet connection.

3. *(Optional)* Create a `.env` file if you have any environment-specific configurations (though none are required for the local model).

## Usage

1. Launch the application:

    ```bash
    python app.py
    ```

    **First Run**: The model will be downloaded and loaded. This may take several minutes and require ~6GB of disk space.

2. Open the provided URL in your web browser (typically http://127.0.0.1:7860)

3. Use the application in one of two ways:

    - Upload your own image using the upload button
    - Select one of the example images

4. Type your question in the text box and press Enter or click "Send"

5. Review the AI's response and continue the conversation with follow-up questions

### Example Questions

Here are some examples of questions you can ask about an uploaded image:

-   "What objects can you see in this image?"
-   "What time of day does this appear to be?"
-   "Is there any text visible in this image? If so, what does it say?"
-   "What colors are most prominent in this scene?"
-   "How many people are in this image and what are they doing?"
-   "Describe the architectural style of the building in this image."

## How It Works

1. **Model Loading**: The Gemma-3n model is loaded locally using the Transformers library with automatic device mapping for optimal performance.

2. **Image Processing**: The uploaded image is temporarily stored and passed directly to the model without needing API encoding.

3. **Question Analysis**: The system processes the user's natural language question.

4. **Context Management**: Previous conversation turns are tracked to maintain context for follow-up questions.

5. **Local Inference**: The image and question (along with any relevant conversation history) are processed by the local Gemma model.

6. **Multimodal Processing**: The model analyzes both the visual content of the image and the linguistic content of the question locally on your device.

7. **Response Generation**: The model generates a natural language response that answers the question based on the image content.

8. **User Interface**: The response is displayed to the user in an intuitive chat interface.

## Limitations

-   Requires significant computational resources (GPU recommended).
-   Initial model download requires internet connection and disk space (~6GB).
-   Inference speed depends on your hardware capabilities.
-   Very complex visual scenes may receive simplified interpretations.
-   The quality of answers depends on the clarity and resolution of the uploaded images.

## Advantages of Local Model

-   **Privacy**: Your images and conversations never leave your device
-   **No API costs**: Run unlimited queries without usage fees
-   **Offline capability**: Works without internet after initial setup
-   **Customization**: Full control over the model and its parameters
-   **No rate limits**: Process as many images as your hardware allows

## Credits

This application uses:

-   [Gemma-3n](https://huggingface.co/google/gemma-3n-e4b-it) model from Google
-   [Transformers](https://huggingface.co/transformers/) for model loading and inference
-   [PyTorch](https://pytorch.org/) for deep learning operations
-   [Gradio](https://gradio.app/) for the web interface
-   [LangChain](https://www.langchain.com/) for conversation management
