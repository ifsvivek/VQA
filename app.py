import os
import base64
import tempfile
from PIL import Image
import gradio as gr
from groq import Groq
from dotenv import load_dotenv
from gradio_multimodalchatbot import MultimodalChatbot
from gradio.data_classes import FileData
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ChatMessageHistory

# Load environment variables including GROQ_API_KEY
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Global variable to store the current image
current_image = None
current_image_path = None


def save_image_to_temp(image_pil):
    """Save PIL image to a temporary file and return the path"""
    if image_pil is None:
        return None

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image_pil.save(temp_file.name, format="JPEG")
    return temp_file.name


def encode_image(image_pil):
    """Convert PIL image to base64 string"""
    import io

    buffered = io.BytesIO()
    image_pil.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def process_chat(chatbot, image, message, chat_history):
    """Process chat with history and context"""
    global current_image, current_image_path

    # Update current image if a new one is provided
    if image is not None:
        current_image = image
        current_image_path = save_image_to_temp(image)

    # If no image is available, request one
    if current_image is None:
        user_msg = {"text": message, "files": []}
        bot_msg = {
            "text": "Please upload an image first before asking questions.",
            "files": [],
        }
        chatbot.append([user_msg, bot_msg])
        return chatbot, chat_history

    # Check if chat_history is a dictionary instead of ChatMessageHistory
    if isinstance(chat_history, dict):
        # Initialize correct chat history object
        chat_history = ChatMessageHistory()

    # Add the new user message to chat history
    chat_history.add_user_message(message)

    # Create user message with image (for display)
    user_msg = {
        "text": message,
        "files": (
            [{"file": FileData(path=current_image_path)}] if current_image_path else []
        ),
    }

    # Format all previous messages for context for the API
    messages = []

    # Process all messages for the API call
    if len(chat_history.messages) >= 2:  # At least one exchange
        # Convert the image for the first message
        base64_image = encode_image(current_image)
        image_content = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        }

        # Add all messages for context
        for idx, msg in enumerate(chat_history.messages):
            if isinstance(msg, HumanMessage):
                if idx > 0:  # Not the first message
                    messages.append({"role": "user", "content": msg.content})
                else:  # First message includes image
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": msg.content},
                                image_content,
                            ],
                        }
                    )
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
    else:
        # For the first exchange, include the image
        base64_image = encode_image(current_image)
        image_content = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        }

        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": message}, image_content],
            }
        )

    try:
        # Call the API with conversation history
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=messages,
            temperature=0.7,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )

        # Extract answer from the response
        answer = completion.choices[0].message.content

        # Add AI response to chat history
        chat_history.add_ai_message(answer)

        # Create bot message (text only for now)
        bot_msg = {"text": answer, "files": []}

        # Update the visual history
        chatbot.append([user_msg, bot_msg])

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        bot_msg = {"text": error_msg, "files": []}
        chatbot.append([user_msg, bot_msg])

    return chatbot, chat_history


def upload_image(image, chatbot, chat_history):
    """Handle image upload and show a system message"""
    global current_image, current_image_path

    if image is None:
        return chatbot, chat_history

    current_image = image
    current_image_path = save_image_to_temp(image)

    # Add a system message showing the uploaded image
    system_msg = {
        "text": "Image uploaded successfully. You can now ask questions about it.",
        "files": [],
    }
    user_msg = {"text": "", "files": [{"file": FileData(path=current_image_path)}]}

    chatbot.append([user_msg, system_msg])
    return chatbot, chat_history


# Create the Gradio chat interface
with gr.Blocks(title="Visual Question Answering with MultimodalChatbot") as demo:
    gr.Markdown("# Visual Question Answering (VQA) Chat System")
    gr.Markdown(
        "Upload an image and have a conversation about it. You can ask follow-up questions!"
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image")

            # Example images
            gr.Examples(
                examples=[
                    [
                        "https://upload.wikimedia.org/wikipedia/commons/f/f2/LPU-v1-die.jpg"
                    ],
                    [
                        "https://upload.wikimedia.org/wikipedia/commons/d/da/SF_From_Marin_Highlands3.jpg"
                    ],
                ],
                inputs=image_input,
            )

            upload_button = gr.Button("Upload & Start Conversation")

        with gr.Column(scale=2):
            # Hidden state for chat history - ensure it's initialized as ChatMessageHistory
            chat_history_state = gr.State(ChatMessageHistory())

            # Multimodal Chatbot interface
            chatbot = MultimodalChatbot(
                value=[],
                height=500,
                bubble_full_width=False,
                show_copy_button=True,
                show_label=False,
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask a question about the image...",
                    label="Your Question",
                    lines=2,
                    scale=8,
                )
                
                with gr.Column(scale=1):
                    # Add buttons in a column for better alignment
                    send_button = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear Conversation")

    # Event handlers
    upload_button.click(
        upload_image,
        [image_input, chatbot, chat_history_state],
        [chatbot, chat_history_state],
    )

    # Connect both the submit action and the send button to the same function
    msg.submit(
        process_chat,
        [chatbot, image_input, msg, chat_history_state],
        [chatbot, chat_history_state],
    )
    
    send_button.click(
        process_chat,
        [chatbot, image_input, msg, chat_history_state],
        [chatbot, chat_history_state],
    )

    def clear_chat():
        global current_image, current_image_path
        current_image = None
        current_image_path = None
        return [], ChatMessageHistory()

    clear.click(clear_chat, None, [chatbot, chat_history_state], queue=False)
    
    # Clear the message box after sending
    msg.submit(lambda: "", None, [msg])
    send_button.click(lambda: "", None, [msg])

if __name__ == "__main__":
    demo.launch(share=True)
