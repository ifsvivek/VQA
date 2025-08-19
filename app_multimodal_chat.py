import gradio as gr
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
import torch
from PIL import Image
import tempfile
import os


# Initialize the local Gemma model
model_path = "./gemma-3n-e4b-it"  # Local model directory
print("Loading Gemma model from local directory... This may take a while.")
model = Gemma3nForConditionalGeneration.from_pretrained(
    model_path, 
    device_map="auto", 
    torch_dtype=torch.bfloat16
).eval()
processor = AutoProcessor.from_pretrained(model_path)
print("Model loaded successfully!")

# Global variables for chat state
current_image = None
chat_history = []


def save_image_to_temp(image_pil):
    """Save PIL image to a temporary file and return the path"""
    if image_pil is None:
        return None
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image_pil.save(temp_file.name, format="JPEG")
    return temp_file.name


def generate_response_with_gemma(image_pil, text_prompt, previous_context=""):
    """Generate response using local Gemma model"""
    
    # Create messages for the model
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant that can analyze images and answer questions about them. Keep your responses concise and informative."}]
        }
    ]
    
    # Add previous context if available
    if previous_context:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": f"Previous conversation context: {previous_context}"}]
        })
    
    # Add current user message with image
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image_pil},
            {"type": "text", "text": text_prompt}
        ]
    })
    
    try:
        # Apply chat template and tokenize
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate response
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=250, do_sample=False)
            generation = generation[0][input_len:]
        
        # Decode the response
        decoded = processor.decode(generation, skip_special_tokens=True)
        return decoded.strip()
        
    except Exception as e:
        return f"Error generating response: {str(e)}"


def format_chat_history(history):
    """Format chat history for display"""
    formatted = []
    for i, (user_msg, bot_msg) in enumerate(history):
        if user_msg:
            formatted.append(f"**You:** {user_msg}")
        if bot_msg:
            formatted.append(f"**Assistant:** {bot_msg}")
    return "\n\n".join(formatted)


def add_message(image, message, history):
    """Add a new message to the chat"""
    global current_image, chat_history
    
    if image is not None:
        current_image = image
        # Save image for display
        image_path = save_image_to_temp(image)
        user_message = f"[Image uploaded] {message}" if message.strip() else "[Image uploaded] What do you see in this image?"
    else:
        if current_image is None:
            return history, "", "Please upload an image first before asking questions."
        user_message = message
    
    if not message.strip() and image is None:
        return history, "", "Please enter a message."
    
    # Add user message to history
    history = history + [[user_message, None]]
    
    return history, "", ""


def respond(history):
    """Generate bot response"""
    global current_image, chat_history
    
    if current_image is None:
        history[-1][1] = "Please upload an image first."
        return history
    
    user_message = history[-1][0]
    
    # Extract just the text part (remove image upload prefix if present)
    if user_message.startswith("[Image uploaded] "):
        text_prompt = user_message[17:]  # Remove "[Image uploaded] " prefix
    else:
        text_prompt = user_message
    
    # Build context from previous messages (last 3 exchanges to keep context manageable)
    context_messages = []
    for i in range(max(0, len(history) - 4), len(history) - 1):
        if history[i][0] and history[i][1]:
            # Clean up the user message for context
            user_text = history[i][0]
            if user_text.startswith("[Image uploaded] "):
                user_text = user_text[17:]
            context_messages.append(f"User: {user_text}")
            context_messages.append(f"Assistant: {history[i][1]}")
    
    previous_context = " | ".join(context_messages) if context_messages else ""
    
    try:
        # Generate response using the model
        response = generate_response_with_gemma(current_image, text_prompt, previous_context)
        history[-1][1] = response
    except Exception as e:
        history[-1][1] = f"Error: {str(e)}"
    
    return history


def clear_chat():
    """Clear chat history and reset image"""
    global current_image, chat_history
    current_image = None
    chat_history = []
    return [], None, "Chat cleared. Upload a new image to start a conversation."


def upload_image_only(image):
    """Handle image upload without message"""
    global current_image
    if image is not None:
        current_image = image
        return "Image uploaded successfully! You can now ask questions about it."
    return "No image uploaded."


# Create the Gradio interface
def create_multimodal_chat():
    with gr.Blocks(title="Multimodal VQA Chat with Gemma", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 Multimodal Visual Question Answering Chat")
        gr.Markdown("Upload an image and have a conversation about it with the local Gemma model!")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Image upload section
                gr.Markdown("### 📷 Upload Image")
                image_input = gr.Image(
                    type="pil", 
                    label="Upload an image",
                    height=300
                )
                
                upload_btn = gr.Button("📤 Upload Image", variant="secondary")
                
                # Status display
                status = gr.Textbox(
                    label="Status",
                    value="Upload an image to start chatting!",
                    interactive=False,
                    lines=2
                )
                
                # Clear button
                clear_btn = gr.Button("🗑️ Clear Chat", variant="stop")
                
            with gr.Column(scale=2):
                # Chat interface
                gr.Markdown("### 💬 Chat")
                chatbot = gr.Chatbot(
                    value=[],
                    height=500,
                    label="Conversation",
                    show_label=False,
                    bubble_full_width=False
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask a question about the image...",
                        label="Your message",
                        lines=2,
                        scale=4
                    )
                    
                    send_btn = gr.Button("📤 Send", variant="primary", scale=1)
        
        # Example prompts
        gr.Markdown("### 💡 Example Questions")
        example_prompts = [
            "What do you see in this image?",
            "Describe the main objects in the image.",
            "What colors are prominent in this image?",
            "What is the setting or location shown?",
            "Are there any people in the image?",
            "What's happening in this scene?"
        ]
        
        with gr.Row():
            for i, prompt in enumerate(example_prompts[:3]):
                gr.Button(prompt, size="sm").click(
                    lambda x=prompt: x, outputs=msg
                )
        
        with gr.Row():
            for i, prompt in enumerate(example_prompts[3:]):
                gr.Button(prompt, size="sm").click(
                    lambda x=prompt: x, outputs=msg
                )
        
        # Event handlers
        
        # Upload image
        upload_btn.click(
            upload_image_only,
            inputs=[image_input],
            outputs=[status]
        )
        
        # Handle message sending
        msg.submit(
            add_message,
            inputs=[image_input, msg, chatbot],
            outputs=[chatbot, msg, status]
        ).then(
            respond,
            inputs=[chatbot],
            outputs=[chatbot]
        )
        
        send_btn.click(
            add_message,
            inputs=[image_input, msg, chatbot],
            outputs=[chatbot, msg, status]
        ).then(
            respond,
            inputs=[chatbot],
            outputs=[chatbot]
        )
        
        # Clear chat
        clear_btn.click(
            clear_chat,
            outputs=[chatbot, image_input, status]
        )
        
        # Auto-upload image when changed
        image_input.change(
            upload_image_only,
            inputs=[image_input],
            outputs=[status]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_multimodal_chat()
    demo.launch(share=True)
