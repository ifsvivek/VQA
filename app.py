import gradio as gr
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
import torch
from PIL import Image
import numpy as np
import os

# Enable TF32 for better performance on your RTX A5000 GPU
torch.set_float32_matmul_precision('high')

# Initialize the local Gemma model with 4-bit quantization for speed
model_path = "./gemma"  # Local model directory
print("Loading Gemma model from local directory... This may take a while.")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"  # Normalized float4 for better accuracy
)

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_path, 
    device_map="auto", 
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config
).eval()

processor = AutoProcessor.from_pretrained(model_path)
print("Model loaded successfully! (With 4-bit quantization for faster inference)")

# Global variables for chat state
current_image = None  # Store resized PIL Image
chat_history = []

def prepare_image(image_pil):
    """Resize, convert to RGB, ensure uint8 dtype, and return PIL Image"""
    if image_pil is None:
        return None
    # Resize to model's expected resolution (896x896 as per model config)
    image_pil = image_pil.resize((896, 896), Image.Resampling.LANCZOS)
    # Convert to RGB and ensure uint8
    image_pil = image_pil.convert("RGB")
    # Explicitly convert to uint8 if needed
    img_array = np.array(image_pil)
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
        image_pil = Image.fromarray(img_array)
    return image_pil

def generate_response_with_gemma(image_pil, text_prompt, previous_context=""):
    """Generate response using local Gemma model"""
    
    # Create the system prompt with context included
    system_text = "You are a helpful assistant that can analyze images and answer questions about them. Keep your responses concise and informative."
    if previous_context:
        system_text += f" Previous conversation context: {previous_context}"
    
    # Create messages for the model with proper alternating structure
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_text}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_pil},  # Pass PIL Image directly
                {"type": "text", "text": text_prompt}
            ]
        }
    ]
    
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
        
        # Generate response with optimizations: lower max tokens, greedy decoding
        with torch.inference_mode():
            generation = model.generate(
                **inputs, 
                max_new_tokens=150,  # Reduced for speed; increase if needed
                do_sample=False,
            )
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
        current_image = prepare_image(image)  # Prepare and store PIL Image
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
    
    # Build context from previous messages (limit to last 2 exchanges for speed)
    context_messages = []
    for i in range(max(0, len(history) - 3), len(history) - 1):
        if history[i][0] and history[i][1]:
            # Clean up the user message for context
            user_text = history[i][0]
            if user_text.startswith("[Image uploaded] "):
                user_text = user_text[17:]
            context_messages.append(f"Q: {user_text} A: {history[i][1]}")
    
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
        current_image = prepare_image(image)
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
            for prompt in example_prompts[:3]:
                gr.Button(prompt, size="sm").click(
                    lambda x=prompt: x, outputs=msg
                )
        
        with gr.Row():
            for prompt in example_prompts[3:]:
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
