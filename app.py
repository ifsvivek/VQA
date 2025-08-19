import gradio as gr
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
import torch
from PIL import Image


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


def generate_response_with_gemma(image_pil, text_prompt):
    """Generate response using local Gemma model"""
    
    # Create messages for the model
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant that can analyze images and answer questions about them."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_pil},
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
        
        # Generate response
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]
        
        # Decode the response
        decoded = processor.decode(generation, skip_special_tokens=True)
        return decoded.strip()
        
    except Exception as e:
        return f"Error generating response: {str(e)}"


def answer_question(image, question):
    """Main function to answer questions about images"""
    if image is None:
        return "Please upload an image first."
    
    if not question.strip():
        return "Please ask a question about the image."
    
    try:
        # Generate response using the model
        response = generate_response_with_gemma(image, question)
        return response
    except Exception as e:
        return f"Error: {str(e)}"


# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="Visual Question Answering with Gemma") as demo:
        gr.Markdown("# Visual Question Answering (VQA) with Local Gemma Model")
        gr.Markdown("Upload an image and ask questions about it!")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")
                question_input = gr.Textbox(
                    label="Ask a question about the image",
                    placeholder="What do you see in this image?",
                    lines=2
                )
                submit_btn = gr.Button("Submit", variant="primary")
                
            with gr.Column():
                output = gr.Textbox(
                    label="Answer",
                    lines=5,
                    placeholder="Upload an image and ask a question to get started..."
                )
        
        # Note: Upload your own images to test the model
        
        # Event handlers
        submit_btn.click(
            answer_question,
            inputs=[image_input, question_input],
            outputs=output
        )
        
        question_input.submit(
            answer_question,
            inputs=[image_input, question_input],
            outputs=output
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)
