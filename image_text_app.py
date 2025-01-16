from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import streamlit as st

# Load Image-to-Text model
def load_image_to_text_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model

# Load Text-to-Image model
def load_text_to_image_model():
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", torch_dtype=torch.float16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

# Load models
image_to_text_processor, image_to_text_model = load_image_to_text_model()
text_to_image_pipe = load_text_to_image_model()

st.title("Image & Text Generator")

# User Input: Option Selection
option = st.selectbox("Choose Input Type:", ["Image-to-Text", "Text-to-Image"])

if option == "Image-to-Text":
    st.header("Upload an Image to Generate Text")
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_image:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Generate Text from Image
        inputs = image_to_text_processor(images=image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        outputs = image_to_text_model.generate(**inputs)
        caption = image_to_text_processor.decode(outputs[0], skip_special_tokens=True)

        # Display the result
        st.subheader("Generated Text:")
        st.write(caption)

        # Download button for the generated text
        st.download_button("Download Text", caption, file_name="image_caption.txt")

elif option == "Text-to-Image":
    st.header("Enter Text to Generate an Image")
    text_prompt = st.text_input("Enter your prompt:")

    if text_prompt:
        # Generate Image from Text
        generated_image = text_to_image_pipe(text_prompt).images[0]

        # Display the result
        st.subheader("Generated Image:")
        st.image(generated_image, caption="Generated from Prompt", use_column_width=True)

        # Download button for the generated image
        generated_image.save("output.png")
        with open("output.png", "rb") as file:
            st.download_button("Download Image", file, file_name="generated_image.png")
