import gradio as gr
from utils import yahoo_bg_generator, SAM
from PIL import Image
import numpy as np
import os

coords = None

# Capture the coordinates when the user clicks on the image
def get_coordinates(evt: gr.SelectData):
    global coords
    coords = evt.index
    print(f"Coordinates you click: {coords}")
    return str(coords)

# update the displayed image after the user uploads an image
def update_display_image(image):
    # Save the uploaded image to a path
    image_path = "uploaded_image.png"
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image.save(image_path)
    return image

# process the image using the saved image path and clicked coordinates
def process_with_image_path(coords):
    image_path = "uploaded_image.png"
    # Check if the image file exists
    if os.path.exists(image_path):
        # Apply SAM to segment the image based on the coordinates
        result = SAM(image_path, coords)
        return result
    else:
        return "Image file not found."

if __name__=="__main__":

    # Create a Gradio Blocks interface
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                # Upload image input field
                input_image = gr.Image(sources="upload", label="Upload your image", interactive=True)
                output_coord = gr.Textbox(label="Click Coordinates", interactive=False)
            with gr.Column(scale=1):
                # Display the image where the user can click
                display_image = gr.Image(label="Click on the image", interactive=False)
            with gr.Column(scale=1):
                # Display the segmented object
                seg_object_image = gr.Image(label="Segmented Object", interactive=False)
            with gr.Column(scale=1):
                # Textbox for the user to enter the prompt for background generation
                prompt_textbox = gr.Textbox(label="Enter Prompt:", interactive=True)
                generate_bg_button = gr.Button("Go!")
                # Display the final image with the new background
                changed_bg_image = gr.Image(label="Image with new background", interactive=False)
        
        # Update the display image when a new image is uploaded
        input_image.change(update_display_image, inputs=input_image, outputs=display_image)
        
        # Capture the coordinates when the user clicks on the image
        display_image.select(get_coordinates, inputs=None, outputs=output_coord)
        
        # Process the image using the coordinates and display the segmented object
        output_coord.change(process_with_image_path, inputs=output_coord, outputs=seg_object_image)
        
        # Function to generate a new image with a background based on a prompt
        def generate_image(seg_image, prompt):
            # Check if the image file exists before generating the new background
            image_path = "uploaded_image.png"
            if os.path.exists(image_path):
                result = yahoo_bg_generator(seg_image, prompt)
                return result
            else:
                return "Image file not found."
        # Generate the new image with background when the button is clicked
        generate_bg_button.click(generate_image, inputs=[seg_object_image, prompt_textbox], outputs=changed_bg_image)
    
    demo.launch(share=True)