import gradio as gr
import numpy as np
import cv2
import os
from utils import quantize_image, update_color, generate_color_variants
from PIL import Image, ImageColor
import re



def process_image(image, n_colors):
    """Process the input image and extract colors"""
    # Convert from BGR (Gradio default) to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Save temporary file for processing
    temp_path = "temp_input_image.jpg"
    cv2.imwrite(temp_path, image)
    
    # Get quantized image and color data
    quantized_image, color_data = quantize_image(temp_path, n_colors)
    
    if quantized_image is None or color_data is None:
        return None, [], [], [], [], [], None, None, None, None
    
    # Extract labels and centers
    labels, centers = color_data
    
    # Generate all image variants
    color_variants = generate_color_variants(quantized_image.shape, labels, centers)
    
    # Convert quantized image from RGB to BGR for display
    quantized_image_bgr = cv2.cvtColor(quantized_image, cv2.COLOR_RGB2BGR)
    
    # Format colors for the color picker group - convert to hex format
    color_list = []
    for color in centers:
        hex_color = f"#{int(color[2]):02x}{int(color[1]):02x}{int(color[0]):02x}"
        color_list.append(hex_color)
    
    return (
        quantized_image_bgr, 
        color_variants["color_images"],
        color_variants["transparent_images"],
        color_variants["mask_images"],
        color_list,
        color_list,
        quantized_image,
        labels,
        centers,
        color_variants,
    )

def update_colors(prev_colors, new_colors, quantized_image, labels, centers, color_variants):
    # inputs=[n_colors_state, updated_colors_state, quantized_image_state, labels_state, centers_state, color_variants_state],
    """Process color changes and update all outputs"""
    if prev_colors == new_colors:
        print("No color changes detected")
        return (
            cv2.cvtColor(quantized_image, cv2.COLOR_RGB2BGR),
            color_variants["color_images"],
            color_variants["transparent_images"],
            color_variants["mask_images"],
            prev_colors,
            prev_colors,
            quantized_image,
            labels,
            centers,
            color_variants,
        )
    
    print("Updating colors:", prev_colors, new_colors)
    
    # Convert hex colors to RGB numpy arrays
    new_centers = []
    for hex_color in new_colors:
        # Remove '#' and convert to RGB
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        new_centers.append(np.array([r, g, b], dtype=np.uint8))
    
    new_centers = np.array(new_centers)
    
    # Update colors in the quantized image
    updated_quantized_image, updated_centers = update_color(quantized_image, labels, centers, new_centers)
    
    # Generate all image variants with the new colors
    updated_color_variants = generate_color_variants(updated_quantized_image.shape, labels, updated_centers)
    
    # outputs=[output_image, color_gallery, transparent_gallery, mask_gallery, colors, n_colors_state, quantized_image_state, labels_state, centers_state, color_variants_state]
    return (
        updated_quantized_image, #output_image
        updated_color_variants["color_images"], # color_gallery
        updated_color_variants["transparent_images"], # transparent_gallery
        updated_color_variants["mask_images"], # mask_gallery
        new_colors.copy(), # colors
        new_colors.copy(), # n_colors_state
        updated_quantized_image, # quantized_image_state
        labels, # labels_state
        updated_centers, # centers_state
        updated_color_variants, # color_variants_state
    )

def create_interface():
    with gr.Blocks(title="Color Separator",
                   css="""
.color-picker {
    z-index: 1000 !important;
}"""
                   ) as demo:
        gr.Markdown("# Image Color Separator")
        gr.Markdown("Upload an image to separate it into distinct color regions.")
        n_colors_state = gr.State([])
        quantized_image_state = gr.State()
        labels_state = gr.State()
        centers_state = gr.State()
        color_variants_state = gr.State()
        updated_colors_state = gr.State()

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Input Image", type="numpy")
                n_colors = gr.Slider(minimum=2, maximum=16, step=1, value=8, label="Number of Colors")
                process_btn = gr.Button("Process Image")

                @gr.render([n_colors_state])
                def color_pickers_rendering(colors):
                    color_pickers = []
                    for i, color in enumerate(colors):
                        color_picker = gr.ColorPicker(
                            label=f"Color {i+1}", 
                            interactive=True,
                            show_label=True,
                            container=True,
                            elem_id=f"color-picker-{i}",
                            value=color,
                            elem_classes=["color-picker-container"],
                            info="Edit colors by clicking on them",
                        )
                        color_pickers.append(color_picker)

                        def update_color_fn(color, n_colors_state, updated_colors_state, i=i):
                            print(updated_colors_state)
                            if updated_colors_state is None:
                                updated_colors_state = n_colors_state.copy()
                            m = re.match(r"rgba\(\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\)", color)
                            if m:
                                updated_color = [int(float(m.group(1))), int(float(m.group(2))), int(float(m.group(3)))]
                                updated_color_hex = f"#{updated_color[0]:02x}{updated_color[1]:02x}{updated_color[2]:02x}"
                                color = updated_color_hex
                                updated_colors_state[i] = updated_color_hex
                            return updated_colors_state
                            
                        
                        color_picker.change(
                            fn=update_color_fn,
                            inputs=[color_picker, n_colors_state, updated_colors_state],
                            outputs=[updated_colors_state],
                        )
                update_colors_btn = gr.Button("Update Colors")
                
                
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Quantized Image"):
                        output_image = gr.Image(label="Quantized Image", type="numpy")
                    
                    with gr.TabItem("Color Images"):
                        color_gallery = gr.Gallery(
                            label="Colors with White Background", 
                            show_label=True,
                            elem_id="color-gallery",
                            columns=[4], 
                            rows=[1],
                            object_fit="contain",
                            height="auto",
                        )
                    
                    with gr.TabItem("Transparent Images"):
                        transparent_gallery = gr.Gallery(
                            label="Colors with Transparent Background", 
                            show_label=True,
                            elem_id="transparent-gallery",
                            columns=[4], 
                            rows=[1],
                            object_fit="contain",
                            height="auto",
                        )
                    
                    with gr.TabItem("Mask Images"):
                        mask_gallery = gr.Gallery(
                            label="Color Masks", 
                            show_label=True,
                            elem_id="mask-gallery",
                            columns=[4], 
                            rows=[1],
                            object_fit="contain",
                            height="auto",
                        )
        
        colors = gr.Textbox(label="Colors")
        
        # Process button trigger
        process_btn.click(
            fn=process_image,
            inputs=[input_image, n_colors],
            outputs=[output_image, color_gallery, transparent_gallery, mask_gallery, colors, n_colors_state, quantized_image_state, labels_state, centers_state, color_variants_state]
        )
        update_colors_btn.click(
            fn=update_colors,
            inputs=[n_colors_state, updated_colors_state, quantized_image_state, labels_state, centers_state, color_variants_state],
            outputs=[output_image, color_gallery, transparent_gallery, mask_gallery, colors, n_colors_state, quantized_image_state, labels_state, centers_state, color_variants_state]
        )
        
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False)
