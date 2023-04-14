import streamlit as st
import io
from PIL import Image
import tempfile
from effects import spiral_function, double_spiral_function, grid_function
import os


def main():
    st.title("Double Spiral")
    uploaded_file_1 = st.file_uploader(
        "Choose a png image file", type=["png", "jpg"], key="uploader_one"
    )
    uploaded_file_2 = st.file_uploader(
        "Choose a png image file", type=["png", "jpg"], key="uploader_two"
    )
    if uploaded_file_1 is not None and uploaded_file_2 is not None:
        input_image_1 = Image.open(uploaded_file_1)
        input_image_2 = Image.open(uploaded_file_2)
        st.image(input_image_1, caption="First Image", use_column_width=True)
        st.image(input_image_2, caption="Second Image", use_column_width=True)
        size = st.sidebar.slider("Size", 100, 500, 300)
        shades = st.sidebar.slider("Shades", 1, 64, 16)
        spiral_points = st.sidebar.slider("Spiral Points", 1000, 10000, 5000)
        spiral_turns = st.sidebar.slider("Spiral Turns", 10, 100, 50)
        spiral_r0 = st.sidebar.slider("Spiral r0", 0, 100, 0)
        spiral_r1_f = st.sidebar.slider("Spiral r1 factor", 0.0, 1.0, 0.5)
        thin = st.sidebar.slider("Thin", 0.0001, 0.0010, 0.00025)
        thick_f = st.sidebar.slider("Thick factor", 0.0, 1.0, 0.95)
        spiral_offset_angle = st.sidebar.slider("Spiral Offset Angle", 0, 360, 0)
        crop = st.sidebar.checkbox("Crop Image")
        colormap = st.sidebar.selectbox(
            "Colormap", ["gray", "viridis", "plasma", "none"]
        )
        rescaler_factor = st.sidebar.slider("Rescaler Factor", 0.0, 2.0, 1.0)
        # Create a temporary file for the input image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file_1:
            input_image_1.save(temp_file_1.name)
            temp_file_path_1 = temp_file_1.name
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file_2:
            input_image_2.save(temp_file_2.name)
            temp_file_path_2 = temp_file_2.name

        if st.button("Generate Double Spiral"):
            output_buffer = io.BytesIO()
            double_spiral_function(
                input_image_1=temp_file_path_1,
                input_image_2=temp_file_path_2,
                size=size,
                n_shades=shades,
                spiral_points=spiral_points,
                spiral_turns=spiral_turns,
                spiral_r0=spiral_r0,
                spiral_r1_f=spiral_r1_f,
                thin=thin,
                thick_f=thick_f,
                spiral_offset_angle=spiral_offset_angle,
                crop=crop,
                colormap=colormap,
                output_image=output_buffer,
                rescaler_factor=rescaler_factor,
            )
            output_image = Image.open(output_buffer)
            st.image(
                output_image, caption="Generated Double Spiral", use_column_width=True
            )
            output_buffer.seek(0)
            st.download_button(
                "Download Double Spiral", output_buffer, file_name="double_spiral.png"
            )


if __name__ == "__main__":
    main()
