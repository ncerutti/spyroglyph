import streamlit as st
import io
from PIL import Image
import tempfile
from effects import spiral_function


def main():
    st.title("Spiral")
    uploaded_file = st.file_uploader("Choose a png image file", type=["png", "jpg"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Uploaded Image", use_column_width=True)
        size = st.sidebar.slider("Size", 100, 500, 300)
        shades = st.sidebar.slider("Shades", 1, 64, 16)
        spiral_points = st.sidebar.slider("Spiral Points", 1000, 10000, 5000)
        spiral_turns = st.sidebar.slider("Spiral Turns", 10, 100, 50)
        spiral_r0 = st.sidebar.slider("Spiral r0", 0, 100, 0)
        spiral_r1_f = st.sidebar.slider("Spiral r1 factor", 0.0, 1.0, 0.5)
        thin = st.sidebar.slider("Thin", 0.0001, 0.0010, 0.00025)
        thick_f = st.sidebar.slider("Thick factor", 0.0, 1.0, 0.85)
        spiral_offset_angle = st.sidebar.slider("Spiral Offset Angle", 0, 360, 0)
        crop = st.sidebar.checkbox("Crop Image")
        colormap = st.sidebar.selectbox("Colormap", ["none", "viridis", "plasma"])
        rescaler_factor = st.sidebar.slider("Rescaler Factor", 0.0, 2.0, 1.0)
        color = st.sidebar.color_picker("Color", "#7731A4")
        alpha = st.sidebar.slider("Alpha", 0.0, 1.0, 0.75)
        # Create a temporary file for the input image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            input_image.save(temp_file.name)
            temp_file_path = temp_file.name

        if st.button("Generate Spiral"):
            output_buffer = io.BytesIO()
            spiral_function(
                input_image=temp_file_path,
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
                color=color,
                output_image=output_buffer,
                rescaler_factor=rescaler_factor,
                alpha=alpha,
            )
            output_image = Image.open(output_buffer)
            st.image(output_image, caption="Generated Spiral", use_column_width=True)
            output_buffer.seek(0)
            st.download_button("Download Spiral", output_buffer, file_name="spiral.png")


if __name__ == "__main__":
    main()
