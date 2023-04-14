import streamlit as st
import io
from PIL import Image
import tempfile
from effects import flow_function


def main():
    st.title("Spiral")
    uploaded_file = st.file_uploader("Choose a png image file", type=["png", "jpg"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Uploaded Image", use_column_width=True)
        size = st.sidebar.slider("Size", 100, 500, 300)
        n_points = st.sidebar.slider("Number of points", 100, 500, 800)
        step_length = st.sidebar.slider("Step length", 1, 10, 1)
        spiral_r0 = st.sidebar.slider("Spiral r0", 0, 100, 0)
        spiral_r1_f = st.sidebar.slider("Spiral r1 factor", 0.0, 1.0, 0.5)
        thin = st.sidebar.slider("Thin", 0.0001, 0.0010, 0.00025)
        thick_f = st.sidebar.slider("Thick factor", 0.0, 1.0, 0.95)
        n_steps = st.sidebar.slider("Number of steps", 0, 1000, 400)
        crop = st.sidebar.checkbox("Crop Image")
        colormap = st.sidebar.selectbox(
            "Colormap", ["gray", "viridis", "plasma", "none"]
        )
        rescaler_factor = st.sidebar.slider("Rescaler Factor", 0.0, 2.0, 1.0)
        # Create a temporary file for the input image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            input_image.save(temp_file.name)
            temp_file_path = temp_file.name

        if st.button("Generate Flow"):
            output_buffer = io.BytesIO()
            flow_function(
                input_image=temp_file_path,
                x_side=size,
                y_side=size,
                n_points=n_points,
                step_length=step_length,
                n_steps=n_steps,
                thin=thin,
                thick=thick_f,
            )
            output_image = Image.open(output_buffer)
            st.image(output_image, caption="Generated Flow", use_column_width=True)
            output_buffer.seek(0)
            st.download_button("Download Flow", output_buffer, file_name="flow.png")


if __name__ == "__main__":
    main()
