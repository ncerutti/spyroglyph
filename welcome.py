import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from PIL import Image
import tempfile
import os

# Will add streamlit-cropper later


def main():
    example_image_path = "./images/example_spiral.png"
    st.set_page_config(
        page_title="Image effects",
        page_icon="🎨",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    st.title("Image effects")
    function_choices = ["Spiral", "Double Spiral", "Grid"]
    selected_function = st.sidebar.selectbox("Choose a function", function_choices)
    if selected_function == "Spiral":
        example_image_path = "./images/example_spiral.png"
    elif selected_function == "Double Spiral":
        example_image_path = "./images/example_double_spiral.png"
    elif selected_function == "Grid":
        example_image_path = "./images/example_grid.png"
    example_image = Image.open(example_image_path)
    st.image(example_image, caption=f"Example {selected_function}", width=400)
    if st.button("Go!"):
        # go to the corresponding page
        if selected_function == "Spiral":
            switch_page("spiral")
        elif selected_function == "Double Spiral":
            switch_page("double_spiral")
        elif selected_function == "Grid":
            gridoglyph_page()


if __name__ == "__main__":
    main()
