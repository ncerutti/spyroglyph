"""Welcome page for the webapp.
Allows the user to choose which function to use and provides an example image.
The selection can be done from the menu on the left as well."""
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from PIL import Image

# Want to add streamlit-cropper later


def main():
    """Provides sample images for the effects."""
    # Page configuration
    example_image_path = "./images/example_spiral.png"
    st.set_page_config(
        page_title="Cool image effects",
        page_icon="🎨",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    st.title("Image effects")

    # Function selection
    function_choices = ["Spiral", "Double Spiral", "Flow"]
    selected_function = st.sidebar.selectbox("Choose a function", function_choices)

    # Display example image
    if selected_function == "Spiral":
        example_image_path = "./images/example_spiral.png"
    elif selected_function == "Double Spiral":
        example_image_path = "./images/example_double_spiral.png"
    elif selected_function == "Flow":
        example_image_path = "./images/example_flow.png"
    example_image = Image.open(example_image_path)
    st.image(example_image, caption=f"Example {selected_function}", width=400)

    # Button to go to the corresponding page
    if st.button("Go!"):
        if selected_function == "Spiral":
            switch_page("spiral")
        elif selected_function == "Double Spiral":
            switch_page("double_spiral")
        elif selected_function == "Flow":
            switch_page("flow")


if __name__ == "__main__":
    main()
