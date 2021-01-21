
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from src import Image, Model

st.set_option("deprecation.showfileUploaderEncoding", False)
st.set_page_config(page_title="Deep detect drawings")

model = Model()


# Header
st.title("Deep detect drawings")
st.write("""
Deep detect drawings is a Python app based on a CNN model made with PyTorch,
to recognize the item that you draw on the canvas. See the [notebook of the model's creation]
(https://colab.research.google.com/drive/1hh1lcDcXK3oxL2cEAPvlNXUY10bAhZp-?usp=sharing) for more details.
""")

col1, col2 = st.beta_columns([6, 4])

# Drawing area
with col1:

    st.subheader("Drawing area")
    st.markdown("Draw something cool, don't make it easy !")
    canvas_result = st_canvas(
        stroke_width=20,
        stroke_color="#fff",
        background_color="#000",
        update_streamlit=True,
        drawing_mode="freedraw",
        key="canvas",
        width=400
    )

image = Image(canvas_result.image_data)

with col2:
    st.subheader("This model can recognize...")
    st.markdown("""
    Between 8 different items:
    
     - Banana
     - Axe
     - Spider
     - Hand
     - House
     - Eyeglasses
     - Cup
     - Diamond
     
     And more soon !
     """)


# Check if the user has written something
if (image.image is not None) and (not image.is_empty()):

    # Get the predicted class
    prediction = model.predict(image.get_prediction_ready())

    col3, col4 = st.beta_columns(2)

    # Display the image predicted by the model
    with col3:

        base_url = "https://f003.backblazeb2.com/file/joffreybvn/deepdrawing/"
        images = [
            base_url + "banana.png",
            base_url + "axe.png",
            base_url + "spider.png",
            base_url + "hand.png",
            base_url + "house.png",
            base_url + "eyeglasses.png",
            base_url + "cup.png",
            base_url + "diamond.png",
        ]

        st.subheader("Recognized image")
        st.markdown("The image recognized by the model")
        st.image(images[prediction], width=250)

    # Display the pro
    with col4:
        st.subheader("Probability distribution")
        st.markdown("Was your drawing hard to recognize ?")
        st.bar_chart(pd.DataFrame(
            model.probabilities,
            columns=[f"{i}" for i in range(8)]
        ).T)


# Sidebar
st.sidebar.header("About the author")
st.sidebar.markdown("""
**Joffrey Bienvenu**

Python dev, studying Machine Learning at BeCode.org.

 - Website: [joffreybvn.be](https://joffreybvn.be/)
 - Twitter: [@joffreybvn](https://twitter.com/Joffreybvn)
 - LinkedIn: [in/joffreybvn](https://www.linkedin.com/in/joffreybvn/)
 - Github: [joffreybvn](https://github.com/joffreybvn)
""")

st.sidebar.header("See on github")
st.sidebar.markdown("""
See the code and fork this project on Github:

[Deep Detect Drawings repository](https://github.com/Joffreybvn/deep-detect-drawings)
""")
