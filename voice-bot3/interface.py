import streamlit as st
from voicebot import send_message

st.title("Your problem is our problem!")
if st.button("text mode"):
    send_message()


st.image(
            "https://media.giphy.com/media/JR0i9DOAVEDrFYyEr9/giphy.gif", # I prefer to load the GIFs using GIPHY
            width=400 # The actual size of most gifs on GIPHY are really small, and using the column-width parameter would make it weirdly big. So I would suggest adjusting the width manually!
        )
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
