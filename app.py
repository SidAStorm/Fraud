import os
import numpy as np
import streamlit as st
from PIL import Image
from datetime import datetime
import dashboard
import classifyPage

# Set Streamlit page configuration
st.set_page_config(
    page_title="SAFIRE",
    page_icon="🤖",
    layout="wide"
)

# Define the pages and icons
PAGES = {
    "Dashboard": {"page": dashboard, "icon": "📊"},
    "Classify Image": {"page": classifyPage, "icon": "🖼️"}
}

# Sidebar styling
# Add logo to the sidebar
st.sidebar.image('./LOGO.png', use_column_width=True)
st.sidebar.title("SAFIRE")
st.sidebar.markdown("SAFIRE is a tool that utilizes the power of Deep Learning to distinguish Real images from the Fake ones.")
st.sidebar.subheader('Navigation')
# Radio buttons for page selection with icons
selection = st.sidebar.radio("", list(PAGES.keys()), format_func=lambda page_name: f"{PAGES[page_name]['icon']} {page_name}")

# Get the selected page and its corresponding icon
selected_page = PAGES[selection]['page']


# Display the selected page
selected_page.app()
