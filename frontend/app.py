import streamlit as st
import requests
st.title("PDF Question Answering System")

question = st.text_input("Ask a question")

# Create a button labeled "Submit"
# The code inside this block runs only when the button is clicked
if st.button("Submit"):
    # Send a POST request to the backend API
    # The question entered by the user is sent as JSON data
    response = requests.post(
        "http://localhost:8000/ask",
        json={"question": question}  # JSON payload containing the user question
    )
    # Check if the API returned a successful response (HTTP status code 200)
    if response.status_code == 200:
        # Extract the answer from the JSON response and display it on the Streamlit app
        st.write(response.json()["answer"])