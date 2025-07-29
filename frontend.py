import streamlit as st
from backend import (
    extract_text_from_files,
    chunk_text,
    ask_question_over_chunks,
    ask_general_question
)
import tempfile
import os

def main():
    # Custom styled title
    st.markdown("""
        <h1 style='text-align: center; color: #0d6efd;'>⚡ GridAI</h1>
        <p style='text-align: center; font-size: 18px; color: #666;'>Your smart assistant for electrical grid planning</p>
        <hr style='margin-top: 0;'>
    """, unsafe_allow_html=True)

    # Upload multiple file types
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF, DOCX, XLSX, CSV, TXT, JSON)",
        type=["pdf", "docx", "xlsx", "xls", "csv", "txt", "json"],
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Processing documents..."):
            temp_dir = tempfile.mkdtemp()
            file_paths = []
            for file in uploaded_files:
                temp_path = os.path.join(temp_dir, file.name)
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                file_paths.append(temp_path)
            full_text = extract_text_from_files(file_paths)
            st.session_state["document_text_chunks"] = chunk_text(full_text)
            st.success(f"Loaded {len(uploaded_files)} documents.")

    # Initialize session state
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Define what happens on submit
    def submit_question():
        user_question = st.session_state.user_input.strip()
        if not user_question:
            return  # ignore empty input

        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # Clear input box
        st.session_state.user_input = ""

        # Set last_question to trigger answer fetching
        st.session_state.last_question = user_question

    # Text input with Enter to submit
    st.text_input(
        "Enter your question:",
        key="user_input",
        on_change=submit_question
    )

    # When question is submitted, get answer
    if st.session_state.last_question:
        with st.spinner("Getting answer..."):
            try:
                if "document_text_chunks" in st.session_state:
                    # Pass the whole chat history and document chunks
                    answer = ask_question_over_chunks(
                        st.session_state["document_text_chunks"],
                        st.session_state.chat_history
                    )
                else:
                    answer = ask_general_question(st.session_state.chat_history)

                # Add assistant's reply to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

                st.markdown("**Answer:**")
                st.write(answer)

                # Reset last_question to avoid repeated calls
                st.session_state.last_question = ""

            except Exception as e:
                st.error(f"Failed to get answer: {e}")

def display_chat_history():
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
                <div style="
                    background-color: #daf1ff;
                    padding: 12px;
                    border-radius: 12px;
                    margin-bottom: 8px;
                    max-width: 70%;
                    align-self: flex-end;
                    ">
                    <b>You:</b><br>{msg['content']}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="
                    background-color: #e6e6e6;
                    padding: 12px;
                    border-radius: 12px;
                    margin-bottom: 8px;
                    max-width: 70%;
                    ">
                    <b>Eric:</b><br>{msg['content']}
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
