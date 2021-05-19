import streamlit as st

def main():
    uploaded_file = st.file_uploader("Upload Files", type=['png', 'jpeg'])
    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write(file_details)

if __name__ == '__main__':
    main()

