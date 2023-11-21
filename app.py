import streamlit as st
def main():
    st.title("CSE 574D Intoduction to Machine Learning")
    st.sidebar.title("EMINST Classification")
    st.sidebar.header("Made by: Charvi")
    activities = ["About Me","Model Prediction"]
    choice = st.sidebar.selectbox("Select ",activities)
    if choice == 'About Me':
        st.header(
            "Name: CHARVI KUSUMA ")
        st.header("UBID: charviku")
        st.header("Date: 11|20|2023")
    elif choice == 'Model Prediction':
        from eminst_part4 import main as eminst_main
        eminst_main()
    else: 
        st.write("Make A selection from the dropdown")

if __name__ == '__main__':
    main()


