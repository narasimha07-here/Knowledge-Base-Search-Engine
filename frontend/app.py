import streamlit as st
from api_utils import APIClient,APIError
from sidebar import render_sidebar
from chat_interface import render_search_interface, render_search_history

st.set_page_config(
    page_title="Knowledge Base Search",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {
        max-width: 100%;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

api_client = APIClient()

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "search_results" not in st.session_state:
    st.session_state["search_results"] = None

if "show_history" not in st.session_state:
    st.session_state["show_history"] = False

def render_login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("🔐 Knowledge Base Login")
        st.divider()
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.subheader("Welcome Back!")
            with st.form("login_form"):
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("🔓 Login", use_container_width=True, type="primary")

                if submitted:
                    if not username or not password:
                        st.error("Please enter both username and password.")
                    else:
                        try:
                            # --- ERROR HANDLING FOR LOGIN ---
                            response = api_client.login_user(username, password)
                            st.session_state["authenticated"] = True
                            st.session_state["user_id"] = response["id"]
                            st.session_state["user"] = response
                            st.success("Login successful!")
                            st.rerun()
                        except APIError as e:
                            # Display the specific error from the backend
                            st.error(f"Login Failed: {e.message}")
        
        with tab2:
            st.subheader("Create New Account")
            with st.form("register_form"):
                new_username = st.text_input("Username", key="reg_username")
                new_email = st.text_input("Email", key="reg_email")
                new_password = st.text_input("Password", type="password", key="reg_password")
                confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
                submitted = st.form_submit_button("📝 Register", use_container_width=True, type="primary")

                if submitted:
                    if not all([new_username, new_email, new_password]):
                        st.error("Please fill in all fields.")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match.")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters long.")
                    else:
                        try:
                            # --- ERROR HANDLING FOR REGISTRATION ---
                            api_client.register_user(new_username, new_email, new_password)
                            st.success("Registration successful! Please switch to the Login tab.")
                            st.balloons()
                        except APIError as e:
                            # Display specific errors like "Username already exists"
                            st.error(f"Registration Failed: {e.message}")

def render_main_app():
    render_sidebar(api_client)
    if st.session_state.get("show_history", False):
        render_search_history(api_client)
    else:
        render_search_interface(api_client)

if not st.session_state["authenticated"]:
    render_login_page()
else:
    render_main_app()