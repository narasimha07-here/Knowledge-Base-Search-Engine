import streamlit as st
from api_utils import APIClient
from datetime import datetime
import requests

def render_search_interface(api_client: APIClient):
    st.title("🔍 Knowledge Base Search")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_mode = st.radio(
            "Search Mode",
            options=["Semantic Search", "Hybrid Search"],  # Fixed: removed "Deep Search"
            horizontal=True,
            label_visibility="collapsed"
        )
    
    with col2:
        if st.button("📜 Search History", use_container_width=True):
            st.session_state["show_history"] = True
            st.rerun()
    
    st.divider()
    
    # Show helpful tips
    with st.expander("💡 Search Tips", expanded=False):
        st.markdown("""
        - **Semantic Search**: Best for conceptual questions and finding similar meanings
        - **Hybrid Search**: Combines keyword matching with semantic understanding for better accuracy
        - Be specific in your questions for more accurate results
        - Ask one question at a time for clearer answers
        """)
    
    query = st.text_input(
        "Ask me anything about your documents...",
        placeholder="Type your question here...",
        key="search_query",
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        search_button = st.button("🔍 Search", use_container_width=True, type="primary")
    
    with col2:
        if st.session_state.get("search_results"):
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state["search_results"] = None
                st.session_state["current_query"] = None
                st.rerun()

    if search_button and query:
        with st.spinner("🔄 Searching your knowledge base..."):
            try:
                if search_mode == "Semantic Search":
                    result = api_client.semantic_search(
                        query=query,
                        userid=st.session_state["user_id"],
                        topk=10
                    )
                else:  # Hybrid Search
                    result = api_client.hybrid_search(
                        query=query,
                        userid=st.session_state["user_id"],
                        topk=20,  # Get more chunks for hybrid
                        use_all_chunks=False,  # Fixed: don't use all chunks
                        keyword_weight=0.3,
                        semantic_weight=0.7
                    )
                
                st.session_state["search_results"] = result
                st.session_state["current_query"] = query
                st.success("✅ Search completed!")
                
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ Search failed: {e.response.status_code} - {e.response.text}")
                if e.response.status_code == 500:
                    st.info("💡 Tip: Make sure you have uploaded documents before searching.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to the API. Please check if the backend server is running.")
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {str(e)}")
    
    if st.session_state.get("search_results"):
        result = st.session_state["search_results"]
        
        # Display query
        st.subheader("❓ Your Question")
        st.info(st.session_state.get("current_query", ""))
        
        st.divider()
        
        # Display answer
        st.subheader("💬 Answer")
        
        response_text = result.get('response', 'No response generated')
        if response_text and response_text.strip():
            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 4px solid #4CAF50;">
                    <p style="font-size: 16px; line-height: 1.8; margin: 0; color: #333;">
                        {response_text}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("⚠️ No answer was generated. Try rephrasing your question or upload more relevant documents.")
        
        st.divider()
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Retrieved Chunks", result.get("retrieved_chunks", 0))
        with col2:
            st.metric("⏱️ Search Time", f"{result.get('search_time', 0):.2f}s")
        with col3:
            search_type = result.get("search_type", "semantic")
            st.metric("🔍 Search Type", search_type.title())

       

def render_search_history(api_client: APIClient):
    st.title("📜 Search History")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("⬅️ Back to Search", use_container_width=True):
            st.session_state["show_history"] = False
            st.rerun()
    
    st.divider()
    
    try:
        history = api_client.get_search_history(st.session_state["user_id"], limit=100)
        
        if history:
            # Summary stats
            total_time = sum(h['search_time'] for h in history)
            avg_time = total_time / len(history) if history else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Searches", len(history))
            with col2:
                st.metric("⏱️ Avg Search Time", f"{avg_time:.2f}s")
            with col3:
                st.metric("⏰ Total Time", f"{total_time:.1f}s")
            
            st.divider()
            
            # Search history list
            for idx, search in enumerate(history, 1):
                created_at = datetime.fromisoformat(search['created_at'])
                
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.write(f"**{idx}. {search['query']}**")
                        st.caption(f"🕒 {created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    with col2:
                        st.metric("Results", search['results_count'])
                        st.caption(f"⏱️ {search['search_time']:.2f}s")
                    
                    st.divider()
        else:
            st.info("📭 No search history yet. Start searching to see your history here!")
            
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Failed to load search history: {e.response.status_code}")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")