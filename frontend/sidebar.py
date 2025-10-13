import streamlit as st
from api_utils import APIClient
from datetime import datetime
import requests

def render_sidebar(api_client: APIClient):
    with st.sidebar:
        st.title("📚 Knowledge Base")
        
        if st.session_state.get("authenticated", False):
            user = st.session_state.get("user", {})
            
            # User info section
            with st.container():
                st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                        <h4 style="margin: 0;">👤 {user.get('username', 'User')}</h4>
                        <p style="margin: 5px 0 0 0; font-size: 14px; color: #666;">
                            ✉️ {user.get('email', '')}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # Stats
            try:
                stats = api_client.get_user_stats(st.session_state["user_id"])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📄 Documents", stats.get("total_documents", 0))
                with col2:
                    st.metric("🔍 Searches", stats.get("total_searches", 0))
                
                storage_mb = stats.get('storage_used_mb', 0)
                storage_color = "🟢" if storage_mb < 50 else "🟡" if storage_mb < 100 else "🔴"
                st.metric(f"{storage_color} Storage Used", f"{storage_mb:.2f} MB")
                
            except Exception as e:
                st.warning("⚠️ Could not load statistics")
            
            st.divider()
            
            # Upload section
            st.subheader("📤 Upload Documents")
            
            with st.expander("ℹ️ Supported Formats", expanded=False):
                st.markdown("""
                - 📝 **Text**: .txt, .md
                - 📄 **PDF**: .pdf
                - 📊 **Data**: .csv, .json
                - 📏 **Size Limit**: 100 MB per file
                """)
            
            uploaded_files = st.file_uploader(
                "Choose files",
                type=["txt", "md", "pdf", "csv", "json"],
                accept_multiple_files=True,
                key="file_uploader",
                help="Upload documents to add them to your knowledge base"
            )
            
            if uploaded_files:
                st.info(f"📋 {len(uploaded_files)} file(s) selected")
                
                # Show file details
                with st.expander("📄 File Details", expanded=True):
                    total_size = 0
                    for file in uploaded_files:
                        size_kb = len(file.getvalue()) / 1024
                        total_size += size_kb
                        st.text(f"• {file.name} ({size_kb:.1f} KB)")
                    st.markdown(f"**Total Size:** {total_size:.1f} KB")
                
                # Upload button
                if st.button("⬆️ Upload Files", use_container_width=True, type="primary"):
                    with st.spinner("Uploading and processing files..."):
                        try:
                            result = api_client.upload_files(st.session_state["user_id"], uploaded_files)
                            
                            if result.get("success"):
                                successful = result.get('successful_uploads', 0)
                                failed = len(result.get('failed_uploads', []))
                                
                                if failed == 0:
                                    st.success(f"✅ All {successful} files uploaded successfully!")
                                    st.balloons()
                                else:
                                    st.warning(f"⚠️ {successful} succeeded, {failed} failed")
                                    
                                    # Show failed files
                                    for fail in result.get('failed_uploads', []):
                                        st.error(f"❌ {fail['filename']}: {fail['error']}")
                            else:
                                st.error("❌ Upload failed!")
                                
                            # Refresh the page
                            st.rerun()
                            
                        except requests.exceptions.HTTPError as e:
                            st.error(f"❌ Upload failed: {e.response.status_code}")
                        except Exception as e:
                            st.error(f"❌ Upload error: {str(e)}")
            
            st.divider()
            
            # Documents list
            st.subheader("📁 My Documents")
            
            try:
                docs = api_client.get_documents(st.session_state["user_id"], limit=20)
                
                if docs:
                    # Search/filter documents
                    search_doc = st.text_input("🔍 Filter documents", placeholder="Type to filter...")
                    
                    filtered_docs = docs
                    if search_doc:
                        filtered_docs = [d for d in docs if search_doc.lower() in d['filename'].lower()]
                    
                    st.caption(f"Showing {len(filtered_docs)} of {len(docs)} documents")
                    
                    for doc in filtered_docs:
                        with st.expander(f"📄 {doc['filename']}", expanded=False):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.write(f"**Size:** {doc['filesize'] / 1024:.2f} KB")
                                st.write(f"**Type:** {doc['filetype']}")
                            
                            with col2:
                                upload_date = datetime.fromisoformat(doc['uploaddate'])
                                st.write(f"**Uploaded:**")
                                st.caption(upload_date.strftime('%Y-%m-%d'))
                                st.caption(upload_date.strftime('%H:%M'))
                            
                            if st.button(f"🗑️ Delete", key=f"del_{doc['id']}", use_container_width=True, type="secondary"):
                                if st.session_state.get(f"confirm_del_{doc['id']}", False):
                                    try:
                                        api_client.delete_document(doc['id'], st.session_state["user_id"])
                                        st.success("✅ Document deleted!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"❌ Delete failed: {str(e)}")
                                else:
                                    st.session_state[f"confirm_del_{doc['id']}"] = True
                                    st.warning("⚠️ Click again to confirm deletion")
                else:
                    st.info("📭 No documents uploaded yet. Upload your first document to get started!")
                    
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ Failed to load documents: {e.response.status_code}")
            except Exception as e:
                st.error(f"❌ Error loading documents: {str(e)}")
            
            st.divider()
            
            # Logout button
            if st.button("🚪 Logout", use_container_width=True, type="primary"):
                st.session_state.clear()
                st.success("👋 Logged out successfully!")
                st.rerun()
        else:
            st.info("Please login to continue")