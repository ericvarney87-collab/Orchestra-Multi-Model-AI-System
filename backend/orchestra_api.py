#!/usr/bin/env python3
"""
Orchestra API Backend - Complete Integration
Connects all features: Chat, RAG, Sessions, Hardware, Web Search, Image Gen
"""
import sys
import ollama
import time
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import threading
import os
import json
import asyncio
import base64
import auth
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import Orchestra
sys.path.insert(0, os.path.dirname(__file__))
import orchestra_v2_9_improved as orchestra
from code_executor import CodeExecutor  # SECURITY: Import secure code executor

active_engine = None # Add this line
pending_browser_context = None  # Store browser context when no engine exists yet

# Initialize code executor (SECURITY FIX)
code_executor = CodeExecutor()

# Initialize auth
user_manager = auth.UserManager()
current_user = None  # Will store logged-in username

app = Flask(__name__)
CORS(app)

# Initialize Orchestra components
config = orchestra.OrchestraConfig()
hw_monitor = orchestra.HardwareMonitor()
program_launcher = orchestra.ProgramLauncher()
# Note: librarian and engine will be created per-user in chat endpoint

# Session storage
SESSIONS_DIR = Path.home() / ".orchestra_sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

# Upload folder
UPLOAD_FOLDER = Path.home() / ".orchestra_uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

# ============================================
# HEALTH & INFO
# ============================================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        "status": "ok",
        "version": "2.9",
        "models_available": len(config.available_models),
        "rag_enabled": config.rag_enabled
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models and experts"""
    return jsonify({
        'available_models': config.available_models,
        'experts': config.experts,
        'conductor': config.conductor,
        'auto_assignments': config.auto_assignments,
        'domain_definitions': config.domain_definitions
    })

# ============================================
# AUTHENTICATION
# ============================================
@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register new user"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    profile_data = data.get('profile', {})
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    success, message = user_manager.create_user(username, password, profile_data)
    
    if success:
        return jsonify({'success': True, 'message': message})
    else:
        return jsonify({'error': message}), 400

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user"""
    global current_user
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    success, message = user_manager.authenticate(username, password)
    
    if success:
        current_user = username
        profile = user_manager.get_profile(username)
        return jsonify({
            'success': True,
            'username': username,
            'profile': profile
        })
    else:
        return jsonify({'error': message}), 401

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """Logout user"""
    global current_user
    current_user = None
    return jsonify({'success': True})

@app.route('/api/auth/current', methods=['GET'])
def get_current_user():
    """Get current logged-in user"""
    if current_user:
        profile = user_manager.get_profile(current_user)
        return jsonify({
            'logged_in': True,
            'username': current_user,
            'profile': profile
        })
    return jsonify({'logged_in': False})

@app.route('/api/auth/status', methods=['GET'])
def auth_status():
    """Alias for /api/auth/current - check login status"""
    return get_current_user()

# ============================================
# MEMORY MANAGEMENT
# ============================================
@app.route('/api/memory/save', methods=['POST'])
def save_memory():
    """Save a response to user's long-term memory with pre-computed embedding"""
    data = request.json
    username = data.get('username')
    
    if not username:
        return jsonify({'error': 'Username required'}), 401
    
    query = data.get('query', '')
    response = data.get('response', '')
    experts = data.get('experts', [])
    image = data.get('image')
    
    # --- OPTIMIZATION: Generate embedding ONCE at save time ---
    import ollama
    combined_text = f"Q: {query}\nA: {response}"
    try:
        emb_res = ollama.embeddings(model=config.embed_model, prompt=combined_text)
        embedding = emb_res['embedding']
    except Exception as e:
        print(f"WARNING: Could not generate embedding for save: {e}")
        embedding = None

    # Create memory entry including the vector
    memory_entry = {
        'query': query,
        'response': response,
        'experts': experts,
        'image': image,
        'embedding': embedding,  # Store the vector in the JSON
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to user's memory folder
    memory_dir = Path.home() / ".orchestra" / "users" / username / "memory"
    memory_file = memory_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    memory_file.write_text(json.dumps(memory_entry, indent=2))
    
    return jsonify({'success': True, 'saved': str(memory_file)})

def get_user_memories(query, user, max_results=3):
    # 1. LENGTH & CONTENT GUARD
    # If user is empty, too long, or contains newlines, it's a corrupted/prompt-injected string.
    if not user or len(user) > 100 or "\n" in user:
        # Log it so you know the frontend sent a bad 'user' value
        print(f"⚠️ Warning: Invalid user string detected ({len(user) if user else 0} chars). Skipping memory.")
        return []

    # 2. PATH CONSTRUCTION
    # Using .joinpath() is slightly safer for cross-platform
    try:
        memory_dir = Path.home() / ".orchestra" / "users" / user / "memory"
        
        # This is where the crash happened. Wrap in try/except for absolute safety.
        if not memory_dir.exists():
            return []
    except OSError:
        print(f"❌ OSError: Path too long for user '{user[:20]}...'")
        return []

    memories = []

    # 3. EXISTING LOGIC (GLOB & LOAD)
    for memory_file in memory_dir.glob("*.json"):
        try:
            memory = json.loads(memory_file.read_text())

            if 'embedding' not in memory:
                combined_text = f"Q: {memory['query']}\nA: {memory['response']}"
                m_res = ollama.embeddings(model=config.embed_model, prompt=combined_text)
                memory['embedding'] = m_res['embedding']
                memory_file.write_text(json.dumps(memory, indent=2))

            memories.append(memory)
        except:
            continue

    if not memories:
        return []

    # 4. VECTOR SEARCH
    try:
        import numpy as np
        q_res = ollama.embeddings(model=config.embed_model, prompt=query)
        q_vec = np.array(q_res['embedding'])

        scores = []
        for memory in memories:
            m_vec = np.array(memory['embedding'])
            # Safeguard against zero-length vectors
            norm = np.linalg.norm(q_vec) * np.linalg.norm(m_vec)
            if norm == 0:
                continue
            similarity = np.dot(q_vec, m_vec) / norm
            scores.append((similarity, memory))

        scores.sort(key=lambda x: x[0], reverse=True)
        # Keeping your 0.4 threshold for high recall
        relevant = [mem for score, mem in scores if score > 0.8][:max_results]

        print(f"🧠 Memory search ({user}): {len(relevant)} hits")
        return relevant

    except Exception as e:
        print(f"ERROR: Memory search failed: {e}")
        return []

@app.route('/api/memory/load', methods=['GET'])
def load_memory_detail():
    """Reads a specific JSON memory file and returns it to the UI."""
    file_path = request.args.get('path')
    
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'Memory file not found'}), 404
        
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        # Return the query and response so handleSessionClick can display them
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_user_librarian(username):
    """Get or create a user-specific Librarian instance"""
    user_rag_path = Path.home() / ".orchestra" / "users" / username / "documents" / "rag_index.json"
    user_rag_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create user-specific librarian
    user_librarian = orchestra.Librarian(config)
    user_librarian.index_path = str(user_rag_path)
    
    # Reload docs from user-specific path
    if user_rag_path.exists():
        try:
            with open(user_rag_path, 'r') as f:
                user_librarian.docs = json.load(f)
        except:
            user_librarian.docs = []
    
    return user_librarian

# ============================================
# SESSION MANAGEMENT (Side Panel)
# ============================================

@app.route('/api/sessions', methods=['GET'])
def get_user_sessions():
    """Load all saved chat sessions for a specific user"""
    username = request.args.get('username', 'Guest')
    search_query = request.args.get('search', None)  # NEW: Semantic search
    
    user_sessions_dir = Path.home() / ".orchestra" / "users" / username / "sessions"
    
    if not user_sessions_dir.exists():
        return jsonify([])

    sessions = []
    for file_path in user_sessions_dir.glob("*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                sessions.append({
                    'id': file_path.stem,
                    'title': data.get('title') or data.get('name') or "Untitled",
                    'timestamp': data.get('timestamp', ''),
                    'path': str(file_path),
                    'pinned': data.get('pinned', False),
                    'folder': data.get('folder', None),
                    'tags': data.get('tags', []),
                    'messages': data.get('messages', [])  # Include for semantic search
                })
        except:
            continue
    
    if search_query and len(search_query) > 3:
        try:
            import numpy as np
            
            # Get query embedding
            q_res = ollama.embeddings(model=config.embed_model, prompt=search_query)
            q_vec = np.array(q_res['embedding'])
            
            scored_sessions = []
            for session in sessions:
                # Create searchable text from session
                searchable = f"{session['title']} "
                
                # Add first few messages
                for msg in session.get('messages', [])[:10]:
                    searchable += msg.get('content', '')[:200] + " "
                
                # Get session embedding
                s_res = ollama.embeddings(model=config.embed_model, prompt=searchable[:2000])
                s_vec = np.array(s_res['embedding'])
                
                # Calculate similarity
                norm = np.linalg.norm(q_vec) * np.linalg.norm(s_vec)
                if norm > 0:
                    similarity = np.dot(q_vec, s_vec) / norm
                    scored_sessions.append((similarity, session))
            
            # Sort by similarity and filter threshold
            scored_sessions.sort(key=lambda x: x[0], reverse=True)
            sessions = [s[1] for s in scored_sessions if s[0] > 0.5][:20]
            
            print(f"🔍 Semantic search: '{search_query}' -> {len(sessions)} results")
            
        except Exception as e:
            print(f"Semantic search error: {e}")
            # Fall back to simple text search
            search_lower = search_query.lower()
            sessions = [s for s in sessions if search_lower in s['title'].lower()]
    
    for session in sessions:
        session.pop('messages', None)
    
    # Sort: pinned first, then by timestamp
    sessions.sort(key=lambda x: (not x.get('pinned', False), x.get('timestamp', '')), reverse=True)
    
    return jsonify(sessions)

@app.route('/api/sessions/load', methods=['GET'])
def load_session_content():
    path = request.args.get('path')
    if not path or not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404
        
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================
# CHAT / AI RESPONSES
# ============================================

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint with full Orchestra integration"""
    global current_user, active_engine
    data = request.json
    message = data.get('message', '')
    
    # --- [CRITICAL IDENTITY GUARD & GLOBAL SYNC] ---
    incoming_user = data.get('username', current_user)
    
    if not incoming_user or len(str(incoming_user)) > 50 or "\n" in str(incoming_user):
        active_user = "Guest"
    else:
        active_user = str(incoming_user)

    current_user = active_user 
    # -----------------------------------------------
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    memory_context = ""
    config.top_k = data.get('settings', {}).get('topK', config.top_k)
    
    # --- [UPDATED API MEMORY BLOCK] ---
    if active_user:
        memories = get_user_memories(message, active_user, max_results=3)
        if memories:
            # Changed header to emphasize "Historical Reference"
            memory_context = "\n\n[USER LONG-TERM MEMORY - REFERENCE ONLY]:\n"
            found_relevant = False
            for mem in memories:
                if mem.get('score', 1.0) > 0.82: # Tightened threshold slightly
                    memory_context += f"• Historical User Query: {mem['query']}\n"
                    # Truncate the response heavily so the Router doesn't get "drowned" in old data
                    memory_context += f"  Historical AI Response: {mem['response'][:150]}...\n\n"
                    found_relevant = True
            
            if not found_relevant:
                memory_context = "" # Clear context if nothing met the threshold
            else:
                print(f"DEBUG: Injected relevant memories for {active_user}")
    # ------------------------------------------------------
    
    # Create user-specific engine with sanitized active_user
    user_librarian = get_user_librarian(active_user) if active_user else orchestra.Librarian(config)
    user_engine = orchestra.OrchestraEngine(config, user_librarian)
    
    global pending_browser_context
    if active_engine and hasattr(active_engine, 'current_page_context') and active_engine.current_page_context:
        user_engine.current_page_context = active_engine.current_page_context
        print(f"🌐 DEBUG: Browser context preserved from previous engine - Page: {active_engine.current_page_context.get('title', 'Unknown')}")
    elif pending_browser_context:
        # Use the pending context from browser sync that happened before any engine existed
        user_engine.current_page_context = pending_browser_context
        print(f"🌐 DEBUG: Browser context loaded from pending queue - Page: {pending_browser_context.get('title', 'Unknown')}")
        pending_browser_context = None  # Clear it after using
    
    active_engine = user_engine
    user_engine.history = [] 
    frontend_history = data.get('history', [])
    
    # We look at the last 5 items to ensure we catch the previous complete exchange
    # before the current message.
    recent_context = frontend_history[-6:] if len(frontend_history) > 6 else frontend_history
    
    # Extract only the completed User/Assistant pairs
    # We iterate through the history and store them in the engine
    for i in range(len(recent_context)):
        if recent_context[i].get('role') == 'assistant':
            # Look for the user message that came before this assistant message
            if i > 0 and recent_context[i-1].get('role') == 'user':
                user_engine.history.append({
                    "user": recent_context[i-1]['content'],
                    "ai": recent_context[i]['content']
                })

    user_engine.history = user_engine.history[-4:]
    
    print(f"🔍 DEBUG: Successfully restored {len(user_engine.history)} complete exchanges.")
    # ---------------------------------------
    # -----------------------------------------------
    
    print(f"DEBUG: Using conductor: {config.conductor}")
    print(f"DEBUG: VRAM optimization: {config.vram_optim}") 

    try:
        response_data = {'text': '', 'experts': [], 'image': None, 'code_files': []}
        image_ready = threading.Event()
        image_ready.set()
        
        def capture_response(text, experts, code_files=None):
            response_data['text'] = text
            response_data['experts'] = experts
            if code_files:
                response_data['code_files'] = code_files  

        def capture_image(img_data):
            image_ready.clear()
            try:
                if img_data is None: return
                if hasattr(img_data, 'save'):
                    from io import BytesIO
                    buffer = BytesIO()
                    img_data.save(buffer, format='PNG')
                    img_bytes = buffer.getvalue()
                else:
                    img_bytes = img_data
                response_data['image'] = base64.b64encode(img_bytes).decode('utf-8')
                print(f"DEBUG: Image captured, size: {len(response_data['image'])} chars")
            except Exception as e:
                print(f"ERROR: Failed to capture image: {e}")
            finally:
                image_ready.set()
        
        class MockGUI:
            def after(self, delay, func): func()
            def show_render_status(self): pass
        
        username_to_run = active_user

        if message.lower().startswith("artisan"):
            image_ready.clear()
            print("DEBUG: Artisan detected - waiting for image generation")

        def run_engine():
            nonlocal username_to_run
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
               loop.run_until_complete(
                   user_engine.generate_response(message, capture_response, capture_image, MockGUI(), username_to_run)
               )
            finally:
                loop.close()
                print("DEBUG: Engine loop complete")

        thread = threading.Thread(target=run_engine)
        thread.start()

        # --- [FIX: ROBUST CONTENT POLLING] ---
        max_wait = 180 
        start_time = time.time()
        last_length = 0
        stable_count = 0
        
        while time.time() - start_time < max_wait:
            current_length = len(response_data['text'])
            if current_length > 0 and current_length == last_length:
                stable_count += 1
            else:
                stable_count = 0
            
            if (stable_count >= 3) or (not thread.is_alive() and current_length > 0):
                break
                
            if not thread.is_alive() and time.time() - start_time > 10: 
                break
                
            last_length = current_length
            time.sleep(0.5)
            
        thread.join(timeout=1)
        # ---------------------------------------

        if not image_ready.is_set():
            print("DEBUG: Waiting for image generation (CPU rendering takes 3-5 minutes)...")
            image_ready.wait(timeout=300) 
        
        print(f"DEBUG API: About to return response. Text Length: {len(response_data['text'])}")
        
        # Save to internal engine history before returning
        user_engine.history.append({
            'user': message,
            'ai': response_data['text'],
            'experts': response_data['experts'],
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'response': response_data['text'],
            'experts': response_data['experts'],
            'image': response_data['image'],
            'code_files': response_data.get('code_files', [])
        })
        
    except Exception as e:
        print(f"❌ Chat error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ============================================
# HARDWARE MONITORING
# ============================================

@app.route('/api/hardware', methods=['GET'])
def hardware():
    """Real-time hardware stats"""
    try:
        stats = hw_monitor.get_update()
        return jsonify(stats)
    except Exception as e:
        return jsonify({
            'cpu_pct': 0, 'ram_pct': 0, 'disk_pct': 0,
            'gpu_load': 0, 'vram_pct': 0, 'temp_pct': 0,
            'text': f'Error: {str(e)}'
        })


# ============================================
# RAG / DOCUMENT LIBRARY
# ============================================

@app.route('/api/library/upload', methods=['POST'])
def upload_document():
    """Upload and index a document for current user"""
    if not current_user:
        return jsonify({'error': 'Not logged in'}), 401
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / filename
        file.save(filepath)
        
        # Use user-specific librarian
        user_librarian = get_user_librarian(current_user)
        success = user_librarian.add_doc(str(filepath))
        
        if success:
            return jsonify({
                'success': True,
                'filename': filename,
                'indexed': True,
                'total_docs': len(user_librarian.docs)
            })
        else:
            return jsonify({'error': 'Failed to index document'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/library/upload-folder', methods=['POST'])
def upload_folder():
    if not current_user:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        files = request.files.getlist('files')
        user_librarian = get_user_librarian(current_user)
        
        indexed_count = 0
        failed = []
        
        for file in files:
            if file and file.filename:
                try:
                    # 1. Save the file first
                    filename = secure_filename(file.filename)
                    filepath = UPLOAD_FOLDER / filename
                    file.save(str(filepath))
                    
                    # 2. Add to librarian
                    # If this is slow, the batching in App.jsx will now handle it
                    success = user_librarian.add_doc(str(filepath))
                    if success:
                        indexed_count += 1
                    else:
                        failed.append(filename)
                except Exception as e:
                    print(f"⚠️ Error saving {file.filename}: {e}")
                    failed.append(file.filename)
        
        return jsonify({
            'success': True,
            'indexed': indexed_count,
            'total': len(files),
            'failed': failed
        })
    except Exception as e:
        print(f"🔥 Critical folder upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/library/search', methods=['POST'])
def search_library():
    """Search indexed documents for current user"""
    if not current_user:
        return jsonify({'error': 'Not logged in'}), 401
    
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        user_librarian = get_user_librarian(current_user)
        results = user_librarian.search(query)
        return jsonify({
            'results': results,
            'total_docs': len(user_librarian.docs)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/library/clear', methods=['POST'])
def clear_library():
    """Clear all indexed documents for current user"""
    if not current_user:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        user_librarian = get_user_librarian(current_user)
        user_librarian.clear_library()
        return jsonify({'success': True, 'message': 'Library cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/library/toggle-rag', methods=['POST'])
def toggle_rag():
    """Toggle RAG on/off (global setting)"""
    config.rag_enabled = not config.rag_enabled
    return jsonify({
        'rag_enabled': config.rag_enabled,
        'message': f"RAG {'enabled' if config.rag_enabled else 'disabled'}"
    })

@app.route('/api/library/stats', methods=['GET'])
def library_stats():
    """Get library statistics for current user"""
    if not current_user:
        return jsonify({
            'total_documents': 0,
            'rag_enabled': config.rag_enabled,
            'embed_model': config.embed_model
        })
    
    user_librarian = get_user_librarian(current_user)
    return jsonify({
        'total_documents': len(user_librarian.docs),
        'rag_enabled': config.rag_enabled,
        'embed_model': config.embed_model
    })

# ============================================
# SESSION MANAGEMENT
# ============================================

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    if not current_user:
        return jsonify({'sessions': []}), 401

    user_dir = Path.home() / ".orchestra" / "users" / current_user / "sessions"

    if not user_dir.exists():
        return jsonify({'sessions': []})

    sessions = []
    for file in user_dir.glob('*.json'):
        try:
            with open(file, 'r') as f:
                session = json.load(f)
                sessions.append({
                    'id': file.stem,
                    'name': session.get('name', file.stem),
                    'timestamp': session.get('timestamp', ''),
                    'message_count': len(session.get('history', []))
                })
        except:
            continue

    sessions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return jsonify({'sessions': sessions})


@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get specific session"""
    try:
        session_file = SESSIONS_DIR / f"{session_id}.json"
        if not session_file.exists():
            return jsonify({'error': 'Session not found'}), 404
        
        with open(session_file, 'r') as f:
            session = json.load(f)
        
        return jsonify(session)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions', methods=['POST'])
def save_session():
    """Save current session"""
    data = request.json
    name = data.get('name', f"Session {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    try:
        session_id = name.lower().replace(' ', '_')
        session_data = {
            'id': session_id,
            'name': name,
            'history': engine.history,
            'timestamp': datetime.now().isoformat()
        }
        
        session_file = SESSIONS_DIR / f"{session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Session saved'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    # Fallback to 'Guest' if no username is provided
    username = request.args.get('username', 'Guest')
    
    # FIX: Point to "sessions" folder to match your save/load logic
    user_sessions_dir = Path.home() / ".orchestra" / "users" / username / "sessions"
    file_path = user_sessions_dir / f"{session_id}.json"
    
    try:
        if file_path.exists():
            file_path.unlink() # Deletes the file
            print(f"🗑️ Deleted session: {session_id} for user: {username}")
            return jsonify({'success': True})
        else:
            print(f"❌ Delete failed: {file_path} not found")
            return jsonify({'error': 'Session not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions/current/clear', methods=['POST'])
def clear_current_session():
    """Clear current chat history"""
    engine.history = []
    return jsonify({'success': True, 'message': 'Current session cleared'})

@app.route('/api/sessions/save', methods=['POST'])
def save_chat_session():
    """Save session to user-specific folder with metadata support"""
    data = request.json
    username = data.get('username', 'Guest')
    session_id = data.get('id') or f"sess_{int(time.time())}"
    title = data.get('title', 'New Session')
    messages = data.get('messages', [])
    
    # NEW: Session metadata
    pinned = data.get('pinned', False)
    folder = data.get('folder', None)  # e.g., "Work", "Personal", "Python Project"
    tags = data.get('tags', [])
    
    if not messages:
        return jsonify({'error': 'No messages to save'}), 400
    
    user_sessions_dir = Path.home() / ".orchestra" / "users" / username / "sessions"
    user_sessions_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        session_data = {
            'id': session_id,
            'title': title,
            'messages': messages,
            'username': username,
            'timestamp': datetime.now().isoformat(),
            'pinned': pinned,
            'folder': folder,
            'tags': tags
        }
        
        file_path = user_sessions_dir / f"{session_id}.json"
        with open(file_path, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        return jsonify({'success': True, 'id': session_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions/<session_id>/metadata', methods=['PATCH'])
def update_session_metadata(session_id):
    """Update session metadata (pin, folder, tags) without rewriting messages"""
    username = request.args.get('username', 'Guest')
    data = request.json
    
    user_sessions_dir = Path.home() / ".orchestra" / "users" / username / "sessions"
    file_path = user_sessions_dir / f"{session_id}.json"
    
    if not file_path.exists():
        return jsonify({'error': 'Session not found'}), 404
    
    try:
        # Load existing session
        with open(file_path, 'r') as f:
            session_data = json.load(f)
        
        # Update only metadata fields
        if 'pinned' in data:
            session_data['pinned'] = data['pinned']
        if 'folder' in data:
            session_data['folder'] = data['folder']
        if 'tags' in data:
            session_data['tags'] = data['tags']
        if 'title' in data:
            session_data['title'] = data['title']
        
        # Save back
        with open(file_path, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions/<session_id>/fork', methods=['POST'])
def fork_session(session_id):
    """
    Create a new session branching from a specific point in the conversation.
    
    This allows users to explore different conversational paths without
    losing the original session.
    """
    username = request.args.get('username', 'Guest')
    data = request.json
    fork_at_index = data.get('fork_at_index', None)  # Message index to branch from
    
    user_sessions_dir = Path.home() / ".orchestra" / "users" / username / "sessions"
    source_file = user_sessions_dir / f"{session_id}.json"
    
    if not source_file.exists():
        return jsonify({'error': 'Source session not found'}), 404
    
    try:
        # Load source session
        with open(source_file, 'r') as f:
            source_data = json.load(f)
        
        # Create new session with forked data
        new_id = f"fork_{session_id}_{int(time.time())}"
        
        # Copy messages up to fork point (or all if not specified)
        messages = source_data.get('messages', [])
        if fork_at_index is not None:
            messages = messages[:fork_at_index + 1]
        
        fork_data = {
            'id': new_id,
            'title': f"🔀 {source_data.get('title', 'Forked Chat')}",
            'messages': messages,
            'username': username,
            'timestamp': datetime.now().isoformat(),
            'pinned': False,
            'folder': source_data.get('folder'),
            'tags': source_data.get('tags', []) + ['forked'],
            'parent_session': session_id,
            'fork_point': fork_at_index
        }
        
        # Save forked session
        fork_file = user_sessions_dir / f"{new_id}.json"
        with open(fork_file, 'w') as f:
            json.dump(fork_data, f, indent=2)
        
        print(f"🔀 Forked session: {session_id} -> {new_id} (at index {fork_at_index})")
        
        return jsonify({
            'success': True,
            'new_session_id': new_id,
            'title': fork_data['title']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/api/sessions/generate-title', methods=['POST'])
def generate_session_title():
    """Generate a concise AI title from the first few messages"""
    data = request.json
    messages = data.get('messages', [])
    
    if not messages or len(messages) < 2:
        return jsonify({'title': 'New Chat'})
    
    # Extract first user message and AI response
    first_user = next((m['content'] for m in messages if m['role'] == 'user'), '')
    first_ai = next((m['content'] for m in messages if m['role'] == 'assistant'), '')
    
    if not first_user:
        return jsonify({'title': 'New Chat'})
    
    # Create a concise summarization prompt
    prompt = f"""Generate a 2-4 word title for this conversation. Be concise and descriptive.

User: {first_user[:200]}
Assistant: {first_ai[:200]}

Title (2-4 words only):"""
    
    try:
        # Use the conductor model for speed (it's already loaded)
        res = ollama.generate(
            model=config.conductor,
            prompt=prompt,
            options={
                "temperature": 0.3,  # Low temp for consistency
                "num_predict": 20,   # Max 20 tokens
                "stop": ["\n", ".", "User:", "Assistant:"]
            }
        )
        
        title = res['response'].strip()
        
        # Clean up the title
        title = title.replace('"', '').replace("'", '').strip()
        
        # Fallback if too long or empty
        if len(title) > 50 or len(title) < 3:
            title = first_user[:30] + "..."
        
        print(f"🏷️ Generated title: '{title}'")
        return jsonify({'title': title})
        
    except Exception as e:
        print(f"Title generation error: {e}")
        # Fallback to first few words of user message
        title = ' '.join(first_user.split()[:4]) + "..."
        return jsonify({'title': title})


# ============================================
# SETTINGS & CONFIGURATION
# ============================================

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get current settings"""
    experts_dict = config.experts.copy()
    # Always keep Artisan as LOCAL_EMBEDDED (Stable Diffusion)
    if 'Artisan_Illustrator' in experts_dict:
        experts_dict['Artisan_Illustrator'] = 'LOCAL_EMBEDDED'
    
    return jsonify({
        'rag_enabled': config.rag_enabled,
        'vram_optim': config.vram_optim,
        'top_k': config.top_k,
        'conductor': config.conductor,
        'experts': experts_dict,
        'theme_mode': config.theme_mode,
        'auto_archive_webpages': getattr(config, 'auto_archive_webpages', True)  # Default to True
    })

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update settings"""
    data = request.json
    
    try:
        if 'rag_enabled' in data:
            config.rag_enabled = data['rag_enabled']
        if 'vram_optim' in data:
            config.vram_optim = data['vram_optim']
        if 'top_k' in data:
            config.top_k = data['top_k']
        if 'auto_archive_webpages' in data:
            config.auto_archive_webpages = data['auto_archive_webpages']
            print(f"DEBUG: Auto-archive webpages set to {data['auto_archive_webpages']}")
        
        # CRITICAL: Unload old conductor before switching to new one
        if 'conductor' in data and data['conductor'] != config.conductor:
            old_conductor = config.conductor
            new_conductor = data['conductor']
            
            print(f"DEBUG: Switching conductor from {old_conductor} to {new_conductor}")
            
            try:
                # Force unload old conductor to free VRAM
                import subprocess
                result = subprocess.run(
                    ['ollama', 'stop', old_conductor],
                    timeout=5,
                    capture_output=True,
                    text=True
                )
                print(f"DEBUG: Unloaded {old_conductor}")
                time.sleep(2)  # Give VRAM time to clear
            except Exception as e:
                print(f"WARNING: Failed to unload {old_conductor}: {e}")
            
            # Now update to new conductor
            config.conductor = new_conductor
        
        if 'experts' in data:
            for expert, model in data['experts'].items():
                # Skip special non-Ollama experts
                if expert not in ['Artisan_Illustrator', 'Chess_Analyst']:
                    config.experts[expert] = model
        
        return jsonify({'success': True, 'message': 'Settings updated'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================
# PROGRAM LAUNCHER & WEB SEARCH
# ============================================

@app.route('/api/web-search', methods=['POST'])
def web_search():
    """Perform web search"""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        results = engine.web_search.search(query, max_results=5)
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sync_web', methods=['POST'])
def sync_web():
    global active_engine
    data = request.json
    
    if active_engine is None:
        # If no engine is active yet, we just acknowledge but don't crash
        return jsonify({"status": "No active engine session"}), 200
        
    try:
        # Use the global reference
        status = asyncio.run(active_engine.get_web_context(data)) 
        return jsonify({"status": status})
    except Exception as e:
        return jsonify({"status": f"Sync Error: {str(e)}"}), 500

# ============================================
# EXPERT USAGE STATS
# ============================================

@app.route('/api/stats/experts', methods=['GET'])
def expert_stats():
    """Get expert usage statistics"""
    return jsonify({
        'usage': config.expert_usage_stats,
        'total_queries': sum(config.expert_usage_stats.values())
    })

@app.route('/api/browser_sync', methods=['POST', 'OPTIONS'])
def browser_sync():
    """Receive context from Electron BrowserView and update the AI engine"""
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200
        
    data = request.json
    global active_engine, pending_browser_context, current_user
    
    print(f"🌐 BROWSER SYNC: Received data - Title: {data.get('title', 'None')}, URL: {data.get('url', 'None')}")
    print(f"🌐 BROWSER SYNC: Content length: {len(data.get('content', ''))} chars")
    print(f"🌐 BROWSER SYNC: active_engine exists: {active_engine is not None}")
    
    # Store context for immediate use
    if not active_engine:
        print("⚠️ BROWSER SYNC: No active engine - context stored but will be used on next chat")
        pending_browser_context = data
    else:
        try:
            asyncio.run(active_engine.get_web_context(data))
            print(f"✅ BROWSER SYNC: Context stored in active_engine")
        except Exception as e:
            print(f"❌ BROWSER SYNC ERROR: {e}")
    
    # NEW: Auto-index webpages into RAG for long-term memory
    # Check if auto-archiving is enabled
    auto_archive_enabled = getattr(config, 'auto_archive_webpages', True)
    
    if auto_archive_enabled and active_engine:
        url = data.get('url', '')
        title = data.get('title', '')
        content = data.get('content', '')
        
        # Skip certain URLs (localhost, blank pages, error pages)
        skip_patterns = ['localhost', '127.0.0.1', 'about:', 'chrome:', 'file://', 'data:']
        should_skip = any(pattern in url.lower() for pattern in skip_patterns)
        
        if not should_skip and content and len(content) > 200:
            try:
                # Use the active engine's librarian to index the webpage
                success = active_engine.lib.add_webpage(url, title, content, auto_save=True)
                if success:
                    print(f"📚 Auto-archived: {title}")
                    return jsonify({
                        "status": "Sync Success + Archived",
                        "message": f"Page archived to knowledge base"
                    }), 200
            except Exception as e:
                print(f"📄 Auto-archive error: {e}")
                import traceback
                traceback.print_exc()
                # Don't fail the whole sync if archiving fails
    
    return jsonify({"status": "Sync Success"}), 200

# ============================================
# CODE EXECUTION (SECURITY: Requires User Confirmation)
# ============================================

@app.route('/api/detect_code', methods=['POST'])
def detect_code_blocks():
    """
    Detect code blocks in LLM response without executing them.
    
    SECURITY: This endpoint is safe - it only detects, never executes.
    Returns code blocks for user review.
    """
    try:
        data = request.json
        response_text = data.get('response_text', '')
        
        # Safe detection only - NO execution
        code_blocks = code_executor.process_response(response_text)
        
        print(f"[CODE_DETECT] Found {len(code_blocks)} code block(s)")
        
        return jsonify({
            'success': True,
            'code_blocks': code_blocks,
            'count': len(code_blocks)
        })
    
    except Exception as e:
        print(f"[CODE_DETECT] Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/execute_code', methods=['POST'])
def execute_code_with_confirmation():
    """
    Execute code ONLY after explicit user confirmation.
    
    SECURITY: Requires user_confirmed=True flag to execute.
    Logs all execution attempts for audit trail.
    """
    try:
        data = request.json
        
        language = data.get('language')
        code = data.get('code')
        user_confirmed = data.get('user_confirmed', False)
        
        # Validate required fields
        if not language or not code:
            return jsonify({
                'success': False,
                'error': 'Missing required fields: language and code'
            }), 400
        
        # SECURITY: Log all execution attempts
        print(f"[CODE_EXEC] Attempt to execute {language} code. Confirmed: {user_confirmed}")
        print(f"[CODE_EXEC] Code length: {len(code)} characters")
        
        # Execute only if confirmed
        result = code_executor.execute_with_confirmation(
            language,
            code,
            user_confirmed=user_confirmed
        )
        
        # Log result
        if result.get('success'):
            print(f"[CODE_EXEC] ✅ Execution successful")
        else:
            print(f"[CODE_EXEC] ❌ Execution failed or denied: {result.get('error')}")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"[CODE_EXEC] Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================
# BROWSER DATA MANAGEMENT
# ============================================

# Storage paths
BROWSER_DATA_DIR = Path.home() / ".orchestra_browser_data"
BROWSER_DATA_DIR.mkdir(exist_ok=True)

def get_user_browser_file(username, data_type):
    """Get path to user's browser data file"""
    return BROWSER_DATA_DIR / f"{username}_{data_type}.json"

def load_browser_data(username, data_type):
    """Load browser data for a user"""
    filepath = get_user_browser_file(username, data_type)
    if filepath.exists():
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_browser_data(username, data_type, data):
    """Save browser data for a user"""
    filepath = get_user_browser_file(username, data_type)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

@app.route('/api/browser/bookmarks', methods=['GET', 'POST', 'DELETE'])
def browser_bookmarks():
    """Manage browser bookmarks"""
    global current_user
    
    if not current_user:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if request.method == 'GET':
        bookmarks = load_browser_data(current_user, 'bookmarks')
        return jsonify({'bookmarks': bookmarks})
    
    elif request.method == 'POST':
        bookmark = request.json
        bookmarks = load_browser_data(current_user, 'bookmarks')
        
        # Check for duplicates
        if not any(b['url'] == bookmark['url'] for b in bookmarks):
            bookmarks.append(bookmark)
            save_browser_data(current_user, 'bookmarks', bookmarks)
        
        return jsonify({'bookmarks': bookmarks})
    
    elif request.method == 'DELETE':
        url = request.json.get('url')
        bookmarks = load_browser_data(current_user, 'bookmarks')
        bookmarks = [b for b in bookmarks if b['url'] != url]
        save_browser_data(current_user, 'bookmarks', bookmarks)
        return jsonify({'bookmarks': bookmarks})

@app.route('/api/browser/history', methods=['GET', 'POST', 'DELETE'])
def browser_history():
    """Manage browser history"""
    global current_user
    
    if not current_user:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if request.method == 'GET':
        history = load_browser_data(current_user, 'history')
        return jsonify({'history': history})
    
    elif request.method == 'POST':
        history_item = request.json
        history = load_browser_data(current_user, 'history')
        
        # Add to history (keep last 1000)
        history.insert(0, history_item)
        history = history[:1000]
        
        save_browser_data(current_user, 'history', history)
        return jsonify({'status': 'saved'})
    
    elif request.method == 'DELETE':
        # Clear all history
        save_browser_data(current_user, 'history', [])
        return jsonify({'history': []})

@app.route('/api/browser/downloads', methods=['GET', 'POST'])
def browser_downloads():
    """Manage browser downloads"""
    global current_user
    
    if not current_user:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if request.method == 'GET':
        downloads = load_browser_data(current_user, 'downloads')
        return jsonify({'downloads': downloads})
    
    elif request.method == 'POST':
        downloads_data = request.json.get('downloads', [])
        save_browser_data(current_user, 'downloads', downloads_data)
        return jsonify({'status': 'saved'})

# ============================================
# CODE EXECUTION RESULT VIEWER
# ============================================

@app.route('/api/view_result/<filename>', methods=['GET'])
def view_result(filename):
    """
    Serve code execution result files.
    
    SECURITY: Only serves files from /tmp that match the pattern
    tmp*.html (code execution results created by CodeExecutor).
    This prevents arbitrary file access.
    """
    try:
        # SECURITY: Validate filename pattern
        # Only allow tmpXXXXXX.html files (code execution results)
        if not filename.startswith('tmp') or not filename.endswith('.html'):
            return jsonify({'error': 'Invalid filename pattern'}), 403
        
        # SECURITY: Prevent path traversal attacks
        # Remove any path separators
        filename = os.path.basename(filename)
        
        # Construct safe filepath
        filepath = os.path.join('/tmp', filename)
        
        # SECURITY: Verify file exists and is in /tmp
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        if not os.path.realpath(filepath).startswith('/tmp/'):
            return jsonify({'error': 'Access denied'}), 403
        
        # Serve the file
        return send_file(filepath, mimetype='text/html')
    
    except Exception as e:
        print(f"[VIEW_RESULT] Error: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================
# START SERVER
# ============================================

if __name__ == '__main__':
    print("""
    ========================================
       ORCHESTRA API BACKEND v2.9
       All Features Connected
    ========================================
    """)
    print(f"🤖 Models available: {len(config.available_models)}")
    print(f"📚 RAG: User-specific libraries enabled")
    print(f"⚙️  Conductor: {config.conductor}")
    print(f"⚙️  VRAM Optim: {config.vram_optim}")
    print("🚀 Starting Flask server on port 5000...")
    
    try:
        app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
    except Exception as e:
        print(f"❌ FLASK CRASHED: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
