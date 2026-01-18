import HardwareMonitor from './HardwareMonitor';
import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send, Settings, Menu, X, ChevronRight, Upload, FolderUp, Trash2, Save,
  MessageSquare, Terminal, Gauge, FileText, Database, Shield, Cloud, DollarSign,
  BookOpen, Users, Briefcase, Brain, Eye, Lightbulb, Cpu, Globe, Plus, Crown
} from 'lucide-react';
import { v4 as uuidv4 } from 'uuid';
import BrowserView from './BrowserView';
import LoginScreen from './Login';
import CodeExecutionDialog from './CodeExecutionDialog';  // SECURITY: Import code execution dialog

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [leftSidebarOpen, setLeftSidebarOpen] = useState(true);
  const [isSidebarLoading, setIsSidebarLoading] = useState(false);
  const [rightSidebarOpen, setRightSidebarOpen] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState({
    theme: 'dark',
    ragEnabled: true,
    vramOptim: true,
    topK: 3,
    conductor: '',
    experts: {}
  });
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [sessionSearchQuery, setSessionSearchQuery] = useState(''); // NEW: Search query
  const [sessionFolders, setSessionFolders] = useState({}); // NEW: Folder organization
  const [availableModels, setAvailableModels] = useState([]);
  const [expertsList, setExpertsList] = useState([]);
  const [activeExperts, setActiveExperts] = useState([]);
  const [hardwareStats, setHardwareStats] = useState({ cpu: 0, ram: 0, gpu: 0, vram: 0 });
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);
  const [savedMemoryId, setSavedMemoryId] = useState(null);  // Add this line
  const [ragStats, setRagStats] = useState({ total_documents: 0, rag_enabled: false });
  const [pendingCodeExecution, setPendingCodeExecution] = useState(null);  // SECURITY: Code execution state
  
  // Tab management state
  const [tabs, setTabs] = useState([
    { id: 'chat', title: 'Chat', type: 'chat', closable: false }
  ]);
  const [activeTab, setActiveTab] = useState('chat');
  
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const folderInputRef = useRef(null);

  // Icon mapping for experts
  const expertIcons = {
    'Code_Logic': Terminal,
    'STEM_Expert': Gauge,
    'Creative_Writer': FileText,
    'Data_Scientist': Database,
    'Cyber_Security': Shield,
    'DevOps_Cloud': Cloud,
    'Finance_Analyst': DollarSign,
    'History_Expert': BookOpen,
    'SQL_Database': Database,
    'Language_Linguist': Globe,
    'Legal_Counsel': FileText,
    'Medical_Expert': FileText,
    'Network_Engineer': Globe,
    'Philosophy_Arts': FileText,
    'Neural_Network_Engineer': Cpu,
    'Psychology_Counselor': Users,
    'Business_Strategist': Briefcase,
    'Research_Scientist': FileText,
    'Vision_Analyst': Eye,
    'Reasoning_Expert': Lightbulb,
    'Chess_Analyst': Crown,
    'Math_Expert': Brain,
  };

  const expertColors = {
    'Code_Logic': 'text-cyber-accent',
    'STEM_Expert': 'text-cyber-purple',
    'Creative_Writer': 'text-pink-400',
    'Data_Scientist': 'text-yellow-400',
    'Cyber_Security': 'text-red-400',
    'DevOps_Cloud': 'text-blue-400',
    'Finance_Analyst': 'text-green-400',
    'History_Expert': 'text-orange-400',
    'SQL_Database': 'text-purple-400',
    'Language_Linguist': 'text-cyan-400',
    'Legal_Counsel': 'text-indigo-400',
    'Medical_Expert': 'text-emerald-400',
    'Network_Engineer': 'text-teal-400',
    'Philosophy_Arts': 'text-violet-400',
    'Neural_Network_Engineer': 'text-fuchsia-400',
    'Psychology_Counselor': 'text-rose-400',
    'Business_Strategist': 'text-amber-400',
    'Research_Scientist': 'text-lime-400',
    'Vision_Analyst': 'text-sky-400',
    'Reasoning_Expert': 'text-pink-400',
    'Chess_Analyst': 'text-red-500',
    'Math_Expert': 'text-cyan-500',
  };

  useEffect(() => {
    checkLoginStatus();
    const interval = setInterval(loadHardwareStats, 2000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (isLoggedIn) {
      loadModels();
      loadSessions();
      loadSettings();
      loadRagStats();
    }
  }, [isLoggedIn]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    // Only auto-save if we are already inside a named session 
    // and there is actually content to save.
    if (currentSessionId && messages.length > 0) {
      const activeSession = sessions.find(s => s.id === currentSessionId);
      if (activeSession) {
        saveSession(activeSession.title);
      }
    }
  }, [messages]); // This fires every time the message list changes

  // Load user-specific settings from localStorage
  useEffect(() => {
    if (currentUser) {
      const savedSettings = localStorage.getItem(`orchestra_settings_${currentUser}`);
      if (savedSettings) {
        const parsed = JSON.parse(savedSettings);
        setSettings(prevSettings => ({
          ...prevSettings,
          ...parsed
        }));
      }
    }
  }, [currentUser]);

  const checkLoginStatus = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/auth/status');
      const data = await response.json();
      setIsLoggedIn(data.logged_in);
      setCurrentUser(data.username);
    } catch (error) {
      console.error('Failed to check login status:', error);
    }
  };

  const handleLogin = async (username, password) => {
    try {
      const response = await fetch('http://localhost:5000/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });
      const data = await response.json();
      if (data.success) {
        setIsLoggedIn(true);
        setCurrentUser(username);
        return { success: true };
      }
      return { success: false, error: data.error };
    } catch (error) {
      return { success: false, error: 'Connection failed' };
    }
  };

  const handleLogout = async () => {
    try {
      await fetch('http://localhost:5000/api/auth/logout', { method: 'POST' });
      setIsLoggedIn(false);
      setCurrentUser(null);
      setMessages([]);
      setSessions([]);
    } catch (error) {
      console.error('Logout failed:', error);
    }
  };

  const loadModels = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/models');
      const data = await response.json();
      setAvailableModels(data.available_models || []);
      setExpertsList(Object.entries(data.experts || {}).map(([name, model]) => ({
        name,
        model,
        displayName: name.replace(/_/g, ' ')
      })));
    } catch (error) {
      console.error('Failed to load models:', error);
    }
  };

  const loadSettings = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/settings');
      const data = await response.json();
      setSettings({
        theme: data.theme_mode || 'dark',
        ragEnabled: data.rag_enabled ?? true,
        vramOptim: data.vram_optim ?? true,
        topK: data.top_k || 3,
        conductor: data.conductor || '',
        experts: data.experts || {}
      });
    } catch (error) {
      console.error('Failed to load settings:', error);
    }
  };

  const saveSettings = async () => {
  try {
    // 1. Save to backend
    await fetch('http://127.0.0.1:5000/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings)
    });

    // 2. UPDATE THE LOCAL EXPERTS LIST IMMEDIATELY
    const updatedExpertsList = Object.entries(settings.experts).map(([name, model]) => ({
      name,
      model,
      displayName: name.replace(/_/g, ' ')
    }));
    setExpertsList(updatedExpertsList);

    // 3. Save to localStorage
    if (currentUser) {
      localStorage.setItem(`orchestra_settings_${currentUser}`, JSON.stringify(settings));
    }
    
    setShowSettings(false);
    alert('Settings saved successfully!');
  } catch (error) {
    console.error('Failed to save settings:', error);
    alert('Failed to save settings');
  }
};

  const loadSessions = async () => {
    if (!currentUser) return;

    setIsSidebarLoading(true);
    try {
      // NEW: Support semantic search
      const params = new URLSearchParams({ username: currentUser });
      if (sessionSearchQuery && sessionSearchQuery.length > 3) {
        params.append('search', sessionSearchQuery);
      }
      
      const response = await fetch(`http://localhost:5000/api/sessions?${params}`);
      const data = await response.json();
      
      setSessions(Array.isArray(data) ? data : []);
      
      const folders = data.reduce((acc, session) => {
        const folder = session.folder || 'Uncategorized';
        if (!acc[folder]) acc[folder] = [];
        acc[folder].push(session);
        return acc;
      }, {});
      setSessionFolders(folders);
      
      console.log("✅ Sidebar sessions updated:", data.length);
    } catch (error) {
      console.error('Failed to load sessions:', error);
    } finally {
      setIsSidebarLoading(false);
    }
  };

  const handleSessionClick = async (session) => {
    setIsSidebarLoading(true);
    try {
      const response = await fetch(`http://localhost:5000/api/sessions/load?path=${encodeURIComponent(session.path)}`);
      const data = await response.json();
    
      if (data.messages) {
        setCurrentSessionId(data.id);
        setMessages(data.messages.map(m => ({
          ...m,
          id: m.id || uuidv4()
        })));
      }
    } catch (error) {
      console.error("Error loading session:", error);
    } finally {
      setIsSidebarLoading(false);
    }
  };

  const saveSession = async (name) => {
    const sessionId = currentSessionId || uuidv4();
  
    const sessionPayload = {
      id: sessionId,
      title: name, 
      messages: messages, 
      username: currentUser,
      timestamp: new Date().toISOString()
    };

    try {
      const response = await fetch('http://localhost:5000/api/sessions/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sessionPayload),
      });

      if (response.ok) {
        // Very important: set this so the NEXT save knows it's the same session
        setCurrentSessionId(sessionId); 
        await loadSessions(); 
        console.log("✅ Session updated successfully.");
      }
    } catch (error) {
      console.error('Update failed:', error);
    }
  };

  // NEW: Auto-generate session title after 2nd message
  const generateSessionTitle = async (messages) => {
    try {
      const response = await fetch('http://localhost:5000/api/sessions/generate-title', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: messages.slice(0, 4) }) // First 2 exchanges
      });
      
      const { title } = await response.json();
      console.log('🏷️ Auto-generated title:', title);
      
      // Auto-save with generated title
      if (title && title !== 'New Chat') {
        await saveSession(title);
      }
      
      return title;
    } catch (error) {
      console.error('Title generation failed:', error);
      return null;
    }
  };

  // NEW: Auto-generate title after 2nd assistant message
  useEffect(() => {
    const assistantMessages = messages.filter(m => m.role === 'assistant');
    
    // Trigger auto-title after 2nd AI response (4 total messages)
    if (assistantMessages.length === 2 && !currentSessionId) {
      generateSessionTitle(messages);
    }
  }, [messages]);

  // NEW: Reload sessions when search query changes
  useEffect(() => {
    const debounceTimer = setTimeout(() => {
      if (sessionSearchQuery !== undefined) {
        loadSessions();
      }
    }, 300); // Debounce search
    
    return () => clearTimeout(debounceTimer);
  }, [sessionSearchQuery]);

  const deleteSession = async (sessionId) => {
    // Add a confirmation to prevent accidental deletions
    if (!confirm('Are you sure you want to delete this session?')) return;

    try {
      // FIX: Pass the username as a query parameter (?username=...)
      // This allows the backend to find the correct folder
      const response = await fetch(`http://localhost:5000/api/sessions/${sessionId}?username=${currentUser}`, { 
        method: 'DELETE' 
      });
    
      if (response.ok) {
        await loadSessions(); // Refresh the sidebar list immediately
      
        // If the chat you just deleted is the one currently on screen, clear the screen
        if (currentSessionId === sessionId) {
          setMessages([]);
          setCurrentSessionId(null);
        }
        console.log("✅ Session deleted successfully");
      } else {
        console.error('Server error during delete:', await response.text());
      }
    } catch (error) {
      console.error('Failed to delete session:', error);
    }
  };

  // NEW: Toggle pin status
  const togglePin = async (sessionId, currentPinned) => {
    try {
      const response = await fetch(
        `http://localhost:5000/api/sessions/${sessionId}/metadata?username=${currentUser}`,
        {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ pinned: !currentPinned })
        }
      );
      
      if (response.ok) {
        await loadSessions();
        console.log(`📌 Session ${!currentPinned ? 'pinned' : 'unpinned'}`);
      }
    } catch (error) {
      console.error('Failed to toggle pin:', error);
    }
  };

  // NEW: Fork session to create a branch
  const forkSession = async (sessionId, atIndex = null) => {
    try {
      const response = await fetch(
        `http://localhost:5000/api/sessions/${sessionId}/fork?username=${currentUser}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ fork_at_index: atIndex })
        }
      );
      
      const data = await response.json();
      if (data.success) {
        console.log('🔀 Forked session:', data.new_session_id);
        await loadSessions();
        // Optionally: Load the forked session
        // handleSessionClick({ id: data.new_session_id, ... });
      }
    } catch (error) {
      console.error('Failed to fork session:', error);
    }
  };

  const loadHardwareStats = async () => {
  try {
    const response = await fetch('http://localhost:5000/api/hardware');
    const data = await response.json();
    console.log('Hardware stats:', data, 'State will be:', data.cpu, data.ram, data.gpu, data.vram);
    setHardwareStats({
      cpu: data.cpu_pct || 0,
      ram: data.ram_pct || 0,
      gpu: data.gpu_load || 0,
      vram: data.vram_pct || 0
    });
  } catch (error) {
    console.error('Failed to load hardware stats:', error);
  }
};

  const loadRagStats = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/library/stats');
      const data = await response.json();
      setRagStats(data);
    } catch (error) {
      console.error('Failed to load RAG stats:', error);
    }
  };

  const handleUpload = async (files) => {
    const formData = new FormData();
    Array.from(files).forEach(file => formData.append('file', file));
    
    try {
      const response = await fetch('http://localhost:5000/api/library/upload', {
        method: 'POST',
        body: formData
      });
      await loadRagStats();
    } catch (error) {
      console.error('Upload failed:', error);
    }
  };

  const handleFolderUpload = async (files) => {
    const fileList = Array.from(files);
    if (fileList.length === 0) return;

    // 1. Start loading state to show the spinner
    setIsSidebarLoading(true);
    console.log(`🚀 Starting batch upload for ${fileList.length} files...`);

    try {
      // 2. Process files in batches of 5
      // This prevents the UI from freezing and avoids server timeouts
      const batchSize = 5;
      for (let i = 0; i < fileList.length; i += batchSize) {
        const batch = fileList.slice(i, i + batchSize);
      
        await Promise.all(batch.map(async (file) => {
          const formData = new FormData();
          // Use your single-file 'upload' endpoint which is more stable
          formData.append('file', file);
        
          const response = await fetch('http://localhost:5000/api/library/upload', {
            method: 'POST',
            body: formData,
          });

          if (!response.ok) {
            console.warn(`Failed to upload: ${file.name}`);
          }
        }));

        console.log(`✅ Progress: ${Math.min(i + batchSize, fileList.length)} / ${fileList.length}`);
      }

      // 3. Final refresh after all files are processed
      await loadRagStats();
      alert(`Successfully processed folder: ${fileList.length} files added to library.`);

    } catch (error) {
      console.error('Folder upload encountered errors:', error);
    } finally {
      // 4. Always turn off the loading spinner
      setIsSidebarLoading(false);
      // Clear the input so you can upload the same folder again if needed
      if (folderInputRef.current) folderInputRef.current.value = '';
    }
  };
  const clearLibrary = async () => {
    if (!confirm('Clear all uploaded documents?')) return;
    try {
      await fetch('http://localhost:5000/api/library/clear', { method: 'POST' });
      await loadRagStats();
    } catch (error) {
      console.error('Failed to clear library:', error);
    }
  };

  const toggleRag = async () => {
    try {
      await fetch('http://localhost:5000/api/library/toggle-rag', { method: 'POST' });
      await loadSettings();
      await loadRagStats();
    } catch (error) {
      console.error('Failed to toggle RAG:', error);
    }
  };

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInput('');

    try {
      const response = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          // Identity Guard: Ensures prompts don't leak into the username field
          username: (currentUser && currentUser.length < 50 && !currentUser.includes('\n')) ? currentUser : "Guest",
          message: input, 
          history: updatedMessages 
        })
      });

      const data = await response.json();

      const assistantMessage = {
        role: 'assistant',
        content: data.response,
        experts: data.experts || [],
        image: data.image || null 
      };

      setMessages(prev => [...prev, assistantMessage]);
    
      setActiveExperts(data.experts || []);
      
      // SECURITY: Check for code blocks in response
      try {
        const codeCheckResponse = await fetch('http://localhost:5000/api/detect_code', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ response_text: data.response })
        });
        
        const codeData = await codeCheckResponse.json();
        
        // If code blocks found, show confirmation dialog for first one
        if (codeData.success && codeData.code_blocks && codeData.code_blocks.length > 0) {
          console.log('[SECURITY] Code blocks detected:', codeData.code_blocks.length);
          setPendingCodeExecution(codeData.code_blocks[0]);
        }
      } catch (codeCheckError) {
        console.error('[SECURITY] Failed to check for code blocks:', codeCheckError);
        // Don't fail the whole response if code check fails
      }
      
      if (data.code_files && data.code_files.length > 0) {
        data.code_files.forEach(filepath => {
          const newTab = {
            id: 'code-' + Date.now() + '-' + Math.random(),
            title: filepath.split('/').pop(),
            type: 'browser',
            url: 'file://' + filepath,
            closable: true
          };
          setTabs(prev => [...prev, newTab]);
          setActiveTab(newTab.id);
        });
      }

    } catch (error) {
      console.error('Failed to send message:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Error: Failed to get response',
        experts: []
      }]);
    }
  };

  const saveMemory = async (query, response, experts) => {
    try {
      await fetch('http://localhost:5000/api/memory/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          // Use the same Guard here to prevent folder name errors on save
          username: (currentUser && currentUser.length < 50 && !currentUser.includes('\n')) ? currentUser : "Guest",
          query, 
          response, 
          experts 
        })
      });
    } catch (error) {
      console.error('Failed to save memory:', error);
    }
  };
  // Tab management functions
  const addBrowserTab = (url = 'https://www.duckduckgo.com') => {
    const newTab = {
      id: uuidv4(),
      title: 'New Tab',
      type: 'browser',
      url: url,
      closable: true
    };
    setTabs([...tabs, newTab]);
    setActiveTab(newTab.id);
  };

  const closeTab = (tabId) => {
    const newTabs = tabs.filter(t => t.id !== tabId);
    setTabs(newTabs);
    if (activeTab === tabId) {
      setActiveTab(newTabs[newTabs.length - 1].id);
    }
  };

  const updateTabTitle = (tabId, title) => {
    setTabs(tabs.map(t => t.id === tabId ? {...t, title: title.substring(0, 20)} : t));
  };

  if (!isLoggedIn) {
    return <LoginScreen onLoginSuccess={(username, profile) => {
      setIsLoggedIn(true);
      setCurrentUser(username);
    }} />;
  }

  return (
    <div className="flex flex-col h-screen bg-cyber-bg text-cyber-txt font-mono overflow-hidden">
      {/* Tab Bar */}
      <div className="flex items-center bg-cyber-surface border-b border-cyber-border px-2 py-1 gap-1">
        {tabs.map(tab => (
          <div
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-2 px-3 py-1.5 cursor-pointer border-b-2 transition-colors text-xs ${
              activeTab === tab.id
                ? 'border-cyber-accent text-cyber-accent bg-cyber-card'
                : 'border-transparent text-cyber-muted hover:text-cyber-txt hover:bg-cyber-card/50'
            }`}
          >
            {tab.type === 'browser' && <Globe size={12} />}
            <span>{tab.title}</span>
            {tab.closable && (
              <X
                size={12}
                onClick={(e) => {
                  e.stopPropagation();
                  closeTab(tab.id);
                }}
                className="hover:text-red-400"
              />
            )}
          </div>
        ))}
        <button
          onClick={() => addBrowserTab()}
          className="p-1.5 hover:bg-cyber-card rounded text-cyber-muted hover:text-cyber-txt"
          title="New browser tab"
        >
          <Plus size={14} />
        </button>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar */}
        <AnimatePresence>
          {leftSidebarOpen && (
            <motion.div
              initial={{ x: -280 }}
              animate={{ x: 0 }}
              exit={{ x: -280 }}
              className="w-64 bg-cyber-side border-r border-cyber-border flex flex-col"
            >
              {/* Logo */}
              <div className="p-6 border-b border-cyber-border">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-gradient-to-br from-cyber-accent to-cyber-purple rounded-lg flex items-center justify-center">
                    <svg viewBox="0 0 24 24" className="w-6 h-6 text-white">
                      <path fill="currentColor" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/>
                      <circle cx="12" cy="12" r="3" fill="currentColor"/>
                      <path fill="currentColor" d="M12 7c-2.76 0-5 2.24-5 5s2.24 5 5 5 5-2.24 5-5-2.24-5-5-5zm0 8c-1.66 0-3-1.34-3-3s1.34-3 3-3 3 1.34 3 3-1.34 3-3 3z" opacity="0.3"/>
                    </svg>
                  </div>
                  <div>
                    <h1 className="text-lg font-bold">ORCHESTRA</h1>
                    <p className="text-xs text-cyber-muted">v2.9 • Multi-Model AI</p>
                  </div>
                </div>
              </div>

              {/* Actions */}
              <div className="p-4 space-y-2">
                <button
                  onClick={() => {
                    setMessages([]);
                    setCurrentSessionId(null);
                  }}
                  className="w-full bg-cyber-accent hover:bg-cyber-accent/80 text-white px-4 py-2 rounded-lg flex items-center justify-center gap-2 transition-colors"
                >
                  <MessageSquare size={16} />
                  <span className="text-sm font-semibold">New Chat</span>
                </button>

                <button
                  onClick={() => {
                    const autoName = `Session ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
                    saveSession(autoName);
                  }}
                  className="w-full bg-cyber-card hover:bg-cyber-border border border-cyber-border text-white px-4 py-2 rounded-lg flex items-center justify-center gap-2 transition-colors"
                >
                  <Save size={16} className="text-cyber-accent" />
                  <span className="text-sm font-semibold">Save Chat</span>
                </button>
              </div> {/* <--- THIS WAS THE MISSING TAG */}
              
              {/* NEW: Search Bar */}
              <div className="px-4 pb-2">
                <input
                  type="text"
                  placeholder="🔍 Search chats..."
                  value={sessionSearchQuery}
                  onChange={(e) => setSessionSearchQuery(e.target.value)}
                  className="w-full bg-cyber-card border border-cyber-border text-cyber-txt px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-cyber-accent transition-colors placeholder-cyber-muted"
                />
                {sessionSearchQuery && (
                  <p className="text-xs text-cyber-muted mt-1">
                    Searching: "{sessionSearchQuery}"
                  </p>
                )}
              </div>
              
              {/* Recent Sessions */}
              <div className="flex-1 overflow-y-auto p-4">
                <h3 className="text-xs font-semibold text-cyber-muted uppercase tracking-wider mb-3">
                  Recent Sessions
                </h3>
                <div className="space-y-2">
                  {sessions.map(session => (
                    <div 
                      key={session.id} 
                      className="group bg-cyber-card hover:bg-cyber-border p-3 rounded-lg transition-colors"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div 
                          className="flex-1 min-w-0 cursor-pointer"
                          onClick={() => handleSessionClick(session)}
                        >
                          <div className="flex items-center gap-2">
                            {/* Pin indicator */}
                            {session.pinned && (
                              <span className="text-cyber-accent text-xs">📌</span>
                            )}
                            {/* Fork indicator */}
                            {session.parent_session && (
                              <span className="text-cyber-purple text-xs">🔀</span>
                            )}
                            {/* FIX: Changed from session.name to session.title */}
                            <p className="text-sm font-medium truncate">{session.title}</p>
                          </div>
                          <div className="flex items-center gap-2 mt-1">
                            <p className="text-xs text-cyber-muted">
                              {new Date(session.timestamp).toLocaleDateString()}
                            </p>
                            {session.folder && (
                              <span className="text-xs text-cyber-accent bg-cyber-accent/10 px-1.5 py-0.5 rounded">
                                {session.folder}
                              </span>
                            )}
                          </div>
                        </div>
                        
                        {/* Action buttons */}
                        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                          {/* Pin/Unpin button */}
                          <button 
                            onClick={(e) => {
                              e.stopPropagation();
                              togglePin(session.id, session.pinned);
                            }}
                            className="text-cyber-muted hover:text-cyber-accent transition-colors"
                            title={session.pinned ? "Unpin" : "Pin"}
                          >
                            {session.pinned ? '📌' : '📍'}
                          </button>
                          
                          {/* Fork button */}
                          <button 
                            onClick={(e) => {
                              e.stopPropagation();
                              forkSession(session.id);
                            }}
                            className="text-cyber-muted hover:text-cyber-purple transition-colors"
                            title="Fork session"
                          >
                            🔀
                          </button>
                          
                          {/* Delete button */}
                          <button 
                            onClick={(e) => {
                              e.stopPropagation();
                              deleteSession(session.id);
                            }}
                            className="text-cyber-muted hover:text-red-400 transition-colors"
                            title="Delete"
                          >
                            <X size={14} />
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
               </div>
             </div>

              {/* Document Library */}
              <div className="p-4 border-t border-cyber-border space-y-2">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-xs font-semibold text-cyber-muted uppercase tracking-wider">
                    Document Library ({ragStats.total_documents})
                  </h3>
                  <button
                    onClick={toggleRag}
                    className={`px-2 py-0.5 rounded text-xs font-semibold ${
                      settings.ragEnabled
                        ? 'bg-cyber-accent text-white'
                        : 'bg-cyber-card text-cyber-muted'
                    }`}
                  >
                    {settings.ragEnabled ? 'ON' : 'OFF'}
                  </button>
                </div>
                
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept=".pdf,.txt,.md,.docx"
                  onChange={(e) => handleUpload(e.target.files)}
                  className="hidden"
                />
                <input
                  ref={folderInputRef}
                  type="file"
                  webkitdirectory="true"
                  directory="true"
                  multiple
                  onChange={(e) => handleFolderUpload(e.target.files)}
                  className="hidden"
                />
                
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full bg-cyber-card hover:bg-cyber-border text-cyber-txt px-3 py-2 rounded-lg flex items-center justify-center gap-2 text-sm transition-colors"
                >
                  <Upload size={14} />
                  <span>Upload Document</span>
                </button>
                
                <button
                  onClick={() => folderInputRef.current?.click()}
                  className="w-full bg-cyber-card hover:bg-cyber-border text-cyber-txt px-3 py-2 rounded-lg flex items-center justify-center gap-2 text-sm transition-colors"
                >
                  <FolderUp size={14} />
                  <span>Upload Folder</span>
                </button>
                
                <button
                  onClick={clearLibrary}
                  className="w-full bg-cyber-card hover:bg-red-900/20 text-red-400 px-3 py-2 rounded-lg flex items-center justify-center gap-2 text-sm transition-colors"
                >
                  <Trash2 size={14} />
                  <span>Clear Library</span>
                </button>
              </div>

              {/* Settings & User */}
              <div className="p-4 border-t border-cyber-border space-y-2">
                <button
                  onClick={() => setShowSettings(true)}
                  className="w-full bg-cyber-card hover:bg-cyber-border text-cyber-txt px-4 py-2 rounded-lg flex items-center gap-2 transition-colors"
                >
                  <Settings size={16} />
                  <span className="text-sm">Settings</span>
                </button>
                
                <div className="flex items-center justify-between p-3 bg-cyber-card rounded-lg">
                  <div className="flex items-center gap-2">
                    <div className="w-8 h-8 bg-cyber-accent rounded-full flex items-center justify-center text-white font-bold text-sm">
                      {currentUser?.charAt(0).toUpperCase()}
                    </div>
                    <span className="text-sm font-medium">{currentUser}</span>
                  </div>
                  <button
                    onClick={handleLogout}
                    className="text-cyber-muted hover:text-red-400 transition-colors"
                  >
                    <X size={16} />
                  </button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Center Content - All tabs rendered, visibility toggled */}
        <div className="flex-1 relative overflow-hidden">
          {/* Chat Tab */}
          <div className={`absolute inset-0 flex flex-col ${activeTab === 'chat' ? 'z-10' : 'z-0 invisible'}`}>
            {/* Chat Header */}
            <div className="bg-cyber-surface border-b border-cyber-border p-4 flex items-center justify-between">
              <div className="flex items-center gap-4">
                <button
                  onClick={() => setLeftSidebarOpen(!leftSidebarOpen)}
                  className="text-cyber-muted hover:text-cyber-txt transition-colors"
                >
                  <Menu size={20} />
                </button>
                <div>
                  <h2 className="font-semibold">Current Session</h2>
                  <p className="text-xs text-cyber-muted">
                    {messages.length > 0 ? `${messages.length} messages` : 'No messages yet'}
                  </p>
                </div>
              </div>
              <button
                onClick={() => setRightSidebarOpen(!rightSidebarOpen)}
                className="text-cyber-muted hover:text-cyber-txt transition-colors"
              >
                <ChevronRight size={20} className={rightSidebarOpen ? 'rotate-180' : ''} />
              </button>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              {messages.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-center">
                  <div className="w-16 h-16 bg-gradient-to-br from-cyber-accent to-cyber-purple rounded-2xl flex items-center justify-center mb-4">
                    <svg viewBox="0 0 24 24" className="w-10 h-10 text-white">
                      <path fill="currentColor" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/>
                      <circle cx="12" cy="12" r="3" fill="currentColor"/>
                    </svg>
                  </div>
                  <h3 className="text-xl font-semibold text-cyber-txt mb-2">Ready to Orchestrate</h3>
                  <p className="text-cyber-muted max-w-md">
                    Send a message to get started. Orchestra will route your query to the best experts.
                  </p>
                </div>
              ) : (
                messages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-full rounded-lg p-4 ${
                        msg.role === 'user'
                          ? 'bg-cyber-accent text-white'
                          : 'bg-cyber-card border border-cyber-border'
                      }`}
                    >
                      <div className="whitespace-pre-wrap">{msg.content}</div>
                      {msg.experts && msg.experts.length > 0 && (
                        <div className="mt-2 pt-2 border-t border-cyber-border/30 flex flex-wrap gap-1">
                          {msg.experts.map((expert, i) => {
                            const Icon = expertIcons[expert] || Terminal;
                            const colorClass = expertColors[expert] || 'text-cyber-muted';
                            return (
                              <span
                                key={i}
                                className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs ${colorClass}`}
                              >
                                <Icon size={12} />
                                {expert.replace(/_/g, ' ')}
                              </span>
                            );
                          })}
                        </div>
                      )}
                      {msg.image && (
                        <div className="mt-3">
                          <img
                            src={`data:image/png;base64,${msg.image}`}
                            alt="Generated"
                            className="max-w-full rounded border border-cyber-border"
                          />
                        </div>
                      )}
                   {msg.role === 'assistant' && (
                     <button
                       onClick={() => {
                         saveMemory(messages[idx-1]?.content, msg.content, msg.experts);
                         setSavedMemoryId(idx);
                         setTimeout(() => setSavedMemoryId(null), 2000);
                       }}
                       className={`mt-2 text-xs transition-all ${
                         savedMemoryId === idx
                           ? 'text-green-400 font-semibold'
                           : 'text-cyber-muted hover:text-cyber-accent'
                       }`}
                     >
                       {savedMemoryId === idx ? '✓ Saved to Memory' : 'Save to Memory'}
                     </button>
                   )}
                    </div>
                  </div>
                ))
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="border-t border-cyber-border p-4">
              <div className="flex gap-2 items-end">
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      sendMessage();
                    }
                  }}
                  placeholder="Ask anything... Orchestra will route to the best experts (Shift+Enter for new line)"
                  className="flex-1 bg-cyber-surface border border-cyber-border rounded-lg px-4 py-3 text-cyber-txt focus:outline-none focus:border-cyber-accent transition-colors resize-none min-h-[48px] max-h-[200px] overflow-y-auto"
                  rows="1"
                  style={{
                   height: 'auto',
                   minHeight: '48px'
                  }}
                  onInput={(e) => {
                   e.target.style.height = 'auto';
                   e.target.style.height = Math.min(e.target.scrollHeight, 200) + 'px';
                  }}
                />
                <button
                  onClick={sendMessage}
                    className="bg-cyber-accent hover:bg-cyber-accent/80 text-white px-6 py-3 rounded-lg flex items-center gap-2 transition-colors"
                >
                    <Send size={18} />
                    <span className="font-semibold">Send</span>
                </button>
              </div>
            </div>
          </div>

          {/* Browser Tabs - All rendered, visibility toggled */}
          {tabs.filter(t => t.type === 'browser').map(tab => (
            <div 
              key={tab.id}
              className={`absolute inset-0 ${activeTab === tab.id ? 'z-10' : 'z-0 invisible'}`}
            >
              <BrowserView
                tab={tab}
                onTitleUpdate={(title) => updateTabTitle(tab.id, title)}
                onNewTab={(url) => addBrowserTab(url)}
              />
            </div>
          ))}
        </div>

        {/* Right Sidebar */}
        <AnimatePresence>
          {rightSidebarOpen && (
            <motion.div
              initial={{ x: 280 }}
              animate={{ x: 0 }}
              exit={{ x: 280 }}
              className="w-80 bg-cyber-side border-l border-cyber-border overflow-y-auto"
            >
              {/* System Monitor */}
              <HardwareMonitor stats={hardwareStats} />

              {/* Active Experts */}
              <div className="p-6">
                <h3 className="text-sm font-semibold text-cyber-muted uppercase tracking-wider mb-4">
                  Expert Models
                </h3>
                <div className="space-y-3">
                  {expertsList.filter(expert => expert.name !== 'Artisan_Illustrator' && expert.name !== 'Chess_Analyst').map(expert => {
                    const Icon = expertIcons[expert.name] || Terminal;
                    const colorClass = expertColors[expert.name] || 'text-cyber-muted';
                    return (
                      <motion.div
                        key={expert.name}
                        whileHover={{ scale: 1.02 }}
                        className={`p-3 bg-cyber-card rounded-lg border transition-all ${
                          activeExperts.includes(expert.name)
                            ? 'border-cyber-accent shadow-neon'
                            : 'border-cyber-border'
                        }`}
                      >
                        <div className="flex items-start gap-3">
                          <Icon size={18} className={colorClass} />
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium">{expert.displayName}</p>
                            <p className="text-xs text-cyber-muted truncate">{expert.model}</p>
                          </div>
                        </div>
                      </motion.div>
                    );
                  })}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Settings Modal */}
      <AnimatePresence>
        {showSettings && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50"
            onClick={() => setShowSettings(false)}
          >
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              onClick={(e) => e.stopPropagation()}
              className="bg-cyber-surface rounded-lg border border-cyber-border max-w-2xl w-full max-h-[80vh] overflow-y-auto"
            >
              <div className="p-6 border-b border-cyber-border flex items-center justify-between">
                <h2 className="text-xl font-bold">Settings</h2>
                <button
                  onClick={() => setShowSettings(false)}
                  className="text-cyber-muted hover:text-cyber-txt"
                >
                  <X size={20} />
                </button>
              </div>

              <div className="p-6 space-y-6">
                {/* Conductor */}
                <div>
                  <label className="block text-sm font-semibold text-cyber-muted mb-3">
                    Conductor Model
                  </label>
                  <select
                    value={settings.conductor}
                    onChange={(e) => setSettings({...settings, conductor: e.target.value})}
                    className="w-full bg-cyber-card border border-cyber-border rounded px-4 py-2 focus:outline-none focus:border-cyber-accent"
                  >
                    {availableModels.map(model => (
                      <option key={model} value={model}>{model}</option>
                    ))}
                  </select>
                </div>

                {/* Top-K */}
                <div>
                  <label className="block text-sm font-semibold text-cyber-muted mb-3">
                    Top-K Experts: {settings.topK}
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="5"
                    value={settings.topK}
                    onChange={(e) => setSettings({...settings, topK: parseInt(e.target.value)})}
                    className="w-full"
                  />
                </div>

                {/* Toggles */}
                <div className="space-y-3">
                  <label className="flex items-center justify-between p-3 bg-cyber-card rounded-lg cursor-pointer">
                    <span className="text-sm font-medium">RAG Enabled</span>
                    <input
                      type="checkbox"
                      checked={settings.ragEnabled}
                      onChange={(e) => setSettings({...settings, ragEnabled: e.target.checked})}
                      className="w-4 h-4"
                    />
                  </label>
                  <label className="flex items-center justify-between p-3 bg-cyber-card rounded-lg cursor-pointer">
                    <span className="text-sm font-medium">VRAM Optimization</span>
                    <input
                      type="checkbox"
                      checked={settings.vramOptim}
                      onChange={(e) => setSettings({...settings, vramOptim: e.target.checked})}
                      className="w-4 h-4"
                    />
                  </label>
                </div>

                {/* Expert Models */}
                {expertsList.length > 0 && (
                  <div>
                    <label className="block text-sm font-semibold text-cyber-muted mb-3">
                      Expert Model Assignments
                    </label>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {expertsList.filter(expert => expert.name !== 'Artisan_Illustrator' && expert.name !== 'Chess_Analyst').map(expert => (
                        <div key={expert.name} className="bg-cyber-card p-3 rounded-lg">
                          <label className="block text-xs font-medium text-cyber-muted mb-2">
                            {expert.displayName}
                          </label>
                          <select
                            value={settings.experts[expert.name] || expert.model}
                            onChange={(e) => setSettings({
                              ...settings,
                              experts: {...settings.experts, [expert.name]: e.target.value}
                            })}
                            className="w-full bg-cyber-surface border border-cyber-border rounded px-3 py-1.5 text-xs focus:outline-none focus:border-cyber-accent"
                          >
                            {availableModels.map(model => (
                              <option key={model} value={model}>{model}</option>
                            ))}
                          </select>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <div className="p-6 border-t border-cyber-border flex justify-end gap-3">
                <button
                  onClick={() => setShowSettings(false)}
                  className="px-4 py-2 rounded-lg bg-cyber-card hover:bg-cyber-border transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={saveSettings}
                  className="px-4 py-2 rounded-lg bg-cyber-accent hover:bg-cyber-accent/80 text-white transition-colors"
                >
                  Save Settings
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* SECURITY: Code Execution Confirmation Dialog */}
      {pendingCodeExecution && (
        <CodeExecutionDialog
          codeBlock={pendingCodeExecution}
          onExecute={async (result) => {
            console.log('[CODE_EXEC] Execution result:', result);
            
            // If execution was successful and produced a file, open it
            if (result.success && result.file) {
              // Extract filename from path
              const filename = result.file.split('/').pop();
              
              // Try HTTP first (more secure), fallback to file://
              const httpUrl = `http://localhost:5000/api/view_result/${filename}`;
              const fileUrl = 'file://' + result.file;
              
              const newTab = {
                id: 'code-result-' + Date.now(),
                title: 'Code Result',
                type: 'browser',
                url: httpUrl,  // Use HTTP endpoint for better security
                closable: true
              };
              
              // Try HTTP endpoint first
              try {
                const testResponse = await fetch(httpUrl, { method: 'HEAD' });
                if (!testResponse.ok) {
                  // Fallback to file:// if HTTP fails
                  console.log('[CODE_EXEC] HTTP endpoint unavailable, using file:// URL');
                  newTab.url = fileUrl;
                }
              } catch (error) {
                // Fallback to file:// if HTTP fails
                console.log('[CODE_EXEC] Using file:// URL as fallback');
                newTab.url = fileUrl;
              }
              
              setTabs(prev => [...prev, newTab]);
              setActiveTab(newTab.id);
            }
            
            // Close dialog
            setPendingCodeExecution(null);
          }}
          onCancel={() => {
            console.log('[CODE_EXEC] User cancelled code execution');
            setPendingCodeExecution(null);
          }}
        />
      )}
    </div>
  );
}

export default App;
