// preload.js - Secure bridge between renderer and main process
const { contextBridge, ipcRenderer } = require('electron');

// Expose safe APIs to renderer
contextBridge.exposeInMainWorld('orchestraAPI', {
  // App info
  getAppPath: () => ipcRenderer.invoke('get-app-path'),
  
  // File operations
  uploadFile: (filePath) => ipcRenderer.invoke('upload-file', filePath),
  
  // Settings
  saveSettings: (settings) => ipcRenderer.invoke('save-settings', settings),
  loadSettings: () => ipcRenderer.invoke('load-settings'),
  
  // Sessions
  saveChatSession: (session) => ipcRenderer.invoke('save-session', session),
  loadChatSessions: () => ipcRenderer.invoke('load-sessions'),
  
  // Platform info
  platform: process.platform
});
