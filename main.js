// main.js - Electron Main Process
const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const express = require('express');

app.disableHardwareAcceleration(); // Disable GPU for Windows compatibility

let mainWindow;
let pythonProcess;
let expressApp;
let expressServer;

// Configuration
const FLASK_PORT = 5000;
const EXPRESS_PORT = 3000;

function startExpressServer() {
  expressApp = express();
  
  // Serve static UI files
  expressApp.use(express.static(path.join(__dirname, 'ui')));
  
  // Proxy API requests to Flask
  expressApp.use('/api', async (req, res) => {
    try {
      const fetch = (await import('node-fetch')).default;
      const response = await fetch('http://localhost:' + FLASK_PORT + req.path, {
        method: req.method,
        headers: req.headers,
        body: req.method !== 'GET' ? JSON.stringify(req.body) : undefined
      });
      const data = await response.json();
      res.json(data);
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });
  
  expressServer = expressApp.listen(EXPRESS_PORT, () => {
    console.log('UI Server running on port ' + EXPRESS_PORT);
  });
}

// Start Python backend
function startPythonBackend() {
  const pythonPath = process.platform === 'win32' ? 'python.exe' : 'python3';
  const scriptPath = path.join(__dirname, 'backend', 'orchestra_api.py');
  
  console.log('🤖 Starting Python backend...');
  
  pythonProcess = spawn(pythonPath, [scriptPath], {
    env: { 
      ...process.env, 
      PYTHONUNBUFFERED: '1',
      PYTHONPATH: `${process.env.HOME}/.local/lib/python3.12/site-packages:${process.env.PYTHONPATH || ''}`
    }
  });
  
  pythonProcess.stdout.on('data', (data) => {
    console.log(`[Python] ${data.toString()}`);
  });
  
  pythonProcess.stderr.on('data', (data) => {
    console.error(`[Python Error] ${data.toString()}`);
  });
  
  pythonProcess.on('close', (code) => {
    console.log(`Python process exited with code ${code}`);
  });
}

// Create main window
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1800,
    height: 1000,
    minWidth: 1200,
    minHeight: 800,
    backgroundColor: '#0a0a0f',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      webviewTag: true,  // ADD THIS LINE
      preload: path.join(__dirname, 'preload.js')
  },
    icon: path.join(__dirname, 'assets', 'icon.png'),
    show: true
  });
  
  mainWindow.loadURL('http://localhost:' + EXPRESS_PORT);
  
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// Wait for servers to be ready
async function waitForServer(port, maxAttempts = 120) {
  const fetch = (await import('node-fetch')).default;
  
  for (let i = 0; i < maxAttempts; i++) {
    try {
      await fetch('http://localhost:' + port + '/api/health');
      return true;
    } catch (error) {
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }
  return false;
}

// App lifecycle
app.whenReady().then(async () => {
  console.log('🎭 Starting Orchestra Desktop App...');
  
  // Start Python backend
  startPythonBackend();
  
  // Wait for Python backend to be ready
  console.log('⏳ Waiting for Python backend...');
  const backendReady = await waitForServer(FLASK_PORT, 120);
  
  if (!backendReady) {
    console.error('❌ Python backend failed to start');
    app.quit();
    return;
  }
  
  console.log('✅ Python backend ready');
  
  // Start Express server for UI
  startExpressServer();
  
  // Create window
  setTimeout(() => {
    createWindow();
  }, 1000);
  
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  // Clean up
  if (pythonProcess) {
    pythonProcess.kill();
  }
  if (expressServer) {
    expressServer.close();
  }
  
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('quit', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
  if (expressServer) {
    expressServer.close();
  }
});

// IPC Handlers
ipcMain.handle('get-app-path', () => {
  return app.getPath('userData');
});

ipcMain.handle('upload-file', async (event, filePath) => {
  // Handle file uploads to Python backend
  const fetch = (await import('node-fetch')).default;
  const FormData = (await import('form-data')).default;
  const fs = require('fs');
  
  const formData = new FormData();
  formData.append('file', fs.createReadStream(filePath));
  
  const response =  await fetch('http://localhost:' + FLASK_PORT + '/api/upload', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
});
