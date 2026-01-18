const { app, BrowserWindow, protocol } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

let mainWindow;
let pythonProcess;

function startPythonBackend() {
  const isPackaged = app.isPackaged;
  const backendPath = isPackaged 
    ? path.join(process.resourcesPath, 'app.asar.unpacked', 'backend')
    : path.join(__dirname, '../backend');
  
  const pythonScript = path.join(backendPath, 'orchestra_api.py');
  console.log('Starting Python backend:', pythonScript);
  
  pythonProcess = spawn('python3', [pythonScript], {
    cwd: backendPath,
    env: { ...process.env }
  });
  
  pythonProcess.stdout.on('data', (data) => {
    console.log(`[Python] ${data.toString().trim()}`);
  });
  
  pythonProcess.stderr.on('data', (data) => {
    console.error(`[Python Error] ${data.toString().trim()}`);
  });
}

async function waitForBackend(maxWait = 15000) {
  const startTime = Date.now();
  const http = require('http');
  
  return new Promise((resolve) => {
    const checkBackend = () => {
      http.get('http://localhost:5000/api/health', (res) => {
        if (res.statusCode === 200) {
          console.log('✅ Backend is ready!');
          resolve(true);
        } else {
          retry();
        }
      }).on('error', retry);
    };
    
    const retry = () => {
      if (Date.now() - startTime < maxWait) {
        setTimeout(checkBackend, 500);
      } else {
        console.log('⚠️ Backend timeout, opening anyway...');
        resolve(false);
      }
    };
    
    checkBackend();
  });
}

async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      webviewTag: true,
      preload: path.join(__dirname, 'preload.js'),
      webSecurity: true,  // Keep browser security ON
      // SECURITY: Allow loading code execution result files from /tmp
      // These are user-approved code execution outputs, not arbitrary files
      allowFileAccessFromFileURLs: true
    },
    backgroundColor: '#0a0e1a',
    show: false
  });

  // SECURITY: Set up Content Security Policy for code execution results
  // Allow file:// only for /tmp directory (where code results are stored)
  mainWindow.webContents.session.webRequest.onHeadersReceived((details, callback) => {
    // Only modify headers for our code execution result files
    if (details.url.startsWith('file:///tmp/')) {
      callback({
        responseHeaders: {
          ...details.responseHeaders,
          'Content-Security-Policy': [
            "default-src 'self' 'unsafe-inline' file: http://localhost:5000; " +
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; " +
            "style-src 'self' 'unsafe-inline';"
          ]
        }
      });
    } else {
      callback({ responseHeaders: details.responseHeaders });
    }
  });

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  if (app.isPackaged) {
    const indexPath = path.join(__dirname, '..', 'dist', 'index.html');
    mainWindow.loadFile(indexPath);
  } else {
    mainWindow.loadURL('http://localhost:5173');
  }
}

app.whenReady().then(async () => {
  // SECURITY: Register file protocol handler for safe file access
  // This allows loading code execution results from /tmp only
  protocol.registerFileProtocol('file', (request, callback) => {
    const url = request.url.substr(7); // Remove 'file://'
    
    // SECURITY: Only allow files from /tmp (code execution results)
    // and the app's own resources
    if (url.startsWith('/tmp/') || url.includes('orchestra-ui-complete')) {
      callback({ path: url });
    } else {
      console.warn(`[SECURITY] Blocked file access: ${url}`);
      callback({ error: -6 }); // FILE_NOT_FOUND
    }
  });

  startPythonBackend();
  console.log('Waiting for backend to start...');
  await waitForBackend();
  await createWindow();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  if (pythonProcess) pythonProcess.kill();
});
