const { app, BrowserWindow, session } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const { initializeBrowserSecurity } = require('./browser-security');

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
      preload: path.join(__dirname, 'preload.js')
    },
    backgroundColor: '#0a0e1a',
    show: false
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
  
  return mainWindow;
}

app.whenReady().then(async () => {
  startPythonBackend();
  console.log('Waiting for backend to start...');
  await waitForBackend();
  
  // Create window
  await createWindow();
  
  // Initialize browser security features
  console.log('🔒 Initializing browser security...');
  initializeBrowserSecurity(mainWindow);
  console.log('✅ Browser security enabled');
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  if (pythonProcess) pythonProcess.kill();
});
