// Add these handlers to your Electron main.js or main process file
// This handles certificate validation and security events

const { app, BrowserWindow, session, dialog } = require('electron');

// Certificate error handler
app.on('certificate-error', (event, webContents, url, error, certificate, callback) => {
  event.preventDefault();
  
  // Log certificate error
  console.error('Certificate error:', {
    url,
    error,
    issuer: certificate.issuerName,
    subject: certificate.subjectName,
    validStart: certificate.validStart,
    validExpiry: certificate.validExpiry
  });
  
  // Show warning to user
  const response = dialog.showMessageBoxSync({
    type: 'warning',
    buttons: ['Go Back (Recommended)', 'Continue Anyway (Unsafe)'],
    defaultId: 0,
    title: 'Security Warning',
    message: 'This site has an invalid security certificate',
    detail: `The certificate for ${url} is not trusted.\n\nError: ${error}\n\nYou should NOT enter passwords or sensitive information on this site.\n\nIssuer: ${certificate.issuerName}\nSubject: ${certificate.subjectName}`
  });
  
  // Only allow if user explicitly chooses unsafe option
  callback(response === 1);
});

// Session download handler with security checks
function setupSecureDownloads(mainWindow) {
  const downloadSession = session.fromPartition('persist:secure-browser');
  
  downloadSession.on('will-download', (event, item, webContents) => {
    const filename = item.getFilename();
    const extension = filename.split('.').pop().toLowerCase();
    const url = item.getURL();
    
    // Dangerous file extensions
    const dangerousExtensions = [
      'exe', 'bat', 'cmd', 'sh', 'dmg', 'pkg', 'app', 
      'deb', 'rpm', 'msi', 'scr', 'vbs', 'js', 'jar',
      'com', 'pif', 'application', 'gadget', 'msp',
      'lnk', 'inf', 'reg'
    ];
    
    // Check if dangerous
    if (dangerousExtensions.includes(extension)) {
      const response = dialog.showMessageBoxSync(mainWindow, {
        type: 'warning',
        buttons: ['Cancel Download', 'Download Anyway'],
        defaultId: 0,
        title: 'Potentially Dangerous File',
        message: `Warning: .${extension} files can harm your computer`,
        detail: `File: ${filename}\nFrom: ${url}\n\nThis file type can execute code on your computer. Only download if you absolutely trust the source.\n\nRecommendation: Scan with antivirus before opening.`
      });
      
      if (response === 0) {
        item.cancel();
        return;
      }
    }
    
    // Suspicious archives
    const archiveExtensions = ['zip', 'rar', '7z', 'tar', 'gz', 'bz2'];
    if (archiveExtensions.includes(extension)) {
      dialog.showMessageBox(mainWindow, {
        type: 'info',
        title: 'Archive File',
        message: 'Scan compressed files before opening',
        detail: `${filename} is an archive that may contain harmful files. Scan with antivirus before extracting.`
      });
    }
    
    // Set download path
    const savePath = app.getPath('downloads') + '/' + filename;
    item.setSavePath(savePath);
    
    // Track download progress
    item.on('updated', (event, state) => {
      if (state === 'interrupted') {
        console.log('Download interrupted:', filename);
      } else if (state === 'progressing') {
        if (item.isPaused()) {
          console.log('Download paused:', filename);
        } else {
          const progress = Math.round((item.getReceivedBytes() / item.getTotalBytes()) * 100);
          console.log(`Download progress: ${filename} - ${progress}%`);
          
          // Send progress to renderer
          mainWindow.webContents.send('download-progress', {
            filename,
            progress,
            receivedBytes: item.getReceivedBytes(),
            totalBytes: item.getTotalBytes()
          });
        }
      }
    });
    
    item.once('done', (event, state) => {
      if (state === 'completed') {
        console.log('Download completed:', savePath);
        
        // Notify user
        dialog.showMessageBox(mainWindow, {
          type: 'info',
          title: 'Download Complete',
          message: `${filename} downloaded successfully`,
          detail: `Location: ${savePath}\n\nRemember to scan with antivirus before opening.`
        });
        
        mainWindow.webContents.send('download-complete', {
          filename,
          path: savePath,
          state: 'completed'
        });
      } else {
        console.log(`Download failed: ${filename} - ${state}`);
        mainWindow.webContents.send('download-failed', {
          filename,
          state
        });
      }
    });
  });
}

// Content Security Policy
function setupContentSecurityPolicy() {
  const browserSession = session.fromPartition('persist:secure-browser');
  
  browserSession.webRequest.onHeadersReceived((details, callback) => {
    // Add strict CSP if not present
    if (!details.responseHeaders['content-security-policy']) {
      details.responseHeaders['content-security-policy'] = [
        "default-src 'self' https:; " +
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https:; " +
        "style-src 'self' 'unsafe-inline' https:; " +
        "img-src 'self' data: https:; " +
        "connect-src 'self' https: wss:; " +
        "frame-ancestors 'none';"
      ];
    }
    
    // Add security headers
    details.responseHeaders['X-Content-Type-Options'] = ['nosniff'];
    details.responseHeaders['X-Frame-Options'] = ['DENY'];
    details.responseHeaders['X-XSS-Protection'] = ['1; mode=block'];
    details.responseHeaders['Referrer-Policy'] = ['strict-origin-when-cross-origin'];
    
    callback({ responseHeaders: details.responseHeaders });
  });
}

// Mixed content blocker
function setupMixedContentBlocking() {
  const browserSession = session.fromPartition('persist:secure-browser');
  
  browserSession.webRequest.onBeforeRequest((details, callback) => {
    const url = details.url;
    const referrer = details.referrer;
    
    // Block HTTP requests from HTTPS pages
    if (referrer && referrer.startsWith('https://') && url.startsWith('http://')) {
      console.warn('Blocked mixed content:', url, 'from', referrer);
      
      // Notify user
      callback({ cancel: true });
      return;
    }
    
    callback({ cancel: false });
  });
}

// Phishing/malware URL checker (basic implementation)
function setupURLFiltering(mainWindow) {
  const browserSession = session.fromPartition('persist:secure-browser');
  
  // Known malicious patterns
  const maliciousPatterns = [
    /data:text\/html/i,  // Data URLs can be used for phishing
    /javascript:/i,       // JavaScript protocol
    /file:\/\//i,        // File protocol (can leak local files)
  ];
  
  browserSession.webRequest.onBeforeRequest((details, callback) => {
    const url = details.url;
    
    for (const pattern of maliciousPatterns) {
      if (pattern.test(url)) {
        console.error('Blocked malicious URL pattern:', url);
        
        dialog.showMessageBox(mainWindow, {
          type: 'error',
          title: 'Security Block',
          message: 'Potentially malicious URL blocked',
          detail: `This URL pattern is commonly used in attacks:\n${url}`
        });
        
        callback({ cancel: true });
        return;
      }
    }
    
    callback({ cancel: false });
  });
}

// Get certificate info for display
function getCertificateInfo(url) {
  return new Promise((resolve, reject) => {
    const { net } = require('electron');
    const request = net.request(url);
    
    request.on('response', (response) => {
      const certificate = response.socket?.getPeerCertificate();
      
      if (certificate) {
        resolve({
          valid: true,
          issuer: certificate.issuer?.CN || 'Unknown',
          subject: certificate.subject?.CN || 'Unknown',
          validFrom: certificate.valid_from,
          validTo: certificate.valid_to,
          fingerprint: certificate.fingerprint
        });
      } else {
        resolve({ valid: false });
      }
    });
    
    request.on('error', (error) => {
      reject(error);
    });
    
    request.end();
  });
}

// Initialize all security features
function initializeBrowserSecurity(mainWindow) {
  console.log('🔒 Initializing browser security features...');
  
  setupSecureDownloads(mainWindow);
  setupContentSecurityPolicy();
  setupMixedContentBlocking();
  setupURLFiltering(mainWindow);
  
  console.log('✅ Browser security initialized');
}

// Export for use in main process
module.exports = {
  initializeBrowserSecurity,
  getCertificateInfo,
  setupSecureDownloads
};

// Usage in your main Electron file:
/*
const { initializeBrowserSecurity } = require('./browser-security');

app.whenReady().then(() => {
  const mainWindow = createWindow();
  initializeBrowserSecurity(mainWindow);
});
*/
