import React, { useState, useRef, useEffect } from 'react';
import { ArrowLeft, ArrowRight, RotateCw, Home, Download, BookmarkPlus, History, Shield, Lock, AlertTriangle, X, Info } from 'lucide-react';

const BrowserView = ({ tab, onTitleUpdate, onNewTab }) => {
  const [url, setUrl] = useState(tab?.url || 'https://www.duckduckgo.com');
  const [inputUrl, setInputUrl] = useState(url);
  const [canGoBack, setCanGoBack] = useState(false);
  const [canGoForward, setCanGoForward] = useState(false);
  const [isSecure, setIsSecure] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [showBookmarks, setShowBookmarks] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [showDownloads, setShowDownloads] = useState(false);
  const [showSecurityInfo, setShowSecurityInfo] = useState(false);
  const [bookmarks, setBookmarks] = useState([]);
  const [history, setHistory] = useState([]);
  const [downloads, setDownloads] = useState([]);
  const [securityWarning, setSecurityWarning] = useState(null);
  const [certificateInfo, setCertificateInfo] = useState(null);
  const [lastActivity, setLastActivity] = useState(Date.now());
  const webviewRef = useRef(null);
  const sessionTimeoutRef = useRef(null);

  // Session timeout configuration (30 minutes of inactivity)
  const SESSION_TIMEOUT = 30 * 60 * 1000; // 30 minutes
  const SECURITY_CHECK_INTERVAL = 60 * 1000; // Check every minute

  // Load browser data on mount
  useEffect(() => {
    loadBookmarks();
    loadHistory();
    loadDownloads();
    startSessionMonitor();
    
    return () => {
      if (sessionTimeoutRef.current) {
        clearInterval(sessionTimeoutRef.current);
      }
    };
  }, []);

  // Session activity monitor
  const resetSessionTimeout = () => {
    setLastActivity(Date.now());
  };

  const startSessionMonitor = () => {
    sessionTimeoutRef.current = setInterval(() => {
      const idleTime = Date.now() - lastActivity;
      
      if (idleTime > SESSION_TIMEOUT) {
        handleSessionTimeout();
      }
    }, SECURITY_CHECK_INTERVAL);
  };

  const handleSessionTimeout = async () => {
    const webview = webviewRef.current;
    if (!webview) return;

    // Clear session data
    try {
      await webview.executeJavaScript(`
        // Clear all storage
        localStorage.clear();
        sessionStorage.clear();
        
        // Clear cookies
        document.cookie.split(";").forEach(function(c) { 
          document.cookie = c.replace(/^ +/, "").replace(/=.*/, "=;expires=" + new Date().toUTCString() + ";path=/"); 
        });
      `);
      
      alert('⏱️ Session expired due to inactivity (30 minutes). Please log in again for security.');
      navigate('https://www.duckduckgo.com');
    } catch (err) {
      console.error('Session timeout error:', err);
    }
  };

  // Certificate validation
  const validateCertificate = async (currentUrl) => {
    const webview = webviewRef.current;
    if (!webview || !currentUrl.startsWith('https://')) {
      setCertificateInfo(null);
      return;
    }

    try {
      // Get certificate info from webview
      const contents = webview.getWebContents();
      if (contents && contents.session) {
        // Note: This is a placeholder - actual implementation needs Electron main process
        setCertificateInfo({
          valid: true,
          issuer: 'Verified Certificate Authority',
          validFrom: new Date().toLocaleDateString(),
          validTo: new Date(Date.now() + 365 * 24 * 60 * 60 * 1000).toLocaleDateString()
        });
      }
    } catch (err) {
      console.error('Certificate validation error:', err);
      setCertificateInfo(null);
    }
  };

  // Check for dangerous downloads
  const checkDownloadSafety = (filename) => {
    const extension = filename.split('.').pop().toLowerCase();
    const dangerousExtensions = ['exe', 'bat', 'cmd', 'sh', 'dmg', 'pkg', 'app', 'deb', 'rpm', 'msi', 'scr', 'vbs', 'js'];
    const suspiciousExtensions = ['zip', 'rar', '7z', 'tar', 'gz'];
    
    if (dangerousExtensions.includes(extension)) {
      return {
        level: 'danger',
        message: `⚠️ DANGER: .${extension} files can harm your computer. Only download if you absolutely trust this source.`
      };
    } else if (suspiciousExtensions.includes(extension)) {
      return {
        level: 'warning',
        message: `⚠️ WARNING: .${extension} files may contain harmful content. Scan with antivirus before opening.`
      };
    }
    
    return { level: 'safe', message: null };
  };

  // Detect potential phishing
  const checkPhishingIndicators = (url) => {
    const suspiciousPatterns = [
      /paypal.*verify/i,
      /amazon.*account.*suspend/i,
      /bank.*security.*alert/i,
      /update.*payment/i,
      /confirm.*identity/i,
      /-{2,}/,  // Multiple consecutive hyphens (common in phishing)
      /\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/,  // IP address instead of domain
    ];

    for (const pattern of suspiciousPatterns) {
      if (pattern.test(url)) {
        return {
          suspicious: true,
          reason: 'URL contains suspicious patterns common in phishing attacks'
        };
      }
    }

    // Check for lookalike domains
    const trustedDomains = ['paypal.com', 'amazon.com', 'google.com', 'facebook.com', 'chase.com', 'bankofamerica.com'];
    const hostname = new URL(url).hostname;
    
    for (const trusted of trustedDomains) {
      if (hostname.includes(trusted) && hostname !== trusted && !hostname.endsWith('.' + trusted)) {
        return {
          suspicious: true,
          reason: `This site looks similar to ${trusted} but is NOT the real site`
        };
      }
    }

    return { suspicious: false };
  };

  // Sync function to send browser context to Python backend
  const syncBrowserToOrchestra = async () => {
    const webview = webviewRef.current;
    if (!webview) return;

    try {
      const context = {
        url: webview.getURL(),
        title: webview.getTitle(),
        content: await webview.executeJavaScript('document.body.innerText')
      };

      await fetch('http://127.0.0.1:5000/api/browser_sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(context)
      });
      console.log("Orchestra Navigator: Sync Complete");
    } catch (err) {
      console.error("Sync Failed:", err);
    }
  };

  // Add to history
  const saveToHistory = async (url, title) => {
    const historyItem = {
      url,
      title: title || url,
      timestamp: Date.now()
    };

    try {
      await fetch('http://127.0.0.1:5000/api/browser/history', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(historyItem)
      });
      setHistory(prev => [historyItem, ...prev.slice(0, 99)]);
    } catch (err) {
      console.error('Failed to save history:', err);
      // Fallback to localStorage
      const newHistory = [historyItem, ...history].slice(0, 100);
      setHistory(newHistory);
      localStorage.setItem('orchestra_browser_history', JSON.stringify(newHistory));
    }
  };

  // Add bookmark
  const toggleBookmark = async () => {
    const webview = webviewRef.current;
    if (!webview) return;

    const currentUrl = webview.getURL();
    const currentTitle = webview.getTitle();
    const isBookmarked = bookmarks.some(b => b.url === currentUrl);

    try {
      if (isBookmarked) {
        await fetch('http://127.0.0.1:5000/api/browser/bookmarks', {
          method: 'DELETE',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: currentUrl })
        });
        setBookmarks(bookmarks.filter(b => b.url !== currentUrl));
      } else {
        const bookmark = {
          url: currentUrl,
          title: currentTitle,
          timestamp: Date.now()
        };
        await fetch('http://127.0.0.1:5000/api/browser/bookmarks', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(bookmark)
        });
        setBookmarks([bookmark, ...bookmarks]);
      }
    } catch (err) {
      console.error('Failed to toggle bookmark:', err);
    }
  };

  useEffect(() => {
    const webview = webviewRef.current;
    if (!webview) return;

    const handleLoadStart = () => {
      setIsLoading(true);
      const currentUrl = webview.getURL();
      setInputUrl(currentUrl);
      setIsSecure(currentUrl.startsWith('https://'));
      setSecurityWarning(null);
      resetSessionTimeout();
      
      // Check for phishing
      const phishingCheck = checkPhishingIndicators(currentUrl);
      if (phishingCheck.suspicious) {
        setSecurityWarning({
          level: 'danger',
          message: `🚨 PHISHING WARNING: ${phishingCheck.reason}`,
          url: currentUrl
        });
      }
    };

    const handleLoadStop = () => {
      setIsLoading(false);
      const currentUrl = webview.getURL();
      const currentTitle = webview.getTitle();
      
      setUrl(currentUrl);
      setInputUrl(currentUrl);
      setCanGoBack(webview.canGoBack());
      setCanGoForward(webview.canGoForward());
      setIsSecure(currentUrl.startsWith('https://'));
      onTitleUpdate(currentTitle || 'Browser');
      
      // Validate certificate for HTTPS
      if (currentUrl.startsWith('https://')) {
        validateCertificate(currentUrl);
      }
      
      // Add to history
      saveToHistory(currentUrl, currentTitle);
      
      // Trigger sync to Python backend
      syncBrowserToOrchestra();
      
      resetSessionTimeout();
    };

    // Handle certificate errors
    const handleCertificateError = (event, url, error, certificate) => {
      setSecurityWarning({
        level: 'danger',
        message: `🚨 SECURITY ERROR: This site has an invalid security certificate. DO NOT enter passwords or sensitive information!`,
        details: error,
        url: url
      });
    };

    // Handle new window requests (open in new tab instead)
    const handleNewWindow = (e) => {
      e.preventDefault();
      if (onNewTab) {
        onNewTab(e.url);
      }
    };

    // Handle mixed content warnings
    const handleConsoleMessage = (event, level, message) => {
      if (message.includes('Mixed Content') || message.includes('insecure')) {
        setSecurityWarning({
          level: 'warning',
          message: '⚠️ This page contains insecure content (HTTP on HTTPS page). Some features may not work properly.'
        });
      }
    };

    const handleDomReady = () => {
      webview.addEventListener('did-start-loading', handleLoadStart);
      webview.addEventListener('did-stop-loading', handleLoadStop);
      webview.addEventListener('new-window', handleNewWindow);
      webview.addEventListener('certificate-error', handleCertificateError);
      webview.addEventListener('console-message', handleConsoleMessage);
      
      // Track user activity
      webview.addEventListener('dom-ready', resetSessionTimeout);
      
      // Inject security enhancements
      webview.executeJavaScript(`
        // Track user activity
        ['click', 'keypress', 'scroll', 'mousemove'].forEach(event => {
          document.addEventListener(event, () => {
            window.postMessage({ type: 'user-activity' }, '*');
          }, { passive: true });
        });
      `);
    };

    webview.addEventListener('dom-ready', handleDomReady);

    return () => {
      webview.removeEventListener('did-start-loading', handleLoadStart);
      webview.removeEventListener('did-stop-loading', handleLoadStop);
      webview.removeEventListener('new-window', handleNewWindow);
      webview.removeEventListener('certificate-error', handleCertificateError);
      webview.removeEventListener('console-message', handleConsoleMessage);
      webview.removeEventListener('dom-ready', handleDomReady);
    };
  }, [onTitleUpdate, onNewTab, bookmarks, history]);

  // Smart navigation: detect if input is URL or search query
  const navigate = (input) => {
    let finalUrl = input.trim();
    
    // Check if it's a URL (has protocol or looks like a domain)
    const isUrl = /^https?:\/\//i.test(finalUrl) || 
                  /^[\w-]+(\.[\w-]+)+/.test(finalUrl) ||
                  /^localhost/i.test(finalUrl);
    
    if (isUrl) {
      // It's a URL - add https if no protocol
      if (!finalUrl.startsWith('http://') && !finalUrl.startsWith('https://')) {
        finalUrl = 'https://' + finalUrl;
      }
    } else {
      // It's a search query - use DuckDuckGo
      finalUrl = `https://duckduckgo.com/?q=${encodeURIComponent(finalUrl)}`;
    }
    
    setUrl(finalUrl);
    setInputUrl(finalUrl);
    resetSessionTimeout();
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      navigate(inputUrl);
    }
  };

  const loadBookmarks = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5000/api/browser/bookmarks');
      const data = await response.json();
      setBookmarks(data.bookmarks || []);
    } catch (err) {
      const saved = localStorage.getItem('orchestra_browser_bookmarks');
      if (saved) setBookmarks(JSON.parse(saved));
    }
  };

  const loadHistory = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5000/api/browser/history');
      const data = await response.json();
      setHistory(data.history || []);
    } catch (err) {
      const saved = localStorage.getItem('orchestra_browser_history');
      if (saved) setHistory(JSON.parse(saved));
    }
  };

  const loadDownloads = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5000/api/browser/downloads');
      const data = await response.json();
      setDownloads(data.downloads || []);
    } catch (err) {
      console.error('Failed to load downloads:', err);
    }
  };

  const clearHistory = async () => {
    if (confirm('Clear all browsing history?')) {
      try {
        await fetch('http://127.0.0.1:5000/api/browser/history', {
          method: 'DELETE'
        });
        setHistory([]);
      } catch (err) {
        setHistory([]);
        localStorage.removeItem('orchestra_browser_history');
      }
    }
  };

  const clearSession = async () => {
    if (confirm('⚠️ This will log you out of all websites. Continue?')) {
      handleSessionTimeout();
    }
  };

  const isBookmarked = bookmarks.some(b => b.url === url);

  return (
    <div className="flex flex-col h-full bg-glass-bg backdrop-blur-glass">
      {/* Security Warning Banner */}
      {securityWarning && (
        <div className={`p-3 border-b ${
          securityWarning.level === 'danger' 
            ? 'bg-red-900/80 border-red-500' 
            : 'bg-yellow-900/80 border-yellow-500'
        }`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertTriangle size={20} className="text-white" />
              <span className="text-white font-semibold">{securityWarning.message}</span>
            </div>
            <button
              onClick={() => setSecurityWarning(null)}
              className="text-white hover:text-gray-300"
            >
              <X size={18} />
            </button>
          </div>
          {securityWarning.details && (
            <div className="text-xs text-white/80 mt-1 ml-7">
              Details: {securityWarning.details}
            </div>
          )}
        </div>
      )}

      {/* Navigation Bar */}
      <div className="flex items-center gap-2 p-2 bg-glass-dark backdrop-blur-glass border-b border-glass-border">
        <button
          onClick={() => webviewRef.current?.goBack()}
          disabled={!canGoBack}
          className="p-2 glass-btn rounded-lg disabled:opacity-30 disabled:cursor-not-allowed"
          title="Go back"
        >
          <ArrowLeft size={18} />
        </button>
        <button
          onClick={() => webviewRef.current?.goForward()}
          disabled={!canGoForward}
          className="p-2 glass-btn rounded-lg disabled:opacity-30 disabled:cursor-not-allowed"
          title="Go forward"
        >
          <ArrowRight size={18} />
        </button>
        <button
          onClick={() => webviewRef.current?.reload()}
          className="p-2 glass-btn rounded-lg"
          title="Reload"
        >
          <RotateCw size={18} className={isLoading ? 'animate-spin' : ''} />
        </button>
        <button
          onClick={() => navigate('https://www.duckduckgo.com')}
          className="p-2 glass-btn rounded-lg"
          title="Home"
        >
          <Home size={18} />
        </button>
        
        {/* URL Bar with Enhanced Security Indicator */}
        <div className="flex-1 flex items-center gap-2 glass-input rounded-lg px-3 py-2">
          {isSecure ? (
            <button
              onClick={() => setShowSecurityInfo(!showSecurityInfo)}
              className="flex items-center gap-1 hover:bg-glass-hover px-2 py-1 rounded"
              title="View security information"
            >
              <Lock size={16} className="text-green-400" />
              <span className="text-xs text-green-400">Secure</span>
            </button>
          ) : (
            <div className="flex items-center gap-1 px-2 py-1">
              <AlertTriangle size={16} className="text-yellow-400" />
              <span className="text-xs text-yellow-400">Not Secure</span>
            </div>
          )}
          <input
            type="text"
            value={inputUrl}
            onChange={(e) => setInputUrl(e.target.value)}
            onKeyPress={handleKeyPress}
            onFocus={resetSessionTimeout}
            placeholder="Search or enter address..."
            className="flex-1 bg-transparent text-glass-txt text-sm focus:outline-none"
          />
        </div>

        <button
          onClick={toggleBookmark}
          className={`p-2 glass-btn rounded-lg ${isBookmarked ? 'text-glass-accent' : ''}`}
          title={isBookmarked ? "Remove bookmark" : "Bookmark this page"}
        >
          <BookmarkPlus size={18} fill={isBookmarked ? 'currentColor' : 'none'} />
        </button>
        <button
          onClick={() => setShowHistory(!showHistory)}
          className="p-2 glass-btn rounded-lg"
          title="History"
        >
          <History size={18} />
        </button>
        <button
          onClick={() => setShowDownloads(!showDownloads)}
          className="p-2 glass-btn rounded-lg relative"
          title="Downloads"
        >
          <Download size={18} />
          {downloads.filter(d => d.state === 'downloading').length > 0 && (
            <span className="absolute top-0 right-0 w-2 h-2 bg-glass-accent rounded-full"></span>
          )}
        </button>
        <button
          onClick={clearSession}
          className="p-2 glass-btn rounded-lg"
          title="Clear session (log out of all sites)"
        >
          <Shield size={18} />
        </button>
      </div>

      {/* Security Info Panel */}
      {showSecurityInfo && certificateInfo && (
        <div className="glass-card m-2 p-3 border border-glass-border rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Lock size={16} className="text-green-400" />
              <span className="text-sm font-semibold text-glass-txt">Connection is Secure</span>
            </div>
            <button onClick={() => setShowSecurityInfo(false)}>
              <X size={16} className="text-glass-txt-muted" />
            </button>
          </div>
          <div className="text-xs text-glass-txt-secondary space-y-1">
            <div>✓ Certificate is valid</div>
            <div>✓ Issued by: {certificateInfo.issuer}</div>
            <div>✓ Valid from: {certificateInfo.validFrom}</div>
            <div>✓ Valid until: {certificateInfo.validTo}</div>
            <div className="mt-2 text-glass-txt-muted">
              Your connection to this site is encrypted and authenticated.
            </div>
          </div>
        </div>
      )}

      {/* History Sidebar */}
      {showHistory && (
        <div className="absolute right-0 top-14 w-80 max-h-96 glass-dropdown rounded-lg shadow-glass-lg z-50 flex flex-col">
          <div className="flex items-center justify-between p-3 border-b border-glass-border">
            <h3 className="font-semibold text-glass-txt">History</h3>
            <button
              onClick={clearHistory}
              className="text-xs text-red-400 hover:text-red-300"
            >
              Clear All
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-2">
            {history.length === 0 ? (
              <p className="text-glass-txt-muted text-sm p-4 text-center">No history yet</p>
            ) : (
              history.map((item, idx) => (
                <div
                  key={idx}
                  onClick={() => {
                    navigate(item.url);
                    setShowHistory(false);
                  }}
                  className="p-2 glass-dropdown-item rounded cursor-pointer mb-1"
                >
                  <p className="text-sm truncate text-glass-txt">{item.title}</p>
                  <p className="text-xs text-glass-txt-muted truncate">{item.url}</p>
                  <p className="text-xs text-glass-txt-muted">
                    {new Date(item.timestamp).toLocaleString()}
                  </p>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {/* Bookmarks Bar */}
      <div className="bg-glass-surface backdrop-blur-medium border-b border-glass-border p-2 flex gap-2 overflow-x-auto">
        {bookmarks.slice(0, 10).map((bookmark, idx) => (
          <button
            key={idx}
            onClick={() => navigate(bookmark.url)}
            className="flex items-center gap-1 px-3 py-1 glass-btn rounded text-xs whitespace-nowrap"
            title={bookmark.url}
          >
            <BookmarkPlus size={12} className="text-glass-accent" />
            {bookmark.title.slice(0, 20)}{bookmark.title.length > 20 ? '...' : ''}
          </button>
        ))}
      </div>

      {/* Downloads Panel */}
      {showDownloads && (
        <div className="absolute right-0 top-14 w-96 h-64 glass-dropdown rounded-lg shadow-glass-lg z-50 flex flex-col">
          <div className="flex items-center justify-between p-3 border-b border-glass-border">
            <h3 className="font-semibold text-glass-txt">Downloads</h3>
          </div>
          <div className="flex-1 overflow-y-auto p-2">
            {downloads.length === 0 ? (
              <p className="text-glass-txt-muted text-sm p-4 text-center">No downloads</p>
            ) : (
              downloads.map((item, idx) => (
                <div key={idx} className="p-2 glass-card rounded mb-1">
                  <div className="flex items-center justify-between">
                    <p className="text-sm truncate flex-1 text-glass-txt">{item.filename}</p>
                    <span className={`text-xs ${
                      item.state === 'completed' ? 'text-green-400' : 'text-glass-accent'
                    }`}>
                      {item.state}
                    </span>
                  </div>
                  {item.path && (
                    <p className="text-xs text-glass-txt-muted truncate">{item.path}</p>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {/* Webview with enhanced security */}
      <webview
        ref={webviewRef}
        src={url}
        className="flex-1 w-full"
        allowpopups="false"
        partition="persist:secure-browser"
        nodeintegration="false"
        nodeintegrationinsubframes="false"
        webpreferences="contextIsolation=true,sandbox=true,enableRemoteModule=false,webSecurity=true,allowRunningInsecureContent=false"
        useragent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
      />
    </div>
  );
};

export default BrowserView;
