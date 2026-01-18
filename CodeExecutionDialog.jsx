import React, { useState } from 'react';
import { Code, AlertTriangle, X, Play, Shield } from 'lucide-react';

/**
 * CodeExecutionDialog - Shows code for user review before execution
 * 
 * SECURITY: This is the REQUIRED confirmation step before executing any LLM code.
 * Never bypass this dialog.
 */
export default function CodeExecutionDialog({ codeBlock, onExecute, onCancel }) {
  const [understood, setUnderstood] = useState(false);

  if (!codeBlock) return null;

  const handleExecute = async () => {
    if (!understood) {
      alert('You must acknowledge the security warning before executing code.');
      return;
    }

    // Call backend with explicit confirmation
    try {
      const response = await fetch('http://127.0.0.1:5000/api/execute_code', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          language: codeBlock.language,
          code: codeBlock.code,
          user_confirmed: true  // EXPLICIT confirmation flag
        })
      });

      const result = await response.json();
      onExecute(result);
    } catch (error) {
      console.error('Code execution error:', error);
      alert('Failed to execute code: ' + error.message);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-glass-dark backdrop-blur-glass border border-glass-border rounded-lg shadow-glass-lg max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        
        {/* Header */}
        <div className="p-4 border-b border-glass-border flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Shield size={20} className="text-yellow-400" />
            <h2 className="text-lg font-semibold text-glass-txt">Code Execution Approval Required</h2>
          </div>
          <button
            onClick={onCancel}
            className="p-2 hover:bg-glass-hover rounded-lg transition"
          >
            <X size={20} />
          </button>
        </div>

        {/* Security Warning */}
        <div className="bg-red-900/30 border-l-4 border-red-500 p-4 m-4">
          <div className="flex items-start gap-3">
            <AlertTriangle size={24} className="text-red-400 flex-shrink-0 mt-1" />
            <div>
              <h3 className="font-semibold text-red-400 mb-2">Security Warning</h3>
              <ul className="text-sm text-red-200 space-y-1">
                <li>• This code will execute on your system with your user permissions</li>
                <li>• It can access your files, network, and system resources</li>
                <li>• Only execute code you understand and trust</li>
                <li>• The AI did not intend this code to be auto-executed - it may be an example or demonstration</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Code Display */}
        <div className="flex-1 overflow-auto p-4">
          <div className="bg-gray-900 rounded-lg border border-gray-700">
            <div className="flex items-center justify-between p-3 border-b border-gray-700">
              <div className="flex items-center gap-2">
                <Code size={16} />
                <span className="text-sm font-medium text-gray-300">
                  {codeBlock.language.toUpperCase()} Code
                </span>
              </div>
              <span className="text-xs text-gray-500">
                {codeBlock.code.split('\n').length} lines, {codeBlock.code.length} characters
              </span>
            </div>
            
            {/* Actual code with syntax highlighting */}
            <pre className="p-4 overflow-x-auto">
              <code className={`language-${codeBlock.language}`}>
                {codeBlock.code}
              </code>
            </pre>
          </div>

          {/* What this code does (if analyzable) */}
          <div className="mt-4 p-3 bg-blue-900/20 border border-blue-500/30 rounded-lg">
            <h4 className="text-sm font-semibold text-blue-400 mb-2">Before executing, ask yourself:</h4>
            <ul className="text-sm text-gray-300 space-y-1">
              <li>• Do I understand what this code does?</li>
              <li>• Does it access files or network resources I don't expect?</li>
              <li>• Is this code actually meant to be executed, or was it just an example?</li>
              <li>• Would I run this code if I wrote it myself?</li>
            </ul>
          </div>
        </div>

        {/* Confirmation Checkbox */}
        <div className="p-4 border-t border-glass-border">
          <label className="flex items-start gap-3 cursor-pointer group">
            <input
              type="checkbox"
              checked={understood}
              onChange={(e) => setUnderstood(e.target.checked)}
              className="mt-1 w-4 h-4 rounded border-gray-600 text-blue-500 focus:ring-2 focus:ring-blue-500"
            />
            <span className="text-sm text-glass-txt group-hover:text-white transition">
              I have reviewed this code, understand what it does, and accept the security risks of executing it on my system
            </span>
          </label>
        </div>

        {/* Actions */}
        <div className="p-4 border-t border-glass-border flex items-center justify-end gap-3">
          <button
            onClick={onCancel}
            className="px-4 py-2 glass-btn rounded-lg hover:bg-glass-hover transition"
          >
            Cancel
          </button>
          <button
            onClick={handleExecute}
            disabled={!understood}
            className={`px-4 py-2 rounded-lg transition flex items-center gap-2 ${
              understood
                ? 'bg-blue-600 hover:bg-blue-700 text-white'
                : 'bg-gray-700 text-gray-500 cursor-not-allowed'
            }`}
          >
            <Play size={16} />
            Execute Code
          </button>
        </div>

      </div>
    </div>
  );
}


/**
 * Example Usage in your App.jsx or Chat component:
 * 
 * const [codeToExecute, setCodeToExecute] = useState(null);
 * 
 * // When LLM response contains code:
 * const handleCodeFound = (codeBlock) => {
 *   // Show confirmation dialog
 *   setCodeToExecute(codeBlock);
 * };
 * 
 * // In your render:
 * {codeToExecute && (
 *   <CodeExecutionDialog
 *     codeBlock={codeToExecute}
 *     onExecute={(result) => {
 *       console.log('Code executed:', result);
 *       setCodeToExecute(null);
 *       // Show result to user
 *     }}
 *     onCancel={() => setCodeToExecute(null)}
 *   />
 * )}
 */
