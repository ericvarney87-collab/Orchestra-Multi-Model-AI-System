import React, { useState } from 'react';
import { User, Lock, UserPlus } from 'lucide-react';
import OrchestraLogo from './OrchestraLogo';

export default function Login({ onLoginSuccess }) {
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    try {
      const endpoint = isRegister ? '/api/auth/register' : '/api/auth/login';
      const response = await fetch('http://localhost:5000' + endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password, profile: {} })
      });

      const data = await response.json();

      if (response.ok) {
        onLoginSuccess(data.username, data.profile);
      } else {
        setError(data.error || 'Authentication failed');
      }
    } catch (err) {
      setError('Connection failed. Is the backend running?');
    }
  };

  return (
    <div className="min-h-screen bg-cyber-bg flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-2">
            <OrchestraLogo size={48} className="text-cyber-accent" />
            <h1 className="text-4xl font-bold text-cyber-accent">Orchestra</h1>
        </div>
          <p className="text-cyber-muted">Multi-Model AI Orchestration</p>
        </div>

        <div className="bg-cyber-card border border-cyber-border rounded-lg p-6 shadow-lg">
          <h2 className="text-xl font-semibold text-cyber-text mb-6">
            {isRegister ? 'Create Account' : 'Login'}
          </h2>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm text-cyber-muted mb-2">
                <User className="w-4 h-4 inline mr-2" />
                Username
              </label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full bg-cyber-bg border border-cyber-border rounded px-3 py-2 text-cyber-txt focus:outline-none focus:border-cyber-accent"
                required
              />
            </div>

            <div>
              <label className="block text-sm text-cyber-muted mb-2">
                <Lock className="w-4 h-4 inline mr-2" />
                Password
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-cyber-bg border border-cyber-border rounded px-3 py-2 text-cyber-txt focus:outline-none focus:border-cyber-accent"
                required
              />
            </div>

            {error && (
              <div className="bg-red-500/10 border border-red-500/50 rounded p-3 text-red-400 text-sm">
                {error}
              </div>
            )}

            <button
              type="submit"
              className="w-full bg-cyber-accent hover:bg-cyber-accent/80 text-white font-semibold py-2 rounded transition-colors"
            >
              {isRegister ? 'Create Account' : 'Login'}
            </button>
          </form>

          <button
            onClick={() => {
              setIsRegister(!isRegister);
              setError('');
            }}
            className="w-full mt-4 text-cyber-muted hover:text-cyber-accent text-sm transition-colors"
          >
            {isRegister ? 'Already have an account? Login' : "Don't have an account? Register"}
          </button>
        </div>
      </div>
    </div>
  );
}
