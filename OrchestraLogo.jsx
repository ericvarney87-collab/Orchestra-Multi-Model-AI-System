import React from 'react';

export default function OrchestraLogo({ size = 40, className = "" }) {
  return (
    <svg 
      width={size} 
      height={size} 
      viewBox="0 0 128 128" 
      className={className}
      xmlns="http://www.w3.org/2000/svg"
    >
      {/* Brain outline */}
      <path 
        d="M64 20 C40 20, 25 35, 25 55 C25 65, 28 75, 35 82 L35 95 C35 105, 45 115, 64 115 C83 115, 93 105, 93 95 L93 82 C100 75, 103 65, 103 55 C103 35, 88 20, 64 20 Z" 
        fill="none" 
        stroke="currentColor" 
        strokeWidth="3" 
        opacity="0.8"
      />
      
      {/* Neural nodes */}
      <circle cx="50" cy="45" r="4" fill="currentColor"/>
      <circle cx="78" cy="45" r="4" fill="currentColor"/>
      <circle cx="64" cy="55" r="4" fill="currentColor"/>
      <circle cx="50" cy="70" r="4" fill="currentColor"/>
      <circle cx="78" cy="70" r="4" fill="currentColor"/>
      <circle cx="64" cy="85" r="4" fill="currentColor"/>
      <circle cx="40" cy="60" r="3" fill="#a855f7"/>
      <circle cx="88" cy="60" r="3" fill="#a855f7"/>
      
      {/* Connections */}
      <line x1="50" y1="45" x2="64" y2="55" stroke="currentColor" strokeWidth="1.5" opacity="0.6"/>
      <line x1="78" y1="45" x2="64" y2="55" stroke="currentColor" strokeWidth="1.5" opacity="0.6"/>
      <line x1="64" y1="55" x2="50" y2="70" stroke="currentColor" strokeWidth="1.5" opacity="0.6"/>
      <line x1="64" y1="55" x2="78" y2="70" stroke="currentColor" strokeWidth="1.5" opacity="0.6"/>
      <line x1="50" y1="70" x2="64" y2="85" stroke="currentColor" strokeWidth="1.5" opacity="0.6"/>
      <line x1="78" y1="70" x2="64" y2="85" stroke="currentColor" strokeWidth="1.5" opacity="0.6"/>
      <line x1="40" y1="60" x2="50" y2="70" stroke="#a855f7" strokeWidth="1" opacity="0.4"/>
      <line x1="88" y1="60" x2="78" y2="70" stroke="#a855f7" strokeWidth="1" opacity="0.4"/>
      
      {/* Pulsing center */}
      <circle cx="64" cy="55" r="6" fill="currentColor" opacity="0.3">
        <animate attributeName="r" values="6;8;6" dur="2s" repeatCount="indefinite"/>
        <animate attributeName="opacity" values="0.3;0.6;0.3" dur="2s" repeatCount="indefinite"/>
      </circle>
    </svg>
  );
}
