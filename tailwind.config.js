/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'cyber-bg': '#0a0e1a',
        'cyber-side': '#0f1419',
        'cyber-card': '#151b26',
        'cyber-surface': '#1a2332',
        'cyber-border': '#2a3441',
        'cyber-txt': '#e8eef6',
        'cyber-muted': '#8b95a8',
        'cyber-accent': '#00d9ff',
        'cyber-purple': '#b794f6',
      },
    },
  },
  plugins: [],
}
