/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        tactical: {
          bg: '#020617', // slate-950
          panel: '#0f172a', // slate-900
          cyan: '#22d3ee', // cyan-400
          amber: '#fbbf24', // amber-400
          rose: '#f43f5e', // rose-500
          emerald: '#34d399', // emerald-400
        },
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
}
