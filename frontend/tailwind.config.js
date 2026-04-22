/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: '#3b82f6', // blue-500
        secondary: '#1f2937', // gray-800
        accent: '#ef4444', // red-500
      }
    },
  },
  plugins: [],
}
