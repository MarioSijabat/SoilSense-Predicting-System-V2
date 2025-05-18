/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",       // Menambahkan path ke file HTML utama
    "./js/script.js",     // Menambahkan path untuk semua file JS di dalam folder js
  ],
  theme: {
    extend: {fontFamily: {
      sans: ['Inter', 'sans-serif'],
      },
    },
  },
  plugins: [],
}

