<!-- for backend -->
cd backend
python -m venv venv                # create virtual env
venv\Scripts\activate              # Windows

pip install flask flask-cors langchain langchain-google-genai google-generativeai langchain-community faiss-cpu pandas   # install dependencies

python app.py                      # run backend

<!-- for frontend -->
cd restaurant-chatbot
npm install
npm run dev  

<!-- add env file -->
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
