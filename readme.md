<!-- for backend -->
cd backend
python -m venv venv                # create virtual env
venv\Scripts\activate              # Windows

pip install flask flask-cors langchain langchain-google-genai google-generativeai langchain-community faiss-cpu pandas   # install dependencies
cd langgraph_bot
python langgraph-chatbot2.py                      # run backend

<!-- add env file in backend folder-->
GEMINI_API_KEY=YOUR_GEMINI_API_KEY

<!-- for frontend -->
cd restaurant-chatbot
npm install
npm run dev  

<!-- for tool project -->
cd .\SlotBookingProject\
cd .\SlotBookingProject\
cd .\SlotBookingProject\