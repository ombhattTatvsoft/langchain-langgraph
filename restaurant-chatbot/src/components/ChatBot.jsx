import React, { useState, useRef, useEffect } from "react";
import parse from 'html-react-parser';
import MicIcon from '@mui/icons-material/Mic';

// Check browser compatibility for SpeechRecognition
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = SpeechRecognition ? new SpeechRecognition() : null;

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isListening, setIsListening] = useState(false);
  const chatRef = useRef();

  // Scroll to the bottom of the chat when new messages are added
  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages]);

  // 🎤 Handle voice input
  const handleVoiceInput = () => {
    if (!recognition) {
      alert("Speech recognition not supported in this browser.");
      return;
    }
    setIsListening(true);
    recognition.lang = "";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    if(isListening)
      recognition.start();

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setInput(transcript);
      setIsListening(false);
    };

    recognition.onerror = (event) => {
      console.error("Speech recognition error:", event.error);
      setIsListening(false);
    };

    recognition.onend = () => setIsListening(false);
  };

  // 🔊 Optional: Speak the bot response
  const speak = (text) => {
    const synth = window.speechSynthesis;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "";
    synth.speak(utterance);
  };

  const sendMessage = async () => {
    if (!input.trim()) return;

    // Add user message to the chat
    const newUserMessage = { role: "user", parts: [{ text: input }] };
    setMessages((prev) => [...prev, newUserMessage]);
    setInput("");
    try {
      // Send message to Flask backend
      const response = await fetch("http://localhost:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input }),
      });

      const data = await response.json();
      console.log(data)
      if (data.error) {
        setMessages((prev) => [
          ...prev,
          { role: "model", parts: [{ text: `Error: Please try again later` }] },
        ]);
      } else {
        // Add bot response to the chat
        setMessages((prev) => [
          ...prev,
          { role: "model", parts: [{ text: data.response }] },
        ]);
      }
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: "model",
          parts: [{ text: `Oops! Something went wrong: ${error.message}` }],
        },
      ]);
    }
  };

  return (
    <div className="w-lg p-6 bg-gray-50 rounded-2xl shadow-lg font-sans">
      <h2 className="text-2xl font-semibold text-center mb-4">
        🍽️ ABC Restaurant Chatbot
      </h2>

      <div
        ref={chatRef}
        className="h-96 overflow-y-auto bg-white border border-gray-200 rounded-xl p-4 mb-4 space-y-4"
      >
        {messages.map((m, i) => (
          <div
          key={i}
          className={`max-w-[80%] ${
            m.role === "user"
              ? "self-end ml-auto text-right"
              : "bg-gray-200 rounded-xl self-start mr-auto text-left"
          }`}
        >
          <div className={`inline-block p-3 rounded-xl ${
            m.role === "user"
              ? "bg-blue-100"
              : ""
          }`}>
            <div className="text-base text-gray-800">{parse(m.parts[0]?.text)}
              <span onClick={() => speak(m.parts[0]?.text)}><MicIcon /></span>
            </div>
          </div>
        </div>
        ))}
      </div>

      <div className="flex items-center gap-2">
        <input
          type="text"
          className="flex-1 p-3 border rounded-lg text-base shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <button
          onClick={handleVoiceInput}
          className={`px-4 py-2 ${isListening ? "bg-red-600" : "bg-blue-600"} text-white rounded-lg hover:bg-blue-700 transition`}
          title="Click to Speak"
        >
          <MicIcon />
        </button>
        <button
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
          onClick={sendMessage}
        >
          Send
        </button>
      </div>
    </div>
  );
};

export default Chatbot;