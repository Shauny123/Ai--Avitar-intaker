// src/hooks/useVoiceAvatar.ts
import { useState, useCallback, useRef, useEffect } from 'react';

interface VoiceAvatarState {
  isListening: boolean;
  isSpeaking: boolean;
  isProcessing: boolean;
  audioLevel: number;
  currentTranscript: string;
  error: string | null;
}

interface VoiceAvatarHook {
  state: VoiceAvatarState;
  startListening: () => void;
  stopListening: () => void;
  speak: (text: string, language: string) => Promise<void>;
  toggleListening: () => void;
}

export const useVoiceAvatar = (
  language: string = 'en',
  onTranscript: (transcript: string) => void,
  onResponse: (response: string) => void
): VoiceAvatarHook => {
  const [state, setState] = useState<VoiceAvatarState>({
    isListening: false,
    isSpeaking: false,
    isProcessing: false,
    audioLevel: 0,
    currentTranscript: '',
    error: null,
  });

  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const synthRef = useRef<SpeechSynthesis | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const microphoneRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Initialize speech recognition
  useEffect(() => {
    if (typeof window !== 'undefined' && 'webkitSpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      
      const recognition = recognitionRef.current;
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = getVoiceLanguageCode(language);

      recognition.onstart = () => {
        setState(prev => ({ ...prev, isListening: true, error: null }));
      };

      recognition.onresult = (event) => {
        let finalTranscript = '';
        let interimTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript;
          } else {
            interimTranscript += transcript;
          }
        }

        setState(prev => ({ 
          ...prev, 
          currentTranscript: finalTranscript || interimTranscript 
        }));

        if (finalTranscript) {
          onTranscript(finalTranscript);
          processUserInput(finalTranscript);
        }
      };

      recognition.onerror = (event) => {
        setState(prev => ({ 
          ...prev, 
          error: `Speech recognition error: ${event.error}`,
          isListening: false 
        }));
      };

      recognition.onend = () => {
        setState(prev => ({ ...prev, isListening: false }));
      };
    }

    // Initialize speech synthesis
    if (typeof window !== 'undefined') {
      synthRef.current = window.speechSynthesis;
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [language]);

  // Audio level monitoring
  const startAudioMonitoring = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContextRef.current = new AudioContext();
      analyserRef.current = audioContextRef.current.createAnalyser();
      microphoneRef.current = audioContextRef.current.createMediaStreamSource(stream);
      
      microphoneRef.current.connect(analyserRef.current);
      analyserRef.current.fftSize = 256;

      const bufferLength = analyserRef.current.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      const updateAudioLevel = () => {
        if (analyserRef.current) {
          analyserRef.current.getByteFrequencyData(dataArray);
          const average = dataArray.reduce((a, b) => a + b) / bufferLength;
          setState(prev => ({ ...prev, audioLevel: average / 255 }));
          animationFrameRef.current = requestAnimationFrame(updateAudioLevel);
        }
      };

      updateAudioLevel();
    } catch (error) {
      console.error('Error accessing microphone:', error);
      setState(prev => ({ ...prev, error: 'Microphone access denied' }));
    }
  }, []);

  const stopAudioMonitoring = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
  }, []);

  // Process user input through AI
  const processUserInput = useCallback(async (transcript: string) => {
    setState(prev => ({ ...prev, isProcessing: true }));

    try {
      const response = await fetch('/api/ai-conversation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: transcript,
          language: language,
          context: 'legal-intake'
        })
      });

      const data = await response.json();
      
      if (data.response) {
        onResponse(data.response);
        await speak(data.response, language);
      }
    } catch (error) {
      console.error('AI processing error:', error);
      setState(prev => ({ 
        ...prev, 
        error: 'Failed to process your request. Please try again.' 
      }));
    } finally {
      setState(prev => ({ ...prev, isProcessing: false }));
    }
  }, [language, onResponse]);

  // Text-to-speech functionality
  const speak = useCallback(async (text: string, lang: string): Promise<void> => {
    return new Promise((resolve, reject) => {
      if (!synthRef.current) {
        reject(new Error('Speech synthesis not supported'));
        return;
      }

      setState(prev => ({ ...prev, isSpeaking: true }));

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = getVoiceLanguageCode(lang);
      utterance.rate = 0.9;
      utterance.pitch = 1.0;
      utterance.volume = 0.8;

      // Select appropriate voice for language
      const voices = synthRef.current.getVoices();
      const preferredVoice = voices.find(voice => 
        voice.lang.startsWith(getVoiceLanguageCode(lang)) && 
        (voice.name.includes('Female') || voice.name.includes('Google'))
      );
      
      if (preferredVoice) {
        utterance.voice = preferredVoice;
      }

      utterance.onend = () => {
        setState(prev => ({ ...prev, isSpeaking: false }));
        resolve();
      };

      utterance.onerror = (event) => {
        setState(prev => ({ ...prev, isSpeaking: false }));
        reject(new Error(`Speech synthesis error: ${event.error}`));
      };

      synthRef.current.speak(utterance);
    });
  }, []);

  const startListening = useCallback(() => {
    if (recognitionRef.current && !state.isListening) {
      recognitionRef.current.start();
      startAudioMonitoring();
    }
  }, [state.isListening, startAudioMonitoring]);

  const stopListening = useCallback(() => {
    if (recognitionRef.current && state.isListening) {
      recognitionRef.current.stop();
      stopAudioMonitoring();
    }
  }, [state.isListening, stopAudioMonitoring]);

  const toggleListening = useCallback(() => {
    if (state.isListening) {
      stopListening();
    } else {
      startListening();
    }
  }, [state.isListening, startListening, stopListening]);

  return {
    state,
    startListening,
    stopListening,
    speak,
    toggleListening,
  };
};

// Helper function to convert our language codes to speech API codes
const getVoiceLanguageCode = (langCode: string): string => {
  const languageMap: Record<string, string> = {
    'en': 'en-US',
    'en-gb': 'en-GB',
    'es': 'es-ES',
    'es-mx': 'es-MX',
    'fr': 'fr-FR',
    'de': 'de-DE',
    'it': 'it-IT',
    'pt-br': 'pt-BR',
    'zh': 'zh-CN',
    'zh-tw': 'zh-TW',
    'ja': 'ja-JP',
    'ko': 'ko-KR',
    'hi': 'hi-IN',
    'bn': 'bn-IN',
    'ur': 'ur-PK',
    'ar': 'ar-SA',
    'he': 'he-IL',
    'ru': 'ru-RU',
    'tr': 'tr-TR',
    'th': 'th-TH',
    'vi': 'vi-VN',
    'id': 'id-ID',
    'tl': 'tl-PH',
    'sw': 'sw-KE',
    'ha': 'ha-NG',
    'yo': 'yo-NG',
    'zu': 'zu-ZA',
    'am': 'am-ET',
    'ne': 'ne-NP',
    'mn': 'mn-MN',
    'el': 'el-GR',
  };

  return languageMap[langCode] || 'en-US';
};

// Voice Avatar Component
// src/components/VoiceAvatar.tsx
import React, { useState, useEffect } from 'react';
import { Mic, MicOff, Volume2, VolumeX, MessageCircle } from 'lucide-react';
import { useVoiceAvatar } from '../hooks/useVoiceAvatar';

interface VoiceAvatarProps {
  language: string;
  intakeData: any;
  onTranscript: (transcript: string) => void;
  onAIResponse: (response: string) => void;
}

export const VoiceAvatar: React.FC<VoiceAvatarProps> = ({
  language,
  intakeData,
  onTranscript,
  onAIResponse,
}) => {
  const [messages, setMessages] = useState<Array<{
    type: 'user' | 'ai';
    content: string;
    timestamp: Date;
  }>>([]);

  const { state, startListening, stopListening, toggleListening, speak } = useVoiceAvatar(
    language,
    (transcript) => {
      setMessages(prev => [...prev, {
        type: 'user',
        content: transcript,
        timestamp: new Date()
      }]);
      onTranscript(transcript);
    },
    (response) => {
      setMessages(prev => [...prev, {
        type: 'ai',
        content: response,
        timestamp: new Date()
      }]);
      onAIResponse(response);
    }
  );

  // Initial greeting
  useEffect(() => {
    const initialGreeting = getInitialGreeting(intakeData, language);
    speak(initialGreeting, language);
    setMessages([{
      type: 'ai',
      content: initialGreeting,
      timestamp: new Date()
    }]);
  }, []);

  // Audio level visualization
  const getAudioLevelHeight = () => {
    return Math.max(20, state.audioLevel * 100);
  };

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-indigo-500 to-purple-600">
      {/* Avatar Section */}
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="text-center">
          {/* Animated Avatar */}
          <div className="relative w-48 h-48 mx-auto mb-6">
            {/* Base avatar circle */}
            <div className={`w-full h-full bg-white rounded-full flex items-center justify-center shadow-2xl transition-all duration-300 ${
              state.isSpeaking ? 'animate-pulse' : ''
            } ${
              state.isListening ? 'ring-4 ring-white ring-opacity-50' : ''
            }`}>
              
              {/* Avatar face */}
              <div className="w-36 h-36 bg-gradient-to-br from-pink-400 to-indigo-400 rounded-full flex items-center justify-center relative overflow-hidden">
                {/* Animated mouth for speaking */}
                {state.isSpeaking && (
                  <div className="absolute bottom-12 left-1/2 transform -translate-x-1/2">
                    <div className="w-8 h-4 bg-white rounded-full animate-bounce"></div>
                  </div>
                )}
                
                {/* Eyes */}
                <div className="absolute top-10 left-1/2 transform -translate-x-1/2 flex space-x-4">
                  <div className="w-3 h-3 bg-white rounded-full"></div>
                  <div className="w-3 h-3 bg-white rounded-full"></div>
                </div>
                
                {/* Compassionate expression */}
                <div className="absolute bottom-16 left-1/2 transform -translate-x-1/2 w-6 h-3 border-2 border-white border-t-0 rounded-b-full"></div>
              </div>
            </div>

            {/* Audio level indicators */}
            {state.isListening && (
              <div className="absolute -bottom-4 left-1/2 transform -translate-x-1/2 flex space-x-1">
                {[...Array(5)].map((_, i) => (
                  <div
                    key={i}
                    className="w-2 bg-white rounded-full transition-all duration-100"
                    style={{
                      height: `${Math.max(8, (state.audioLevel * 50) + (i * 4))}px`,
                      opacity: state.audioLevel > (i * 0.2) ? 1 : 0.3
                    }}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Status Text */}
          <h1 className="text-white text-2xl font-bold mb-2">
            Welcome, {intakeData.firstName}
          </h1>
          
          <p className="text-indigo-100 text-sm mb-6">
            {state.isListening ? 'Listening...' : 
             state.isSpeaking ? 'Speaking...' :
             state.isProcessing ? 'Processing...' :
             'Tap to speak'}
          </p>

          {/* Voice Controls */}
          <div className="flex justify-center space-x-4">
            <button
              onClick={toggleListening}
              disabled={state.isSpeaking || state.isProcessing}
              className={`p-4 rounded-full transition-all duration-200 ${
                state.isListening 
                  ? 'bg-red-500 hover:bg-red-600 animate-pulse' 
                  : 'bg-white bg-opacity-20 hover:bg-opacity-30'
              } ${
                (state.isSpeaking || state.isProcessing) 
                  ? 'opacity-50 cursor-not-allowed' 
                  : 'cursor-pointer'
              }`}
            >
              {state.isListening ? (
                <MicOff className="w-6 h-6 text-white" />
              ) : (
                <Mic className="w-6 h-6 text-white" />
              )}
            </button>

            <button
              onClick={() => speak("I'm here to help you with your legal questions.", language)}
              disabled={state.isSpeaking || state.isListening}
              className="p-4 rounded-full bg-white bg-opacity-20 hover:bg-opacity-30 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {state.isSpeaking ? (
                <VolumeX className="w-6 h-6 text-white" />
              ) : (
                <Volume2 className="w-6 h-6 text-white" />
              )}
            </button>
          </div>

          {/* Current transcript */}
          {state.currentTranscript && (
            <div className="mt-4 p-3 bg-white bg-opacity-20 rounded-lg">
              <p className="text-white text-sm italic">
                "{state.currentTranscript}"
              </p>
            </div>
          )}

          {/* Error display */}
          {state.error && (
            <div className="mt-4 p-3 bg-red-500 bg-opacity-20 rounded-lg">
              <p className="text-white text-sm">
                {state.error}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Conversation History (Mobile) */}
      <div className="md:hidden max-h-32 overflow-y-auto bg-black bg-opacity-20 p-4">
        <div className="space-y-2">
          {messages.slice(-3).map((msg, idx) => (
            <div key={idx} className={`text-xs ${
              msg.type === 'user' ? 'text-blue-200' : 'text-white'
            }`}>
              <strong>{msg.type === 'user' ? 'You' : 'Assistant'}:</strong> {msg.content}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// Helper function for initial greeting in different languages
const getInitialGreeting = (intakeData: any, language: string): string => {
  const greetings: Record<string, (name: string, state: string) => string> = {
    'en': (name, state) => `Hello ${name}, I see you're in ${state}. I'm here to help you with your legal case. Please tell me what happened.`,
    'es': (name, state) => `Hola ${name}, veo que está en ${state}. Estoy aquí para ayudarle con su caso legal. Por favor cuénteme qué pasó.`,
    'fr': (name, state) => `Bonjour ${name}, je vois que vous êtes en ${state}. Je suis là pour vous aider avec votre affaire juridique. Veuillez me dire ce qui s'est passé.`,
    'zh': (name, state) => `您好${name}，我看到您在${state}。我在这里帮助您处理法律案件。请告诉我发生了什么。`,
    'ar': (name, state) => `مرحباً ${name}، أرى أنك في ${state}. أنا هنا لمساعدتك في قضيتك القانونية. يرجى إخباري بما حدث.`,
    'hi': (name, state) => `नमस्ते ${name}, मैं देख रहा हूं कि आप ${state} में हैं। मैं आपके कानूनी मामले में आपकी मदद के लिए यहां हूं। कृपया मुझे बताएं कि क्या हुआ।`,
  };

  const greeting = greetings[language] || greetings['en'];
  return greeting(intakeData.firstName, intakeData.state);
};

export default VoiceAvatar;