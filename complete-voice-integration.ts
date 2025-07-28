// src/pages/api/voice-process.ts
import { NextApiRequest, NextApiResponse } from 'next';
import formidable from 'formidable';
import AudioProcessingPipeline from '../../services/audioProcessingPipeline';

export const config = {
  api: {
    bodyParser: false,
  },
};

interface VoiceProcessResponse {
  transcription: {
    text: string;
    confidence: number;
    language: string;
    processingMethod: string;
    cost: number;
  };
  aiResponse: {
    text: string;
    audio?: string; // Base64 encoded audio
    confidence: number;
    sources: Array<{
      content: string;
      source: string;
      relevance: number;
    }>;
    legalContext: string[];
    nextQuestions: string[];
  };
  metadata: {
    processingTime: number;
    qualityMetrics: any;
    totalCost: number;
  };
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<VoiceProcessResponse | { error: string }>
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const startTime = Date.now();

  try {
    // Parse the multipart form data
    const form = formidable({
      maxFileSize: 10 * 1024 * 1024, // 10MB limit
      keepExtensions: true,
    });

    const [fields, files] = await form.parse(req);
    
    const audioFile = Array.isArray(files.audio) ? files.audio[0] : files.audio;
    const language = Array.isArray(fields.language) ? fields.language[0] : fields.language || 'en';
    const userContext = JSON.parse(
      Array.isArray(fields.userContext) ? fields.userContext[0] : fields.userContext || '{}'
    );

    if (!audioFile) {
      return res.status(400).json({ error: 'No audio file provided' });
    }

    // Initialize audio processing pipeline
    const pipeline = new AudioProcessingPipeline({
      whisperEndpoint: process.env.WHISPER_ENDPOINT || 'https://api.openai.com',
      flamingoEndpoint: process.env.NVIDIA_FLAMINGO_ENDPOINT || 'https://api.nvidia.com',
      qualityThreshold: 6.0,
      costOptimization: true,
      fallbackEnabled: true,
    });

    // Convert file to blob
    const fs = require('fs');
    const audioBuffer = fs.readFileSync(audioFile.filepath);
    const audioBlob = new Blob([audioBuffer], { type: audioFile.mimetype || 'audio/webm' });

    // Process audio through the complete pipeline
    const { transcription, rags } = await pipeline.processAudio(
      audioBlob,
      language,
      userContext
    );

    // Generate voice response
    const audioResponse = await generateVoiceResponse(
      rags.response,
      language,
      userContext
    );

    // Calculate total processing time and cost
    const processingTime = Date.now() - startTime;
    const totalCost = transcription.cost + (audioResponse.cost || 0);

    // Build response
    const response: VoiceProcessResponse = {
      transcription: {
        text: transcription.text,
        confidence: transcription.confidence,
        language: transcription.language,
        processingMethod: transcription.processingMethod,
        cost: transcription.cost,
      },
      aiResponse: {
        text: rags.response,
        audio: audioResponse.audioBase64,
        confidence: rags.confidence,
        sources: rags.sources,
        legalContext: rags.legalContext,
        nextQuestions: await generateFollowUpQuestions(
          transcription.text,
          rags.legalContext,
          language
        ),
      },
      metadata: {
        processingTime,
        qualityMetrics: transcription.qualityMetrics,
        totalCost,
      },
    };

    res.status(200).json(response);

  } catch (error) {
    console.error('Voice processing error:', error);
    res.status(500).json({
      error: 'Voice processing failed. Please try again.',
    });
  }
}

/**
 * Generate voice response using text-to-speech
 */
async function generateVoiceResponse(
  text: string,
  language: string,
  userContext: any
): Promise<{ audioBase64: string; cost: number }> {
  
  // Use OpenAI TTS if available
  if (process.env.OPENAI_API_KEY) {
    return await generateOpenAIVoice(text, language);
  }
  
  // Fallback to browser-based TTS (no cost)
  return {
    audioBase64: '', // Empty - will use browser TTS
    cost: 0
  };
}

/**
 * Generate OpenAI TTS voice
 */
async function generateOpenAIVoice(
  text: string,
  language: string
): Promise<{ audioBase64: string; cost: number }> {
  
  const voiceMap: Record<string, string> = {
    'en': 'alloy',
    'es': 'nova',
    'fr': 'shimmer',
    'de': 'alloy',
    'it': 'nova',
    'pt': 'shimmer',
    'zh': 'alloy',
    'ja': 'nova',
    'ko': 'shimmer',
    'ar': 'fable',
    'hi': 'onyx',
  };

  const response = await fetch('https://api.openai.com/v1/audio/speech', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'tts-1',
      input: text,
      voice: voiceMap[language] || 'alloy',
      response_format: 'mp3',
      speed: 0.9,
    }),
  });

  if (!response.ok) {
    throw new Error(`TTS API error: ${response.statusText}`);
  }

  const audioBuffer = await response.arrayBuffer();
  const audioBase64 = Buffer.from(audioBuffer).toString('base64');
  
  // Calculate cost (approximately $0.015 per 1K characters)
  const cost = (text.length / 1000) * 0.015;

  return { audioBase64, cost };
}

/**
 * Generate contextual follow-up questions
 */
async function generateFollowUpQuestions(
  transcriptionText: string,
  legalContext: string[],
  language: string
): Promise<string[]> {
  
  const questionTemplates: Record<string, Record<string, string[]>> = {
    en: {
      'personal-injury': [
        'When exactly did this accident occur?',
        'Did you seek medical attention immediately?',
        'Were there any witnesses present?',
        'Do you have photographs of your injuries or the scene?',
      ],
      'family-law': [
        'How long were you married?',
        'Do you have children together?',
        'Are there significant assets to divide?',
        'Have you attempted mediation?',
      ],
      'criminal': [
        'What specific charges are you facing?',
        'When is your next court date?',
        'Have you spoken to the police without an attorney?',
        'Do you have any evidence in your favor?',
      ],
      'employment': [
        'How long did you work for this employer?',
        'Did you file any internal complaints?',
        'Do you have documentation of the incidents?',
        'Were there witnesses to the events?',
      ],
    },
    es: {
      'personal-injury': [
        '¿Cuándo ocurrió exactamente este accidente?',
        '¿Buscó atención médica inmediatamente?',
        '¿Había testigos presentes?',
        '¿Tiene fotografías de sus lesiones o la escena?',
      ],
      'family-law': [
        '¿Por cuánto tiempo estuvieron casados?',
        '¿Tienen hijos juntos?',
        '¿Hay bienes significativos que dividir?',
        '¿Han intentado la mediación?',
      ],
    },
  };

  const contextQuestions = legalContext.map(context => {
    const questions = questionTemplates[language]?.[context] || questionTemplates.en[context] || [];
    return questions[Math.floor(Math.random() * questions.length)];
  }).filter(Boolean);

  // Add generic follow-up if no specific context
  if (contextQuestions.length === 0) {
    const genericQuestions = language === 'es' 
      ? ['¿Puede contarme más detalles sobre lo que pasó?', '¿Cuándo ocurrió esto?']
      : ['Can you tell me more details about what happened?', 'When did this occur?'];
    
    contextQuestions.push(genericQuestions[0]);
  }

  return contextQuestions.slice(0, 2); // Return max 2 questions
}

// src/components/EnhancedVoiceAvatar.tsx
import React, { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Volume2, VolumeX, Loader, Zap, Brain } from 'lucide-react';

interface EnhancedVoiceAvatarProps {
  language: string;
  intakeData: any;
  onTranscript: (transcript: string) => void;
  onAIResponse: (response: string) => void;
  onCostUpdate: (cost: number) => void;
}

export const EnhancedVoiceAvatar: React.FC<EnhancedVoiceAvatarProps> = ({
  language,
  intakeData,
  onTranscript,
  onAIResponse,
  onCostUpdate,
}) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [processingMethod, setProcessingMethod] = useState<string>('');
  const [qualityScore, setQualityScore] = useState<number>(0);
  const [totalCost, setTotalCost] = useState<number>(0);
  const [error, setError] = useState<string>('');
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Initialize audio recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000,
        } 
      });

      // Set up audio level monitoring
      audioContextRef.current = new AudioContext();
      analyserRef.current = audioContextRef.current.createAnalyser();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyserRef.current);
      analyserRef.current.fftSize = 256;

      // Start recording
      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      audioChunksRef.current = [];
      
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        await processAudio(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setError('');
      
      // Start audio level monitoring
      monitorAudioLevel();

    } catch (error) {
      console.error('Failed to start recording:', error);
      setError('Microphone access denied. Please allow microphone access and try again.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsProcessing(true);
      
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    }
  };