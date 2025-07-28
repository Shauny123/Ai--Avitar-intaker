// src/pages/api/ai-conversation.ts
import { NextApiRequest, NextApiResponse } from 'next';

interface ConversationRequest {
  message: string;
  language: string;
  context: string;
  intakeData?: any;
  conversationHistory?: Array<{
    role: 'user' | 'assistant';
    content: string;
  }>;
}

interface ConversationResponse {
  response: string;
  followUpQuestions?: string[];
  extractedInfo?: any;
  confidence: number;
  error?: string;
}

// Legal intake conversation prompts by language
const SYSTEM_PROMPTS = {
  en: `You are a compassionate AI legal intake assistant. Your role is to:
1. Gather detailed information about the user's legal situation
2. Ask clarifying questions in a warm, professional manner
3. Show empathy for their situation
4. Extract key legal facts (dates, names, damages, etc.)
5. Explain that you're not providing legal advice
6. Guide them through the intake process smoothly

Keep responses conversational, under 50 words, and ask one question at a time.`,

  es: `Eres un asistente de IA compasivo para admisión legal. Tu papel es:
1. Recopilar información detallada sobre la situación legal del usuario
2. Hacer preguntas aclaratorias de manera cálida y profesional
3. Mostrar empatía por su situación
4. Extraer hechos legales clave (fechas, nombres, daños, etc.)
5. Explicar que no estás proporcionando asesoría legal
6. Guiarlos a través del proceso de admisión sin problemas

Mantén las respuestas conversacionales, menos de 50 palabras, y haz una pregunta a la vez.`,

  ar: `أنت مساعد ذكي متعاطف لاستقبال الحالات القانونية. دورك هو:
1. جمع معلومات مفصلة عن الوضع القانوني للمستخدم
2. طرح أسئلة توضيحية بطريقة دافئة ومهنية
3. إظهار التعاطف مع وضعهم
4. استخراج الحقائق القانونية الأساسية (التواريخ، الأسماء، الأضرار، إلخ)
5. توضيح أنك لا تقدم استشارة قانونية
6. إرشادهم خلال عملية الاستقبال بسلاسة

احتفظ بالردود محادثية، أقل من 50 كلمة، واطرح سؤالاً واحداً في كل مرة.`,
};

// Legal question categories and follow-ups
const LEGAL_CATEGORIES = {
  'personal-injury': {
    keywords: ['accident', 'injury', 'hurt', 'hospital', 'car crash', 'slip', 'fall'],
    questions: [
      'When did this accident happen?',
      'Did you seek medical attention immediately?',
      'Was a police report filed?',
      'Do you have any witnesses?',
      'What injuries did you sustain?'
    ]
  },
  'family-law': {
    keywords: ['divorce', 'custody', 'marriage', 'children', 'spouse', 'alimony'],
    questions: [
      'How long were you married?',
      'Do you have children together?',
      'Have you tried mediation?',
      'Are there any assets to divide?',
      'Is there domestic violence involved?'
    ]
  },
  'criminal': {
    keywords: ['arrested', 'charged', 'police', 'court', 'bail', 'crime'],
    questions: [
      'What are you being charged with?',
      'When is your court date?',
      'Have you been arrested before?',
      'Do you have a public defender?',
      'What evidence do they have?'
    ]
  },
  'employment': {
    keywords: ['fired', 'job', 'workplace', 'harassment', 'discrimination', 'wages'],
    questions: [
      'How long did you work there?',
      'Did you file a complaint with HR?',
      'Do you have documentation?',
      'Were there witnesses?',
      'What was your position?'
    ]
  }
};

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<ConversationResponse>
) {
  if (req.method !== 'POST') {
    return res.status(405).json({
      response: '',
      confidence: 0,
      error: 'Method not allowed'
    });
  }

  try {
    const {
      message,
      language = 'en',
      context,
      intakeData,
      conversationHistory = []
    }: ConversationRequest = req.body;

    // Validate input
    if (!message || message.trim().length === 0) {
      return res.status(400).json({
        response: '',
        confidence: 0,
        error: 'Message is required'
      });
    }

    // Process the conversation
    const aiResponse = await processConversation({
      message,
      language,
      context,
      intakeData,
      conversationHistory
    });

    res.status(200).json(aiResponse);
  } catch (error) {
    console.error('AI Conversation Error:', error);
    res.status(500).json({
      response: 'I apologize, but I encountered an error. Please try again.',
      confidence: 0,
      error: 'Internal server error'
    });
  }
}

async function processConversation({
  message,
  language,
  context,
  intakeData,
  conversationHistory
}: ConversationRequest): Promise<ConversationResponse> {
  
  // Detect legal category
  const category = detectLegalCategory(message);
  
  // Extract key information
  const extractedInfo = extractLegalInfo(message);
  
  // Generate contextual response
  const response = await generateAIResponse({
    message,
    language,
    category,
    extractedInfo,
    intakeData,
    conversationHistory
  });

  // Get follow-up questions
  const followUpQuestions = getFollowUpQuestions(category, language);

  return {
    response,
    followUpQuestions: followUpQuestions.slice(0, 2), // Limit to 2 questions
    extractedInfo,
    confidence: calculateConfidence(message, extractedInfo)
  };
}

function detectLegalCategory(message: string): string {
  const lowerMessage = message.toLowerCase();
  
  for (const [category, config] of Object.entries(LEGAL_CATEGORIES)) {
    const matches = config.keywords.filter(keyword => 
      lowerMessage.includes(keyword)
    );
    
    if (matches.length > 0) {
      return category;
    }
  }
  
  return 'general';
}

function extractLegalInfo(message: string): any {
  const info: any = {};
  
  // Extract dates
  const dateRegex = /(\d{1,2}\/\d{1,2}\/\d{2,4}|\d{1,2}-\d{1,2}-\d{2,4}|january|february|march|april|may|june|july|august|september|october|november|december)/gi;
  const dates = message.match(dateRegex);
  if (dates) info.dates = dates;
  
  // Extract monetary amounts
  const moneyRegex = /\$[\d,]+(?:\.\d{2})?/g;
  const amounts = message.match(moneyRegex);
  if (amounts) info.amounts = amounts;
  
  // Extract names (basic pattern)
  const nameRegex = /(?:my|the|mr\.?|mrs\.?|ms\.?|dr\.?)\s+([A-Z][a-z]+ [A-Z][a-z]+)/g;
  const names = message.match(nameRegex);
  if (names) info.names = names;
  
  // Extract locations
  const locationRegex = /(?:in |at |on )([A-Z][a-z]+(?: [A-Z][a-z]+)*(?:,? [A-Z]{2})?)/g;
  const locations = message.match(locationRegex);
  if (locations) info.locations = locations;
  
  return info;
}

async function generateAIResponse({
  message,
  language,
  category,
  extractedInfo,
  intakeData,
  conversationHistory
}: {
  message: string;
  language: string;
  category: string;
  extractedInfo: any;
  intakeData?: any;
  conversationHistory: Array<{role: string, content: string}>;
}): Promise<string> {
  
  // Use OpenAI API if available, otherwise use rule-based responses
  if (process.env.OPENAI_API_KEY) {
    return await generateOpenAIResponse({
      message,
      language,
      category,
      extractedInfo,
      intakeData,
      conversationHistory
    });
  } else {
    return generateRuleBasedResponse({
      message,
      language,
      category,
      extractedInfo,
      intakeData,
      conversationHistory
    });
  }
}

async function generateOpenAIResponse({
  message,
  language,
  category,
  extractedInfo,
  intakeData,
  conversationHistory
}: any): Promise<string> {
  try {
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.