// src/services/audioProcessingPipeline.ts
import { VectorStore } from './vectorStore';
import { LegalKnowledgeBase } from './legalKnowledgeBase';

interface AudioProcessingConfig {
  whisperEndpoint: string;
  flamingoEndpoint: string;
  qualityThreshold: number;
  costOptimization: boolean;
  fallbackEnabled: boolean;
}

interface AudioQualityMetrics {
  snr: number; // Signal-to-noise ratio
  clarity: number; // Speech clarity score
  backgroundNoise: number;
  duration: number;
  sampleRate: number;
  bitrate: number;
}

interface TranscriptionResult {
  text: string;
  confidence: number;
  language: string;
  timestamps: Array<{start: number, end: number, text: string}>;
  processingMethod: 'whisper' | 'flamingo3' | 'hybrid';
  cost: number;
  qualityMetrics: AudioQualityMetrics;
}

interface RAGSResponse {
  response: string;
  sources: Array<{
    content: string;
    source: string;
    relevance: number;
  }>;
  confidence: number;
  legalContext: string[];
}

export class AudioProcessingPipeline {
  private config: AudioProcessingConfig;
  private vectorStore: VectorStore;
  private legalKB: LegalKnowledgeBase;
  private audioContext: AudioContext | null = null;

  constructor(config: AudioProcessingConfig) {
    this.config = config;
    this.vectorStore = new VectorStore();
    this.legalKB = new LegalKnowledgeBase();
  }

  /**
   * Main audio processing pipeline with intelligent routing
   */
  async processAudio(
    audioBlob: Blob,
    language: string = 'en',
    userContext: any = {}
  ): Promise<{transcription: TranscriptionResult, rags: RAGSResponse}> {
    
    // Step 1: Analyze audio quality
    const qualityMetrics = await this.analyzeAudioQuality(audioBlob);
    console.log('📊 Audio Quality Analysis:', qualityMetrics);

    // Step 2: Choose optimal processing path based on quality and cost
    const processingMethod = this.selectProcessingMethod(qualityMetrics);
    console.log('🎯 Selected Processing Method:', processingMethod);

    // Step 3: Transcribe audio using selected method
    const transcription = await this.transcribeAudio(
      audioBlob, 
      language, 
      processingMethod,
      qualityMetrics
    );

    // Step 4: Process through RAGS for legal context
    const rags = await this.processWithRAGS(
      transcription.text,
      language,
      userContext,
      transcription
    );

    return { transcription, rags };
  }

  /**
   * Analyze audio quality to determine optimal processing method
   */
  private async analyzeAudioQuality(audioBlob: Blob): Promise<AudioQualityMetrics> {
    return new Promise(async (resolve) => {
      const arrayBuffer = await audioBlob.arrayBuffer();
      
      if (!this.audioContext) {
        this.audioContext = new AudioContext();
      }

      try {
        const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
        const channelData = audioBuffer.getChannelData(0);
        
        // Calculate Signal-to-Noise Ratio
        const snr = this.calculateSNR(channelData);
        
        // Calculate speech clarity (frequency analysis)
        const clarity = this.calculateClarity(channelData, audioBuffer.sampleRate);
        
        // Calculate background noise level
        const backgroundNoise = this.calculateBackgroundNoise(channelData);

        resolve({
          snr,
          clarity,
          backgroundNoise,
          duration: audioBuffer.duration,
          sampleRate: audioBuffer.sampleRate,
          bitrate: audioBlob.size * 8 / audioBuffer.duration
        });
      } catch (error) {
        console.error('Audio analysis failed:', error);
        // Return default low-quality metrics to trigger fallback
        resolve({
          snr: 5,
          clarity: 0.3,
          backgroundNoise: 0.8,
          duration: 0,
          sampleRate: 16000,
          bitrate: 32000
        });
      }
    });
  }

  /**
   * Select optimal processing method based on quality and cost
   */
  private selectProcessingMethod(metrics: AudioQualityMetrics): 'whisper' | 'flamingo3' | 'hybrid' {
    const qualityScore = (metrics.snr * 0.4) + (metrics.clarity * 0.4) + ((1 - metrics.backgroundNoise) * 0.2);
    
    console.log(`🔍 Quality Score: ${qualityScore.toFixed(2)}/10`);

    // High quality → Whisper (better accuracy, higher cost)
    if (qualityScore > 7 && this.config.costOptimization) {
      return 'whisper';
    }
    
    // Medium quality → Hybrid approach
    if (qualityScore > 4) {
      return 'hybrid';
    }
    
    // Low quality → NVIDIA Flamingo 3 (optimized for poor audio, lower cost)
    return 'flamingo3';
  }

  /**
   * Transcribe audio using selected method
   */
  private async transcribeAudio(
    audioBlob: Blob,
    language: string,
    method: 'whisper' | 'flamingo3' | 'hybrid',
    qualityMetrics: AudioQualityMetrics
  ): Promise<TranscriptionResult> {
    
    const startTime = Date.now();
    let result: TranscriptionResult;

    try {
      switch (method) {
        case 'whisper':
          result = await this.transcribeWithWhisper(audioBlob, language);
          break;
          
        case 'flamingo3':
          result = await this.transcribeWithFlamingo3(audioBlob, language, qualityMetrics);
          break;
          
        case 'hybrid':
          result = await this.transcribeHybrid(audioBlob, language, qualityMetrics);
          break;
          
        default:
          throw new Error(`Unknown processing method: ${method}`);
      }

      result.processingMethod = method;
      result.qualityMetrics = qualityMetrics;
      
      console.log(`✅ Transcription completed in ${Date.now() - startTime}ms using ${method}`);
      return result;

    } catch (error) {
      console.error(`❌ Transcription failed with ${method}:`, error);
      
      // Fallback to alternative method
      if (method !== 'flamingo3' && this.config.fallbackEnabled) {
        console.log('🔄 Attempting fallback to NVIDIA Flamingo 3...');
        return await this.transcribeWithFlamingo3(audioBlob, language, qualityMetrics);
      }
      
      throw error;
    }
  }

  /**
   * Whisper transcription (high quality audio)
   */
  private async transcribeWithWhisper(audioBlob: Blob, language: string): Promise<TranscriptionResult> {
    const formData = new FormData();
    formData.append('file', audioBlob, 'audio.webm');
    formData.append('model', 'whisper-1');
    formData.append('language', this.mapLanguageCode(language));
    formData.append('response_format', 'verbose_json');
    formData.append('timestamp_granularities[]', 'word');

    const response = await fetch(`${this.config.whisperEndpoint}/v1/audio/transcriptions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
      },
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Whisper API error: ${response.statusText}`);
    }

    const data = await response.json();
    
    return {
      text: data.text,
      confidence: this.calculateWhisperConfidence(data),
      language: data.language || language,
      timestamps: data.words || [],
      processingMethod: 'whisper',
      cost: this.calculateWhisperCost(audioBlob),
      qualityMetrics: {} as AudioQualityMetrics // Will be filled by caller
    };
  }

  /**
   * NVIDIA Flamingo 3 transcription (optimized for poor quality audio)
   */
  private async transcribeWithFlamingo3(
    audioBlob: Blob, 
    language: string, 
    qualityMetrics: AudioQualityMetrics
  ): Promise<TranscriptionResult> {
    
    // Convert audio to optimal format for Flamingo 3
    const processedAudio = await this.preprocessAudioForFlamingo(audioBlob, qualityMetrics);
    
    const formData = new FormData();
    formData.append('audio', processedAudio, 'audio.wav');
    formData.append('language', language);
    formData.append('enhance_audio', 'true'); // Enable audio enhancement
    formData.append('noise_reduction', 'adaptive'); // Adaptive noise reduction
    formData.append('quality_mode', 'high_tolerance'); // Optimized for poor quality

    const response = await fetch(`${this.config.flamingoEndpoint}/api/v1/transcribe`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.NVIDIA_API_KEY}`,
        'Accept': 'application/json',
      },
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Flamingo 3 API error: ${response.statusText}`);
    }

    const data = await response.json();
    
    return {
      text: data.transcription,
      confidence: data.confidence || 0.8,
      language: data.detected_language || language,
      timestamps: data.word_timestamps || [],
      processingMethod: 'flamingo3',
      cost: this.calculateFlamingoVCost(processedAudio),
      qualityMetrics
    };
  }

  /**
   * Hybrid transcription approach
   */
  private async transcribeHybrid(
    audioBlob: Blob, 
    language: string, 
    qualityMetrics: AudioQualityMetrics
  ): Promise<TranscriptionResult> {
    
    // Process with both services in parallel
    const [whisperResult, flamingoResult] = await Promise.allSettled([
      this.transcribeWithWhisper(audioBlob, language),
      this.transcribeWithFlamingo3(audioBlob, language, qualityMetrics)
    ]);

    // Choose best result based on confidence and cost
    let bestResult: TranscriptionResult;
    
    if (whisperResult.status === 'fulfilled' && flamingoResult.status === 'fulfilled') {
      const whisperScore = whisperResult.value.confidence * 0.7 + (1 - whisperResult.value.cost / 100) * 0.3;
      const flamingoScore = flamingoResult.value.confidence * 0.7 + (1 - flamingoResult.value.cost / 100) * 0.3;
      
      bestResult = whisperScore > flamingoScore ? whisperResult.value : flamingoResult.value;
      bestResult.processingMethod = 'hybrid';
      
      console.log(`🤖 Hybrid: Whisper=${whisperScore.toFixed(2)}, Flamingo=${flamingoScore.toFixed(2)}, Selected=${bestResult.processingMethod}`);
      
    } else if (whisperResult.status === 'fulfilled') {
      bestResult = whisperResult.value;
    } else if (flamingoResult.status === 'fulfilled') {
      bestResult = flamingoResult.value;
    } else {
      throw new Error('Both transcription services failed');
    }

    return bestResult;
  }

  /**
   * RAGS processing for legal context
   */
  private async processWithRAGS(
    text: string,
    language: string,
    userContext: any,
    transcription: TranscriptionResult
  ): Promise<RAGSResponse> {
    
    console.log('🧠 Processing with RAGS...');
    
    // Step 1: Extract legal entities and intent
    const legalEntities = await this.extractLegalEntities(text, language);
    
    // Step 2: Retrieve relevant legal documents from vector store
    const relevantDocs = await this.vectorStore.similaritySearch(
      text,
      {
        language,
        legalContext: userContext.caseType || 'general',
        jurisdiction: userContext.state || 'federal',
        limit: 5
      }
    );

    // Step 3: Get legal knowledge base context
    const legalContext = await this.legalKB.getContext(legalEntities, language);

    // Step 4: Generate contextual response
    const response = await this.generateContextualResponse({
      originalText: text,
      relevantDocs,
      legalContext,
      language,
      userContext,
      transcriptionQuality: transcription.confidence
    });

    return {
      response: response.text,
      sources: relevantDocs.map(doc => ({
        content: doc.content,
        source: doc.metadata.source,
        relevance: doc.similarity
      })),
      confidence: response.confidence,
      legalContext: legalContext.map(ctx => ctx.category)
    };
  }

  /**
   * Audio quality calculation helpers
   */
  private calculateSNR(channelData: Float32Array): number {
    let signalPower = 0;
    let noisePower = 0;
    
    // Simple SNR calculation
    for (let i = 0; i < channelData.length; i++) {
      const sample = channelData[i];
      signalPower += sample * sample;
    }
    
    // Estimate noise from quieter sections
    const sortedSamples = Array.from(channelData).sort((a, b) => Math.abs(a) - Math.abs(b));
    const noiseThreshold = sortedSamples.length * 0.1; // Bottom 10%
    
    for (let i = 0; i < noiseThreshold; i++) {
      const sample = sortedSamples[i];
      noisePower += sample * sample;
    }
    
    const snr = 10 * Math.log10(signalPower / (noisePower || 0.001));
    return Math.max(0, Math.min(20, snr)); // Clamp between 0-20
  }

  private calculateClarity(channelData: Float32Array, sampleRate: number): number {
    // Simplified clarity calculation based on frequency distribution
    const fftSize = 2048;
    const fft = new FFT(fftSize);
    
    let clarityScore = 0;
    const chunks = Math.floor(channelData.length / fftSize);
    
    for (let i = 0; i < chunks; i++) {
      const chunk = channelData.slice(i * fftSize, (i + 1) * fftSize);
      const spectrum = fft.forward(chunk);
      
      // Focus on speech frequencies (300-3400 Hz)
      const speechBandPower = this.getPowerInBand(spectrum, 300, 3400, sampleRate);
      const totalPower = this.getTotalPower(spectrum);
      
      clarityScore += speechBandPower / totalPower;
    }
    
    return clarityScore / chunks;
  }

  private calculateBackgroundNoise(channelData: Float32Array): number {
    // Calculate RMS of quieter sections
    const sortedSamples = Array.from(channelData)
      .map(x => Math.abs(x))
      .sort((a, b) => a - b);
    
    const quietSamples = sortedSamples.slice(0, Math.floor(sortedSamples.length * 0.2));
    const rms = Math.sqrt(quietSamples.reduce((sum, x) => sum + x * x, 0) / quietSamples.length);
    
    return Math.min(1, rms * 10); // Normalize to 0-1
  }

  /**
   * Cost calculation helpers
   */
  private calculateWhisperCost(audioBlob: Blob): number {
    // Whisper pricing: $0.006 per minute
    const durationMinutes = audioBlob.size / (16000 * 2 * 60); // Estimate based on size
    return durationMinutes * 0.006;
  }

  private calculateFlamingoVCost(audioBlob: Blob): number {
    // NVIDIA Flamingo 3 typically lower cost for poor quality audio
    const durationMinutes = audioBlob.size / (16000 * 2 * 60);
    return durationMinutes * 0.003; // ~50% of Whisper cost
  }

  /**
   * Audio preprocessing for Flamingo 3
   */
  private async preprocessAudioForFlamingo(
    audioBlob: Blob, 
    qualityMetrics: AudioQualityMetrics
  ): Promise<Blob> {
    
    if (!this.audioContext) {
      this.audioContext = new AudioContext();
    }

    const arrayBuffer = await audioBlob.arrayBuffer();
    const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
    
    // Apply noise reduction and normalization for poor quality audio
    const processedBuffer = await this.applyAudioEnhancements(audioBuffer, qualityMetrics);
    
    // Convert back to blob
    return this.audioBufferToBlob(processedBuffer);
  }

  private async applyAudioEnhancements(
    audioBuffer: AudioBuffer, 
    qualityMetrics: AudioQualityMetrics
  ): Promise<AudioBuffer> {
    
    const channelData = audioBuffer.getChannelData(0);
    const enhanced = new Float32Array(channelData.length);
    
    // Apply adaptive filtering based on quality metrics
    if (qualityMetrics.backgroundNoise > 0.5) {
      // High noise - apply aggressive noise reduction
      this.applyNoiseReduction(channelData, enhanced, 0.8);
    } else {
      // Low noise - light filtering
      this.applyNoiseReduction(channelData, enhanced, 0.3);
    }
    
    // Normalize audio levels
    this.normalizeAudio(enhanced);
    
    // Create new AudioBuffer with enhanced data
    const enhancedBuffer = this.audioContext!.createBuffer(
      1, 
      enhanced.length, 
      audioBuffer.sampleRate
    );
    enhancedBuffer.copyToChannel(enhanced, 0);
    
    return enhancedBuffer;
  }

  private applyNoiseReduction(input: Float32Array, output: Float32Array, strength: number): void {
    // Simple noise gate + low-pass filter
    const threshold = 0.01 * (1 - strength);
    
    for (let i = 0; i < input.length; i++) {
      if (Math.abs(input[i]) < threshold) {
        output[i] = 0; // Gate out low-level noise
      } else {
        output[i] = input[i];
      }
    }
  }

  private normalizeAudio(data: Float32Array): void {
    const max = Math.max(...data.map(x => Math.abs(x)));
    if (max > 0) {
      const scale = 0.95 / max;
      for (let i = 0; i < data.length; i++) {
        data[i] *= scale;
      }
    }
  }

  private async audioBufferToBlob(audioBuffer: AudioBuffer): Promise<Blob> {
    // Convert AudioBuffer to WAV blob
    const length = audioBuffer.length;
    const sampleRate = audioBuffer.sampleRate;
    const arrayBuffer = new ArrayBuffer(44 + length * 2);
    const view = new DataView(arrayBuffer);
    
    // WAV header
    const writeString = (offset: number, string: string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, length * 2, true);
    
    // Convert float samples to 16-bit PCM
    const channelData = audioBuffer.getChannelData(0);
    let offset = 44;
    for (let i = 0; i < length; i++) {
      const sample = Math.max(-1, Math.min(1, channelData[i]));
      view.setInt16(offset, sample * 0x7FFF, true);
      offset += 2;
    }
    
    return new Blob([arrayBuffer], { type: 'audio/wav' });
  }

  private mapLanguageCode(code: string): string {
    const mapping: Record<string, string> = {
      'en': 'en', 'es': 'es', 'fr': 'fr', 'de': 'de', 'it': 'it',
      'zh': 'zh', 'ja': 'ja', 'ko': 'ko', 'ar': 'ar', 'hi': 'hi',
      'pt-br': 'pt', 'ru': 'ru', 'tr': 'tr', 'th': 'th', 'vi': 'vi'
    };
    return mapping[code] || 'en';
  }

  private calculateWhisperConfidence(data: any): number {
    // Calculate confidence based on Whisper response metadata
    return data.segments?.reduce((acc: number, seg: any) => 
      acc + (seg.avg_logprob || 0), 0) / (data.segments?.length || 1) + 1 || 0.8;
  }

  // Placeholder implementations for complex methods
  private getPowerInBand(spectrum: any, lowFreq: number, highFreq: number, sampleRate: number): number {
    // FFT band power calculation - simplified
    return 0.5;
  }

  private getTotalPower(spectrum: any): number {
    // Total spectrum power - simplified
    return 1.0;
  }

  private async extractLegalEntities(text: string, language: string): Promise<any[]> {
    // Extract legal entities using NLP - placeholder
    return [];
  }

  private async generateContextualResponse(params: any): Promise<{text: string, confidence: number}> {
    // Generate response using RAGS - placeholder
    return { text: "Thank you for sharing that information. Can you tell me more about when this happened?", confidence: 0.9 };
  }
}

// Simple FFT implementation placeholder
class FFT {
  constructor(private size: number) {}
  
  forward(input: Float32Array): any {
    // Simplified FFT - in production use a proper FFT library
    return new Array(this.size).fill(0);
  }
}

export default AudioProcessingPipeline;