// src/services/vectorStore.ts
import { createHash } from 'crypto';

interface DocumentChunk {
  id: string;
  content: string;
  embedding: number[];
  metadata: {
    source: string;
    legalDomain: string;
    jurisdiction: string;
    language: string;
    lastUpdated: Date;
    relevanceScore?: number;
  };
  similarity?: number;
}

interface SearchOptions {
  language: string;
  legalContext: string;
  jurisdiction: string;
  limit: number;
  threshold?: number;
}

export class VectorStore {
  private documents: Map<string, DocumentChunk> = new Map();
  private embeddings: Map<string, number[]> = new Map();
  private initialized = false;

  constructor() {
    this.initializeWithLegalDocuments();
  }

  /**
   * Initialize vector store with legal document embeddings
   */
  private async initializeWithLegalDocuments(): Promise<void> {
    console.log('🔍 Initializing Vector Store with Legal Documents...');
    
    // Load pre-computed legal document embeddings
    const legalDocuments = await this.loadLegalDocuments();
    
    for (const doc of legalDocuments) {
      await this.addDocument(doc);
    }
    
    this.initialized = true;
    console.log(`✅ Vector Store initialized with ${this.documents.size} legal documents`);
  }

  /**
   * Add a document to the vector store
   */
  async addDocument(doc: Omit<DocumentChunk, 'id' | 'embedding'>): Promise<string> {
    const id = this.generateDocumentId(doc.content, doc.metadata.source);
    
    // Generate embedding for document content
    const embedding = await this.generateEmbedding(doc.content, doc.metadata.language);
    
    const documentChunk: DocumentChunk = {
      id,
      ...doc,
      embedding
    };
    
    this.documents.set(id, documentChunk);
    this.embeddings.set(id, embedding);
    
    return id;
  }

  /**
   * Semantic search using cosine similarity
   */
  async similaritySearch(
    query: string, 
    options: SearchOptions
  ): Promise<DocumentChunk[]> {
    
    if (!this.initialized) {
      await this.waitForInitialization();
    }

    // Generate embedding for query
    const queryEmbedding = await this.generateEmbedding(query, options.language);
    
    // Calculate similarities
    const similarities: Array<{id: string, similarity: number}> = [];
    
    for (const [id, docEmbedding] of this.embeddings.entries()) {
      const doc = this.documents.get(id)!;
      
      // Filter by context and jurisdiction
      if (!this.matchesContext(doc, options)) {
        continue;
      }
      
      const similarity = this.cosineSimilarity(queryEmbedding, docEmbedding);
      
      if (similarity > (options.threshold || 0.3)) {
        similarities.push({ id, similarity });
      }
    }
    
    // Sort by similarity and apply limit
    similarities.sort((a, b) => b.similarity - a.similarity);
    const topResults = similarities.slice(0, options.limit);
    
    // Return documents with similarity scores
    return topResults.map(result => {
      const doc = this.documents.get(result.id)!;
      return {
        ...doc,
        similarity: result.similarity
      };
    });
  }

  /**
   * Generate embeddings using various methods
   */
  private async generateEmbedding(text: string, language: string): Promise<number[]> {
    // Try OpenAI embeddings first (best quality)
    if (process.env.OPENAI_API_KEY) {
      return await this.generateOpenAIEmbedding(text);
    }
    
    // Fallback to local embeddings
    return await this.generateLocalEmbedding(text, language);
  }

  /**
   * OpenAI embeddings (high quality)
   */
  private async generateOpenAIEmbedding(text: string): Promise<number[]> {
    try {
      const response = await fetch('https://api.openai.com/v1/embeddings', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          input: text,
          model: 'text-embedding-3-small'
        }),
      });

      const data = await response.json();
      return data.data[0].embedding;
    } catch (error) {
      console.error('OpenAI embedding failed:', error);
      return await this.generateLocalEmbedding(text, 'en');
    }
  }

  /**
   * Local embeddings (fallback)
   */
  private async generateLocalEmbedding(text: string, language: string): Promise<number[]> {
    // Simple TF-IDF based embedding for fallback
    const words = this.tokenize(text, language);
    const embedding = new Array(384).fill(0); // 384-dimensional embedding
    
    // Create a hash-based embedding
    for (let i = 0; i < words.length; i++) {
      const word = words[i];
      const hash = this.hashString(word);
      
      for (let j = 0; j < embedding.length; j++) {
        embedding[j] += Math.sin(hash + j) * (1 / (i + 1));
      }
    }
    
    // Normalize
    const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    return embedding.map(val => val / magnitude);
  }

  /**
   * Calculate cosine similarity between two vectors
   */
  private cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  /**
   * Check if document matches search context
   */
  private matchesContext(doc: DocumentChunk, options: SearchOptions): boolean {
    // Language match
    if (doc.metadata.language