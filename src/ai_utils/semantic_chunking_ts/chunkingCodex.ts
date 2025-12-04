import { promises as fs } from "node:fs";
import path from "node:path";
import { split as splitSentences } from "sentence-splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { encoding_for_model } from "tiktoken";
import type { Tiktoken, TiktokenModel } from "tiktoken";

export interface SemanticChunkerConfig {
    embeddingModel: string;
    tokenizerModel?: TiktokenModel;
    similarityThreshold?: number;
    openAIApiKey?: string;
}

export interface ChunkerRequest {
    text: string;
    outputDir: string;
    maxTokens: number;
    minTokens?: number;
    baseFilename?: string;
    metadata?: Record<string, unknown>;
}

export interface ChunkMetadata {
    id: number;
    text: string;
    sentences: string[];
    tokenCount: number;
    filePath: string;
    metadata?: Record<string, unknown>;
}

export class SemanticChunkerCodex {
    private readonly embeddings: OpenAIEmbeddings;
    private readonly tokenizer: Tiktoken;
    private readonly similarityThreshold: number;

    constructor(private readonly config: SemanticChunkerConfig) {
        if (!config.embeddingModel) throw new Error("embeddingModel is required");

        // Instantiate OpenAI embeddings once – chunking reuses the same client.
        this.embeddings = new OpenAIEmbeddings({
            apiKey: config.openAIApiKey ?? process.env.OPENAI_API_KEY,
            model: config.embeddingModel,
        });

        // Tokenizer handles token limits + cost estimates for different models.
        this.tokenizer = encoding_for_model(config.tokenizerModel ?? "text-embedding-3-small");
        this.similarityThreshold = config.similarityThreshold ?? 0.82;
    }

    async chunkAndSave(request: ChunkerRequest): Promise<ChunkMetadata[]> {
        if (!request.text.trim()) throw new Error("Cannot chunk empty text");
        if (!request.outputDir) throw new Error("outputDir is required");
        if (request.maxTokens <= 0) throw new Error("maxTokens must be greater than 0");

        const sentences = this.splitIntoSentences(request.text);
        if (!sentences.length) return [];

        // Embed each sentence and count tokens in parallel to save latency.
        const [sentenceEmbeddings, tokenCounts] = await Promise.all([
            this.embeddings.embedDocuments(sentences),
            Promise.resolve(sentences.map((sentence) => this.countTokens(sentence))),
        ]);

        const chunks = this.buildChunks(
            sentences,
            sentenceEmbeddings,
            tokenCounts,
            request.maxTokens,
            request.minTokens ?? 0
        );

        await fs.mkdir(request.outputDir, { recursive: true });

        const results: ChunkMetadata[] = [];
        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i];
            const filename =
                (request.baseFilename ?? "chunk") + `_${String(i + 1).padStart(4, "0")}.txt`;
            const filePath = path.join(request.outputDir, filename);

            await fs.writeFile(filePath, chunk.text, "utf8");

            results.push({
                ...chunk,
                id: i,
                filePath,
                metadata: request.metadata
                    ? {
                          ...request.metadata,
                          chunkIndex: i,
                          sentenceCount: chunk.sentences.length,
                          tokenCount: chunk.tokenCount,
                      }
                    : undefined,
            });
        }

        return results;
    }

    dispose(): void {
        this.tokenizer.free();
    }

    private splitIntoSentences(text: string): string[] {
        return splitSentences(text)
            .filter((node) => node.type === "Sentence")
            .map((node) => node.raw.trim())
            .filter(Boolean);
    }

    private buildChunks(
        sentences: string[],
        embeddings: number[][],
        tokenCounts: number[],
        maxTokens: number,
        minTokens: number
    ): Array<Omit<ChunkMetadata, "id" | "filePath">> {
        const chunks: Array<Omit<ChunkMetadata, "id" | "filePath">> = [];
        let bufferSentences: string[] = [];
        let bufferTokens = 0;

        for (let i = 0; i < sentences.length; i++) {
            const sentence = sentences[i];
            const tokens = tokenCounts[i];
            const similarity = i === 0 ? 1 : this.cosineSimilarity(embeddings[i - 1], embeddings[i]);

            // Split when we would exceed the token budget or similarity drops.
            const wouldExceed = bufferTokens + tokens > maxTokens;
            const belowMin = bufferTokens < minTokens;
            const shouldSplitBySimilarity = similarity < this.similarityThreshold && !belowMin;

            if ((wouldExceed && !belowMin) || shouldSplitBySimilarity) {
                chunks.push(this.createChunk(bufferSentences, bufferTokens));
                bufferSentences = [];
                bufferTokens = 0;
            }

            bufferSentences.push(sentence);
            bufferTokens += tokens;
        }

        if (bufferSentences.length) {
            chunks.push(this.createChunk(bufferSentences, bufferTokens));
        }

        // Merge trailing fragment into the previous chunk if it's too tiny.
        if (minTokens > 0 && chunks.length > 1) {
            const lastChunk = chunks[chunks.length - 1];
            if (lastChunk.tokenCount < minTokens) {
                const prevChunk = chunks[chunks.length - 2];
                prevChunk.sentences = prevChunk.sentences.concat(lastChunk.sentences);
                prevChunk.text = prevChunk.sentences.join(" ").trim();
                prevChunk.tokenCount += lastChunk.tokenCount;
                chunks.pop();
            }
        }

        return chunks;
    }

    private createChunk(
        sentences: string[],
        tokenCount: number
    ): Omit<ChunkMetadata, "id" | "filePath" | "metadata"> {
        return {
            sentences: [...sentences],
            text: sentences.join(" ").trim(),
            tokenCount,
        };
    }

    private countTokens(text: string): number {
        return this.tokenizer.encode(text).length;
    }

    private cosineSimilarity(a: number[], b: number[]): number {
        if (a.length !== b.length) throw new Error("Embedding vectors must have equal length");

        let dot = 0;
        let normA = 0;
        let normB = 0;

        for (let i = 0; i < a.length; i++) {
            const valA = a[i];
            const valB = b[i];
            dot += valA * valB;
            normA += valA * valA;
            normB += valB * valB;
        }

        const denom = Math.sqrt(normA) * Math.sqrt(normB);
        if (denom === 0) return 0;

        return dot / denom;
    }
}
