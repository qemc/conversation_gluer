import { split as splitSentences } from "sentence-splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { cosineSimilarity } from "@langchain/core/utils/math";
import { encoding_for_model, type TiktokenModel } from "tiktoken";

type sentenceWindow = {
    id: number,
    sentence: string,
    window?: string,
    windowEmbeddings: number[]
    distanceNext?: number
    isBoundaryCandidate?: boolean,
    distancePercentile?: number
}
export type Chunk = {
    chunkId: string,
    chunkPositionInText: number,
    chunkText: string,
    chunkEmbeddings: number[],
    chunkMetadata: Record<string,string | string[]>,
    chunkTokens: number
}

export class SemanticChunker{
    constructor(
        private readonly windowsSize: number,
        private readonly embeddingModel: string,
    ){}
    private countTokens(text: string):number{

        if(!text) throw new Error("Text field parameter is undefined in countTokens funciton");
        const embeddingEncoding = encoding_for_model(this.embeddingModel as TiktokenModel);

        return embeddingEncoding.encode(text).length
    }
    private splitText(text: string): string[]{

        const result: {
            type: string;
            raw: string;
            }[] =  splitSentences(text);

        const sentences: string[] = result
            .filter(node => node.type === "Sentence")
            .map(node => node.raw.trim());

        return sentences
    }
    private async getSentecesWindow(sentences: string[]){

        let sentenceWindows: sentenceWindow[] = []
        for (let i = 0; i<sentences.length; i++){
            if(this.windowsSize> 0){

                // sentences before
                let sentenceBefore = '';
                for (let x = i - this.windowsSize; x < i; x++) {
                    if (x < 0) continue;
                    sentenceBefore += sentences[x] + " ";
                }
                // sentences after
                let sentenceAfter = '';
                for (let x = i + 1; x < sentences.length && x <= i + this.windowsSize; x++) {
                    sentenceAfter += sentences[x] + " ";
                }
                // gluing text window to flatten differences
                const senteceWindowText = sentenceBefore + ' ' + sentences[i] + ' ' + sentenceAfter
                // console.log(`${senteceWindowText} \n`)
                

                let sentenceWindow: sentenceWindow = {
                    id: i,
                    sentence: sentences[i],
                    window: senteceWindowText,
                    windowEmbeddings: await this.getEmbedding(senteceWindowText)
                }
                
                // cosine similarity/distance to the next sentences
                sentenceWindows.push(sentenceWindow);
                if (i > 0) {
                    if (!sentenceWindows[i - 1].windowEmbeddings)
                        throw new Error("Embeddings undefined");

                    if (!sentenceWindows[i].windowEmbeddings)
                        throw new Error("Embeddings undefined");

                    const previousEmbeddings = sentenceWindows[i - 1].windowEmbeddings!;
                    const currentEmbeddings = sentenceWindows[i].windowEmbeddings!;

                    const sim = cosineSimilarity([previousEmbeddings], [currentEmbeddings]);
                    const distance = 1 - sim[0][0];

                    sentenceWindows[i - 1].distanceNext = distance;
                }
            }
            else if (this.windowsSize === 0){

                let sentece: sentenceWindow = {
                    id: i,
                    sentence: sentences[i],
                    windowEmbeddings: await this.getEmbedding(sentences[i])
                }   

                sentenceWindows.push(sentece);
                if (i > 0) {
                    if (!sentenceWindows[i - 1].windowEmbeddings)
                        throw new Error("Embeddings undefined");

                    if (!sentenceWindows[i].windowEmbeddings)
                        throw new Error("Embeddings undefined");

                    const previousEmbeddings = sentenceWindows[i - 1].windowEmbeddings!;
                    const currentEmbeddings = sentenceWindows[i].windowEmbeddings!;

                    const sim = cosineSimilarity([previousEmbeddings], [currentEmbeddings]);
                    const distance = 1 - sim[0][0];

                    sentenceWindows[i - 1].distanceNext = distance;
                }
            }
        }
        // for(let i = 0; i<sentenceWindows.length; i++){
        //         console.log(sentenceWindows[i])
        //     }
        return sentenceWindows
    }
    private async processSentenceWindows(text:string){

        const sentences = this.splitText(text)
        const windows = await this.getSentecesWindow(sentences)

        return windows
    }
    // -----------------------------
    async chunkPercentile(
        text: string, // text to be chunked
        percentile: number,
        metadata: Record<string,string | string[] | number>
    ): Promise<Chunk[]> {

    const sentenceWindows = await this.processSentenceWindows(text);
    // console.log(sentenceWindows);

    this.computePercentileRanks(sentenceWindows);
    // console.log(sentenceWindows);

    const chunks: Chunk[] = [];
    let currentChunkSentences: string[] = [];

    for (let i = 0; i < sentenceWindows.length; i++) {
        const win = sentenceWindows[i];

        currentChunkSentences.push(win.sentence);

        const dp = win.distancePercentile;
        const isLast = i === sentenceWindows.length - 1;

        const isBoundary = !isLast && dp !== undefined && dp >= percentile;

        if (isBoundary) {
        
        const chunk = {
            chunkId: crypto.randomUUID(),
            chunkPositionInText: chunks.length + 1,
            chunkText: currentChunkSentences.join(" ").trim(),
            chunkEmbeddings: await this.getEmbedding(currentChunkSentences.join(" ").trim()),
            chunkMetadata: metadata,
            chunkTokens: this.countTokens(currentChunkSentences.join(" ").trim()),
        } as Chunk
        
        chunks.push(chunk);
        currentChunkSentences = [];
        }
    }

    if (currentChunkSentences.length > 0) {

        const chunk = {
            chunkId: crypto.randomUUID(),
            chunkPositionInText: chunks.length + 1,
            chunkText: currentChunkSentences.join(" ").trim(),
            chunkEmbeddings: await this.getEmbedding(currentChunkSentences.join(" ").trim()),
            chunkMetadata: metadata,
            chunkTokens: this.countTokens(currentChunkSentences.join(" ").trim()),
        } as Chunk

        chunks.push(chunk);
    }
    return chunks;
    }
    async chunkPercentileMinMax(
        text: string,
        percentile: number,
        metadata: Record<string, string | string[]>,
        minTokens: number,
        maxTokens: number
    ): Promise<Chunk[]> {

    const sentenceWindows = await this.processSentenceWindows(text);
    this.computePercentileRanks(sentenceWindows);

    const chunks: Chunk[] = [];
    let currentChunkSentences: string[] = [];
    let currentChunkTokens = 0;

    const makeChunk = async () => {
        const chunkText = currentChunkSentences.join(" ").trim();

        const chunk: Chunk = {
        chunkId: crypto.randomUUID(),
        chunkPositionInText: chunks.length + 1,
        chunkText,
        chunkEmbeddings: await this.getEmbedding(chunkText),
        chunkMetadata: metadata,
        chunkTokens: currentChunkTokens || this.countTokens(chunkText),
        };

        chunks.push(chunk);
        currentChunkSentences = [];
        currentChunkTokens = 0;
    };

    for (let i = 0; i < sentenceWindows.length; i++) {
        const win = sentenceWindows[i];
        const isLast = i === sentenceWindows.length - 1;

        const sentence = win.sentence.trim();
        const sentenceTokens = this.countTokens(sentence);

        // add current sentence to the chunk
        currentChunkSentences.push(sentence);
        currentChunkTokens += sentenceTokens;

        const dp = win.distancePercentile;

        const meetsSemanticBoundary =
        !isLast && dp !== undefined && dp >= percentile;

        const meetsMinSize = currentChunkTokens >= minTokens;
        const exceedsMaxSize = currentChunkTokens >= maxTokens;

        const shouldSplit =
        // semantic boundary + chunk is big enough
        (meetsSemanticBoundary && meetsMinSize) ||
        // hard cap on chunk size
        exceedsMaxSize;

        if (shouldSplit) {
        makeChunk();
        }
    }

    // flush tail chunk (if anything left)
    if (currentChunkSentences.length > 0) {
        makeChunk();
    }

    return chunks;
    }
    async chunkMeanStd(
        text: string
    ){
        let sentences = this.processSentenceWindows(text)
    }
    async chunkMedianMad(
        text: string
    ){
        let sentences = this.processSentenceWindows(text)
    }


    private async getEmbedding(text:string): Promise<number[]>{

        if (!text || text.trim() === "") throw new Error("Empty text passed to getEmbedding()");

        const embeddings = new OpenAIEmbeddings({
            model: this.embeddingModel, 
            apiKey: process.env.OPENAI_API_KEY,
        });

        const result = await embeddings.embedQuery(text);
        return result;
    }
    private computePercentileRanks(sentenceWindows: sentenceWindow[]): void {
        // Collect distances with index references
        const distances: { index: number; value: number }[] = [];

        for (let i = 0; i < sentenceWindows.length - 1; i++) {
            const d = sentenceWindows[i].distanceNext;
            if (typeof d === "number") {
            distances.push({ index: i, value: d });
            }
        }

        if (distances.length === 0) return;

        // Sort distances ascending
        const sorted = [...distances].sort((a, b) => a.value - b.value);
        const n = sorted.length;

        // Assign percentile rank per distance
        for (let rank = 0; rank < n; rank++) {
            const { index } = sorted[rank];
            const percentile = (rank / (n - 1)) * 100; // 0..100 scale

            sentenceWindows[index].distancePercentile = percentile;
        }
    }
}




// To do:
// Figure out the percentile chunking  from GPT
// Add min and max token count
