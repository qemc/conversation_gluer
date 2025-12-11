import { promises as fs } from "fs";
import * as path from "path";
import { SemanticChunker, Chunk } from "./semantic_chunking_ts/chunking.js";
import { QdrantClient } from "@qdrant/js-client-rest";


const FACTS_PATH = process.env['FACTS_PATH'] as string;


async function handleFactsLoad(): Promise<string[]>{
    try {
        const entries = await fs.readdir(FACTS_PATH, {withFileTypes: true})
        const facts:string[] = []

        for(const entry of entries){
            if(entry.isFile()){
                const fullPath = path.join(FACTS_PATH, entry.name)
                const content = await fs.readFile(fullPath, 'utf-8')
                facts.push(content)
            }
        }
        return facts
    } catch (error) {
        return []
    }
}

export async function handleFactsChunking(){
    const Chunker = new SemanticChunker(
        1,
        'text-embedding-3-small'
    )
    const facts:string[] = await handleFactsLoad()
    let chunks:Chunk[] = []

    for(let i = 0; i < facts.length; i++){
        
        const chunkedFact = await Chunker.chunkPercentile(
            facts[i],
            90,
            {factId: i}
        )
        chunks = [...chunks, ...chunkedFact]
    }
    console.dir(chunks, {depth: null})
    return chunks
}


const client = new QdrantClient({
    url: process.env.QDRANT_URL,
    apiKey: process.env.QDRANT_API_KEY,
});


export async function uploadPreCalculatedChunks(collectionName: string) {

    try {
        const chunks: Chunk[] = await handleFactsChunking();

        console.log(`🔹 Preparing to upload ${chunks.length} chunks...`);

            const points = chunks.map((chunk) => ({
            id: chunk.chunkId,         
            vector: chunk.chunkEmbeddings, 
            payload: {
                text: chunk.chunkText, 
                ...chunk.chunkMetadata, 
                position: chunk.chunkPositionInText,
                token_count: chunk.chunkTokens
            }
        }));

        const result = await client.upsert(collectionName, {
            wait: true,
            points: points
        });

        console.log("✅ Upload status:", result.status);
    } catch (error) {
        console.error("❌ Upload failed:", error);
    }
}