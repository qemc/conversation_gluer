import { QdrantClient } from "@qdrant/js-client-rest";
import dotenv from "dotenv";
dotenv.config();

export class QdrantWrapper {
  private client: QdrantClient;

  constructor() {
    const url = process.env.QDRANT_URL;
    const apiKey = process.env.QDRANT_API_KEY;

    if (!url || !apiKey) throw new Error("Missing Qdrant credentials in .env");

    this.client = new QdrantClient({ url, apiKey });
  }

  // Create a collection if it doesn't exist
  async initCollection(name: string, vectorSize: number = 1536): Promise<void> {

    const existing = await this.client.getCollections();
    const exists = existing.collections.some((c) => c.name === name);

    if (exists) {
      console.log(`✅ Collection '${name}' already exists`);
      return;
    }

    console.log(`🆕 Creating collection '${name}'...`);

    await this.client.createCollection(name, {
      vectors: { size: vectorSize, distance: "Cosine" },
    });
    console.log(`✅ Collection '${name}' created`);
  }

  // Insert or update a single vector with metadata 
  async upsertVector(
    collection: string,
    id: string | number,
    vector: number[],
    payload: Record<string, any> = {}
  ): Promise<void> {

    await this.client.upsert(collection, {
      points: [
        {
          id,
          vector,
          payload,
        },
      ],
    });
    console.log(`🟢 Inserted vector ${id} into '${collection}'`);
  }

  // Batch insert
  async upsertBatch(
    collection: string,
    items: { id: string | number; vector: number[]; payload?: Record<string, any> }[]
  ): Promise<void> {

    await this.client.upsert(collection, { points: items });
    console.log(`🟢 Inserted batch of ${items.length} vectors into '${collection}'`);
  }

  // Search for most similar vectors
  async search(
    collection: string,
    queryVector: number[],
    limit: number = 3
  ): Promise<any[]> {

    const result = await this.client.search(collection, {
      vector: queryVector,
      limit,
    });
    return result;
  }

  // Delete a collection 
  async deleteCollection(name: string): Promise<void> {
    
    await this.client.deleteCollection(name);
    console.log(`🗑️ Deleted collection '${name}'`);
  }
}