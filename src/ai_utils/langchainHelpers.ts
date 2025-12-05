import { HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { loadApiKey } from "./apiSetUp.js";
import { RunnableLambda } from "@langchain/core/runnables";
import { OpenAIEmbeddings } from "@langchain/openai";
import { ChainIO } from "../types.js";
import { QdrantClient } from "@qdrant/js-client-rest";
import readline from "readline";
import { BaseMessage } from "@langchain/core/messages";
import { string } from "zod/v3";

export function system_user_prompt(
    system_prompt: string,
    user_prompt: string
): ChatPromptTemplate{

    if (!system_prompt) throw new Error('Empty system prompt in \'system_user_prompt\'');
    if (!user_prompt) throw new Error('Empty system prompt in \'system_user_prompt\'');

    const user_template: HumanMessagePromptTemplate = HumanMessagePromptTemplate.fromTemplate(user_prompt)
    const system_template: SystemMessagePromptTemplate = SystemMessagePromptTemplate.fromTemplate(system_prompt)

    const final_prompt: ChatPromptTemplate = ChatPromptTemplate.fromMessages([
        system_template,
        user_template
    ])
    return final_prompt
}


export function LangChainOpenAImodel(
    model_: string = 'gpt-5-nano-2025-08-07'
): ChatOpenAI {

    const apiKey = loadApiKey('openai');
    if(!apiKey) throw new Error('Error for apiKey in LangChainOpenAI')

    const model = new ChatOpenAI({
        model: model_,
        openAIApiKey: apiKey
    })
    return model
}

// this function declares the runnable that will be responsible for launching the query

export function make_router(

    model: ChatOpenAI, // model which will process the question
    output_key: string, // the key under which the llm answer will be saved to the output
    prompt: ChatPromptTemplate // The prompt that will be used for that call
){
    return RunnableLambda.from<ChainIO, ChainIO> (async (input) => {

        const promptValue = await prompt.invoke(input)
        const promptMessages = promptValue.toChatMessages()
        const llm_response = await model.invoke(promptValue)

        let rawText = "";

        if (BaseMessage.isInstance(llm_response)){
          const content = llm_response.content;

          if (typeof content === "string") {
                rawText = content.trim();
          }
        }
        else{
          rawText = String(llm_response).trim()
        }
        
        const hasToolCalls = (llm_response?.tool_calls?.length ?? 0) > 0;

        if (!rawText && !hasToolCalls) {
            throw new Error(`Router error: Empty response.`);
        }

        let tokensUsed = 0;
        // Standard LangChain JS usage location
        if (llm_response?.usage_metadata?.total_tokens) {
            tokensUsed = llm_response.usage_metadata.total_tokens;
        }

        const prevTokens = Number(input.tokens_so_far) || 0;

        return {
            ...input,
            [output_key]: rawText,
            api_response: llm_response, 
            tokens_so_far: tokensUsed + prevTokens,
            prompt_messages: promptMessages
        };
    });
};



export async function get_embedding(text:string): Promise<number[]>{

    if (!text || text.trim() === "") {
    throw new Error("Empty text passed to getEmbedding()");
        }
        const embeddings = new OpenAIEmbeddings({
            model: "text-embedding-3-small", 
            apiKey: process.env.OPENAI_API_KEY,
        });

        const result = await embeddings.embedQuery(text);

        return result;
}

// createQdrantCollection(qdrant_client, 'zad51_coll', 1536, 'Cosine')
export async function createQdrantCollection(

  client: QdrantClient,
  collectionName: string,
  vectorSize: number = 1536,
  distance: "Cosine" | "Dot" | "Euclid" = "Cosine"

) {
  try {

    console.log(`🆕 Creating collection '${collectionName}'...`);

    await client.createCollection(collectionName, {
      vectors: {
        size: vectorSize,
        distance,
      },
    });

    console.log(`✅ Collection '${collectionName}' created successfully.`);
  } catch (err: any) {

    if (err.message.includes("already exists")) {

      console.log(`⚠️ Collection '${collectionName}' already exists.`);
    } else {

      console.error(`❌ Failed to create collection '${collectionName}':`, err);
      throw err;
    }
  }
}

export const qdrant_default_client = new QdrantClient({
  url: process.env.QDRANT_URL!,
  apiKey: process.env.QDRANT_API_KEY!,
});



async function askUser(question: string): Promise<boolean> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  return new Promise((resolve) => {
    rl.question(`${question} (y/n): `, (answer) => {
      rl.close();
      resolve(answer.toLowerCase().startsWith("y"));
    });
  });
}

export function make_human_router(
  displayKey: string = "to_accept",
  outputKey: string = "human_approved"
) {
  return RunnableLambda.from(async (input: Record<string, any>) => {
    const displayValue = input[displayKey] ?? JSON.stringify(input, null, 2);
    console.log("🤖 Model proposed:", displayValue);

    const approve = await askUser("Approve? (y/n)");

    return {
      ...input,
      [outputKey]: approve,
    };
  });
}


