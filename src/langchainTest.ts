import { HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate } from "@langchain/core/prompts";
import { LangChainOpenAImodel } from "./ai_utils/langchainHelpers.js";
import { system_user_prompt } from "./ai_utils/langchainHelpers.js";
import { make_router } from "./ai_utils/langchainHelpers.js";
import { RunnableLambda } from "@langchain/core/runnables";
import { ChainIO } from "./types.js";


export type ChainInput = Record<string, string>;
const model = LangChainOpenAImodel();



const prompt1 = system_user_prompt(
  "You are a helpful assistant who prepares study material for {audience}.",
  "Summarize the core principles of {topic} in 2-3 sentences."
);

// 🧩 Prompt 2 – Write explanatory paragraph
const prompt2 = system_user_prompt(
  "You are an educational writer specializing in {topic}.",
  "Using this summary: {summary}, write one clear explanatory paragraph aimed at {audience}."
);


// 🧩 Prompt 3 – Generate review questions
const prompt3 = system_user_prompt(
  "You are a teacher creating review questions for {audience}.",
  "Based on this paragraph: {paragraph}, generate three thoughtful questions that test comprehension."
);

const router1: RunnableLambda<ChainIO,ChainIO> = make_router(
    model,
    'summary',
    prompt1
)

const router2: RunnableLambda<ChainIO,ChainIO> = make_router(
    model,
    'paragraph',
    prompt2
)

const router3: RunnableLambda<ChainIO,ChainIO> = make_router(
    model,
    'output',
    prompt3
)

const chain = router1.pipe(router2).pipe(router3)
const result1 = await chain.invoke({topic: 'football', audience: 'young footbal players'})
const result2 = await chain.invoke({topic: 'AI', audience: 'students of politechnika poznańska'})

console.log(result1)
console.log(result2)







