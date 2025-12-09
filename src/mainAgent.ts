import { StateGraph, START, END, Annotation, messagesStateReducer, MemorySaver } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { Question } from "./types.js";
import { system_user_prompt } from "./ai_utils/langchainHelpers.js";


const model = new ChatOpenAI({
    model: 'o3'
})

const State = Annotation.Root({
    task: Annotation<string>,
    keptInfo: Annotation<string>,
    questionId: Annotation<number>,
    question: Annotation<string>,
    startInfo: Annotation<string>
})

async function extractInfo(state: typeof State.State){

    const defineOneShotItems = system_user_prompt(
        ``,
        ``
    )
    const mainPrompt = system_user_prompt(
        ``,
        ``
    )
    return{}
}

async function taskDefinitionNode(){
    system_user_prompt(
        ``,
        ``
    )
    return{}
}

// Define one shot prompt items from question, search through conversation for that items, to store it in state.  
//
// Define task for each question. 
//
// Define tools for task completition
//
// Create tools for Agent 
//
// Prepare Vector DB for Agent 
//
// Define important informaiton after each task completition

export async function invokeMainAgent(questions: Question[]) {
    
    console.log(questions)
}


