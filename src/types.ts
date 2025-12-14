import { BaseMessage } from "@langchain/core/messages"

export interface Conversation{
    start:string,
    end:string,
    length:number, 
    id: number
}

export interface Data{
    conversations: Conversation[],
    parts: string[]
}
export interface Conversation_embeddings{
    conversation: Conversation,
    start_embedding: number[],
    end_embedding: number[]
}

export interface Part_embeddings{
    text: string,
    embeddings: number[]
}

export interface Part_cosine_similarity{
    text:string,
    cosine_sim: number
}

export type ChainIO = Record<string, any>;

export type ConvDetails = {
    text: string,
    details: Record<string,string[]>
}


export type JsonFileConv = {
    convId: number,
    conversation: string[]
}

export type saveCache = {
    currConvId: number,
    usedParts: string[]
}

// ID is needed to send answer to API
export type Question = {
    questionId: number,
    question: string
}

export type Answer = Question & {
    answer: string
}

export type ValidationPayload = {
    task: 'phone',
    apikey: string,
    answer: Record<string,string>
}


