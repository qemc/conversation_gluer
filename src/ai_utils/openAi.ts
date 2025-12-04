import 'dotenv/config';
import OpenAI from 'openai';
import { loadApiKey } from './apiSetUp.js';
import type { ChatCompletionMessageParam } from 'openai/resources.js';
import { ChatOpenAI } from '@langchain/openai';
import { Chat } from 'openai/resources';
import { error } from 'console';

function loadOpenAIClient(): OpenAI {

    const apiKey = loadApiKey('openai');
    const openAiClient = new OpenAI({apiKey});
    return openAiClient
}

export async function askOpenAI_text(
    userPrompt: string,
    systemPrompt: string,
    model: string = 'gpt-5-nano-2025-08-07'
): Promise<string>{

    const client: OpenAI = loadOpenAIClient();
    const messages: ChatCompletionMessageParam[] = [];
    
    if (!userPrompt?.trim()) throw new Error("Empty user prompt");
    if (!systemPrompt?.trim()) throw new Error("Empty system prompt");

    messages.push({role: 'system', content: systemPrompt})
    messages.push({role: 'user', content: userPrompt})

    try {

        const result = await client.chat.completions.create({
            model: model,
            messages: messages
        });

        const reurntedMessage = result.choices[0]?.message?.content?.trim();
        if(!reurntedMessage) throw new Error('OpenAI returned no message content.');

        console.log(reurntedMessage)
        return reurntedMessage 

    } catch (error) {
        console.error('OpenAI call failed:', error)
        throw new Error(`askOpenAI_text failed at: \n\n ${systemPrompt} \n\n ${userPrompt}`)
    }
}

