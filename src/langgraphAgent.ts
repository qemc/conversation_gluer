import { StateGraph, START, END, Annotation, messagesStateReducer, MemorySaver } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { Conversation, Data, JsonFileConv } from "./types.js";
import { BaseMessage } from "@langchain/core/messages";
import * as readline from "node:readline/promises"
import {stdin, stdout } from "node:process";
import { make_router, system_user_prompt } from "./ai_utils/langchainHelpers.js";
import { JsonOutputParser } from "@langchain/core/output_parsers";
import { filterOutItems, saveJsonToFile } from "./utils/utils.js";

const model = new ChatOpenAI({
    model: 'o3'
})
const parser = new JsonOutputParser();

const State = Annotation.Root({

    statementsList: Annotation<string[]>,
    conversations: Annotation<Conversation[]>,
    aiResponses: Annotation<BaseMessage[]>({
        reducer: messagesStateReducer,
        default: () => []
    }),
    convId: Annotation<number>,
    isApproved: Annotation<boolean>,
    previousResponses: Annotation<string[]>,
    parsingError: Annotation<boolean>
})

async function promptNode(state: typeof State.State) {

    const statementsList = state.statementsList;
    const convId = state.convId;

    const conversationToProcess = state.conversations[convId];

    console.log('conversation to proceed with:')
    console.log(`start: ${conversationToProcess.start}`)
    console.log(`end: ${conversationToProcess.end}`)
    console.log(`ID: ${convId}`)

    let previous = 'NONE';
    if (!state.isApproved){
        previous = state.previousResponses.join('\n----------\n')
    }

    let prompt = system_user_prompt(
    `
        Jesteś ekspertem lingwistycznym i detektywem danych. Twoim zadaniem jest rekonstrukcja konkretnej rozmowy na podstawie jej początku, końca oraz rozsypanego zbioru zdań.

        Otrzymasz:
        1. START: Początek rozmowy.
        2. END: Koniec rozmowy.
        3. LENGTH: Dokładna długość docelowej tablicy (wliczając START i END).
        4. POOL: Zbiór zdań. UWAGA: Ten zbiór zawiera zdania z TEJ rozmowy oraz SZUM informacyjny (zdania z zupełnie innych rozmów). Musisz odfiltrować te, które nie pasują do kontekstu.
        5. PREVIOUS ANSWER: Wynik twojej poprzedniej próby (lub NONE).

        ZASADY KRYTYCZNE:
        - Logika Dialogu: Zwracaj uwagę na osoby (Speaker A -> Speaker B). Jeśli A zadaje pytanie, B odpowiada.
        - Spójność Tematyczna: Jeśli START dotyczy "włamania", a w POOL są zdania o "zakupach", odrzuć zdania o zakupach. Środek musi łączyć START z END.
        - Długość: Tablica wyjściowa musi mieć DOKŁADNIE długość {length}.
        - Format: Zwróć CZYSTĄ tablicę JSON (array of strings). Żadnych markdownów, żadnego 'json', żadnych komentarzy.
        
        OBSŁUGA BŁĘDÓW (PREVIOUS ANSWER):
        - Jeśli pole PREVIOUS ANSWER zawiera tablicę, oznacza to, że Twoja poprzednia próba była BŁĘDNA.
        - Błąd mógł polegać na: wybraniu zdań z innej rozmowy (zły kontekst) LUB złej kolejności zdań.
        - ZABRANIA SIĘ zwracania identycznej tablicy jak w PREVIOUS ANSWER. Musisz znaleźć inną kombinację lub kolejność.

        Cel: Tablica ["Start...", "Środek1...", "Środek2...", "End..."] tworząca spójną historię.
    `,
    `
        Zrekonstruuj rozmowę.

        START: {first_sentence}
        END: {last_sentence}
        LENGTH: {length}
        
        PREVIOUS ANSWER (To rozwiązanie było błędne, popraw je): 
        {previous}

        POOL (Wybierz pasujące zdania, odrzuć niepasujące do kontekstu):
        {parts}

        Wygeneruj tylko tablicę JSON:
    `
    )
    const newChatTest = make_router(model, 'result', prompt)

    const result = await newChatTest.invoke({
        first_sentence: conversationToProcess.start,
        last_sentence: conversationToProcess.end,
        length: conversationToProcess.length,
        parts: statementsList, 
        previous: previous
    })

    console.dir(result, { depth: null, colors: true });
    
    const currentResponse = result.api_response.content as string
    const previousResponses = state.previousResponses
    previousResponses.push(currentResponse)

    return{
        aiResponses: [result.api_response],
        isApproved: false,
        parsingError: false,
        previousResponses: previousResponses
    }
}

async function parserNode(state: typeof State.State){

    const lastMessage = state.aiResponses[state.aiResponses.length - 1];
    const content = lastMessage.content as string;


    try{
        const parsedData = await parser.parse(content);
        if (!Array.isArray(parsedData)) {
        throw new Error("Output was not an array");
        }

        const cleanList = parsedData as string[];
        const statements = state.statementsList
        const {filteredList, removedCount} = filterOutItems(statements, cleanList)

        console.log(`Number of items should be removed: ${state.conversations[state.convId].length - 2}`)
        console.log(`Number of items items has been removed: ${removedCount}`)
        
        console.log(`Filtered list:\n${filteredList}`)
        if (removedCount !== state.conversations[state.convId].length - 2) {
            throw new Error("Removed items count does not equal length of needed items");
        }

        return{
            statementsList: filteredList
        }
    }
    catch(err){
        console.log(err)
        return {
            parsingError: true
        }
    }
}

// prompt -> hitl (AI in the future) -> parser + remove good looking items from the list 
//                                   -> repeat + include mistaken sequence

function humanApprove(state: typeof State.State){
    return{}
}

async function saverNode(state: typeof State.State){

    const lastMessage = state.aiResponses[state.aiResponses.length - 1];
    const content = lastMessage.content as string;
    
    const parsedData = await parser.parse(content);
    
    const jsonContent = {
        convId: state.convId,
        conversation: parsedData
    } as JsonFileConv

    const CONV_PATH = process.env['CONV_PATH'] as string;
    saveJsonToFile(`conv${state.convId}.json`,CONV_PATH, jsonContent)

    return {
        convId: state.convId + 1,
        previousResponses: []
    }
}
function routerAfterHuman(state: typeof State.State){

    if(state.isApproved){
        return 'finish'
    }
    else{
        return 'retry'
    }
}

function finalRouter(state: typeof State.State){
    if(state.convId === state.conversations.length){
        return 'end'
    }
    else{
        return 'continue'
    }
}

function parsingErrorHandlerRouter(state: typeof State.State){
    if(state.parsingError){
        return 'tryAgain'
    }
    else{

        return 'goodToGo'
    }
}

export async function invokeAgent(data: Data){

    const conversations = data.conversations
    conversations.sort((a, b) => a.length - b.length);

    let parts_ = data.parts
    let parts = parts_.map((key) => key.trim())

    // initial state definition
    const initialState: typeof State.State = {
        statementsList: parts,
        conversations: conversations,
        aiResponses: [], 
        convId: 0,
        isApproved: true, 
        previousResponses: [],
        parsingError: false
    }

    //config definition
    const config = { configurable: { thread_id: "cli-session-1" } };

    // checkpointer definition
    const checkpointer = new MemorySaver();

    // graph definition
    const workflow = new StateGraph(State)
        .addNode('prompt', promptNode)
        .addNode('human', humanApprove)
        .addNode('parser', parserNode)
        .addNode('saver', saverNode)
        .addEdge('prompt','human')
        .addConditionalEdges(
            'human',
            routerAfterHuman,
            {
                finish: 'parser',
                retry: 'prompt'
            }
        )
        .addEdge(START, 'prompt')
        .addConditionalEdges(
            'parser',
            parsingErrorHandlerRouter,
            {
                tryAgain: 'prompt',
                goodToGo: 'saver'
            }
        )
        .addConditionalEdges(
            'saver',
            finalRouter,
            {
                end: END,
                continue: 'prompt'
            }
        )

    // first workflow compilation
    const app = workflow.compile({
        checkpointer, 
        interruptBefore: ['human']
    });

    // app invokation
    await app.invoke(initialState, config)
    // after above command Agent starts his job

    // HITL implementation
    const rl = readline.createInterface({ input: stdin, output: stdout });

    while(true){

        const snapshot = await app.getState(config)
        
        const values = snapshot.values;
        const convId = values.convId;

        if(snapshot.next.length === 0) break;
        if(convId === values.conversations.length) break;

        const answer = await rl.question("Approve? (y/n): ")

        let approval = false

        if(answer === 'y'){
            approval = true
        }
        if(answer === 'q'){
            break
        }

        await app.updateState(config, {
            isApproved: approval
        });

        await app.invoke(null, config);
    }
    rl.close();
}


// TO DO:
// Implement saverNode to the graph structure - done
// Test if saverNode works properly - done
// Think of implementing initial state from file (so the successfully processed conversations won't be processed multiple times) 
// implement human feedback, allow user enter manual suggestions
