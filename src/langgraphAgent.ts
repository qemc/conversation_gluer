import { StateGraph, MessagesAnnotation, START, END, Annotation, messagesStateReducer, MemorySaver, interrupt } from "@langchain/langgraph";
import { ChatOpenAI, messageToOpenAIRole } from "@langchain/openai";
import { Conversation, Data } from "./types.js";
import { BaseMessage } from "@langchain/core/messages";
import * as readline from "node:readline/promises"
import { stdin, stdout } from "node:process";
import { make_router, system_user_prompt } from "./ai_utils/langchainHelpers.js";






const model = new ChatOpenAI({
    model: 'gpt-5-nano'
})


const State = Annotation.Root({

    statementsList: Annotation<string[]>({
        reducer: (current, update) => {
            return current.filter(item => !update.includes(item)); 
        },
        default: () => [],
    }),
    conversations: Annotation<Conversation[]>,

    aiResponses: Annotation<BaseMessage[]>({
        reducer: messagesStateReducer,
        default: () => []
    }),
    convId: Annotation<number>,
    isApproved: Annotation<boolean>
})

async function promptNode(state: typeof State.State) {

    const statementsList = state.statementsList;
    const convId = state.convId;

    const conversationToProcess = state.conversations[convId];

    console.log(state.conversations.length)

    console.log('conversation to proceed with:')
    console.log(`start: ${conversationToProcess.start}`)
    console.log(`end: ${conversationToProcess.end}`)
    console.log(`ID: ${convId}`)

    const prompt = system_user_prompt(
        `
        You are a {book} fan. 
        `,
        `
        How {number_of_part} part of {book} has finished?
        `
    )


    const newChatTest = make_router(model, 'result', prompt)
    const result = await newChatTest.invoke({
        book: 'Harry Potter',
        number_of_part: '5th'
    })

    console.dir(result, { depth: null, colors: true });

    return{
        convId: state.convId + 1
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

// prompt -> hitl (AI in the future) -> parser + remove good looking items from the list 
//                                   -> repeat + include mistaken sequence

function humanApprove(state: typeof State.State){
    console.log("hey, I'm a human node")
    return{}
}

export async function invokeAgent(data: Data){

    const conversations = data.conversations
    const parts = data.parts

    // initial state definition
    const initialState: typeof State.State = {
        statementsList: parts,
        conversations: conversations,
        aiResponses: [], 
        convId: 0,
        isApproved: false
    }

    //config definition
    const config = { configurable: { thread_id: "cli-session-1" } };

    // checkpointer definition
    const checkpointer = new MemorySaver();

    // graph definition
    const workflow = new StateGraph(State)
        .addNode('prompt', promptNode)
        .addNode('human', humanApprove)
        .addEdge('prompt','human')
        .addConditionalEdges(
            'human',
            routerAfterHuman,
            {
                finish: "__end__",
                retry: 'prompt'
            }
        )
        .addEdge('human', '__end__')
        .addEdge('__start__', 'prompt');
    
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


        // console.log(lastResponse)
        const answer = await rl.question("Approve? (y/n): ")

        let approval = false

        if(answer === 'y'){
            approval = true
        }

        await app.updateState(config, {
            isApproved: approval
        });

        await app.invoke(null, config);
    }

    rl.close();

}
