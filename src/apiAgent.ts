import { StateGraph, START, END, Annotation, messagesStateReducer, MemorySaver } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { make_router, system_user_prompt } from "./ai_utils/langchainHelpers.js";
import {postPayload } from "./utils/utils.js";


const model = new ChatOpenAI({
    model: 'gpt-5-nano'
})

const State = Annotation.Root({

    context: Annotation<string>,
    password: Annotation<string>,
    endpoints: Annotation<string[]>,
    apiResponses: Annotation<any[]>

})

async function getPasswordNode(state: typeof State.State){
    const prompt = system_user_prompt(`
        **Rola**
        Jesteś sprawnym analitykiem tekstowym, którego rolą jest wyjandywanie haseł w tekście.

        **Zadanie** 
        W tekście, który znajduje się w polu <kontekst>, znajdź słowo które według kontekstu będzie hasłem do API. Uwaznie przeanalizuj załączony tekst. Hasłem będzie jedno słowo.
        `,`
        <kontekst>
        {context}
        </kontekst>
        `)

        const chain = make_router(model, 'result', prompt)
        const result = await chain.invoke({
            context: state.context
        })

        console.log(result?.result)
        if(!result?.result) throw new Error('Password was not found')

        return{
            password: result.result
        }
}

function getEndpointsNode(state: typeof State.State){

    const text = state.context;
    const rawMatches = text.match(/https?:\/\/[^\s"']+/gi) || [];

    const finalEndpointList =  [...new Set(rawMatches.map(url => url.replace(/[.,;:]+$/, '')))];

    return {
        endpoints: finalEndpointList
    }

}

async function tryEndpiointNode(state: typeof State.State){

    const payload: Record<string, string> = {password: state.password} 
    const apiResponses: any[] = []

    for (let x = 0; x < state.endpoints.length; x++){
        const endpoint = state.endpoints[x];
        const response = await postPayload<Record<string, string> >(payload, endpoint)

        apiResponses.push(JSON.stringify(response))
    }
    return{
        apiResponses: apiResponses
    }
}


export async function invokeApiAgent(data: string){

    // initial state definition
    const initialState: typeof State.State = {
        context: data,
        password: '',
        endpoints: [],
        apiResponses: []
    }

    //config definition
    const config = { configurable: { thread_id: "cli-session-1" } };
    // checkpointer definition
    const checkpointer = new MemorySaver();
    // graph definition
    const workflow = new StateGraph(State)
        .addNode('password', getPasswordNode)
        .addNode('endpoints', getEndpointsNode)
        .addNode('execute', tryEndpiointNode)
        .addEdge(START, 'password')
        .addEdge(START, 'endpoints')
        .addEdge('password', 'execute')
        .addEdge('endpoints', 'execute')
        .addEdge('execute', END)
        
    // first workflow compilation
    const app = workflow.compile({
        checkpointer 
    });
    // app invokation
    const finalState = await app.invoke(initialState, config)
    return finalState.apiResponses.join('\n------\n') as string
}
