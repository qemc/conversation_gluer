import { StateGraph, MessagesAnnotation, START, END, Annotation, messagesStateReducer, MemorySaver, interrupt } from "@langchain/langgraph";
import { ChatOpenAI, messageToOpenAIRole } from "@langchain/openai";
import { Conversation, Data } from "./types.js";
import { BaseMessage } from "@langchain/core/messages";
import * as readline from "node:readline/promises"
import { stdin, stdout } from "node:process";
import { make_router, system_user_prompt } from "./ai_utils/langchainHelpers.js";
import { JsonOutputParser } from "@langchain/core/output_parsers";
import { filterOutItems } from "./utils/utils.js";

const model = new ChatOpenAI({
    model: 'gpt-5-nano'
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
            Jesteś zaawansowanym algorytmem lingwistycznym specjalizującym się w rekonstrukcji uszkodzonych logów rozmów. Twoim zadaniem jest odtworzenie poprawnej kolejności dialogu na podstawie fragmentów.

            Otrzymasz następujące dane wejściowe:
            1. START: Pierwsza wypowiedź (lub wypowiedzi) rozpoczynająca rozmowę.
            2. END: Ostatnia wypowiedź (lub wypowiedzi) kończąca rozmowę.
            3. LENGTH: Całkowita wymagana liczba wypowiedzi w odtworzonej rozmowie (wliczając START i END).
            4. POOL: Zbiór dostępnych wypowiedzi, z których musisz wybrać te pasujące, aby uzupełnić lukę między START a END.
            5. Zbór wyrazeń zawartych w POOL jest w kolejności losowej. 

            ZASADY:
            - Logika konwersacji: Każde kolejne zdanie musi logicznie wynikać z poprzedniego (pytanie -> odpowiedź, akcja -> reakcja).
            - Spójność: Zachowaj ciągłość wątków (np. jeśli rozmowa dotyczy hasła API, nie wstawiaj zdań o zakupach, chyba że pasują do kontekstu).
            - Długość: Wynikowa tablica MUSI mieć dokładnie długość równą parametrowi LENGTH.
            - Unikalność: Nie używaj tego samego zdania dwukrotnie, chyba że wynika to z logiki (ale w tym zadaniu zdania są unikalne).
            - Format wyjścia: Zwróć TYLKO I WYŁĄCZNIE surową tablicę JSON (array of strings). Nie dodawaj żadnych znaczników markdown (\`\`\`json), komentarzy ani wyjaśnień.

            Twoim celem jest zwrócenie tablicy: ["Wypowiedź1", "Wypowiedź2", ..., "WypowiedźN"].
            Pierwsze elementy to treść START, ostatnie to treść END, a środek to idealnie dobrane zdania z POOL.

        `,
        `
            Zrekonstruuj rozmowę na podstawie poniższych danych:

            START: {first_sentence}
            END: {last_sentence}
            LENGTH: {length}

            POOL (Wybierz pasujące zdania z tej listy, aby połączyć START i END):
            {parts}

            Pamiętaj:
            1. Rozmowa musi mieć sens logiczny.
            2. Musi składać się łącznie z dokładnie {length} wypowiedzi.
            3. Wynik to tylko tablica JSON.
        `
    )

    const newChatTest = make_router(model, 'result', prompt)

    const result = await newChatTest.invoke({
        first_sentence: conversationToProcess.start,
        last_sentence: conversationToProcess.end,
        length: conversationToProcess.length,
        parts: statementsList
    })

    console.dir(result, { depth: null, colors: true });

    return{
        convId: state.convId + 1,
        aiResponses: [result.api_response]
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
        const newStatements = filterOutItems(statements, cleanList)
        
        return{
            statementsList: newStatements
        }
    
    }
    catch(err){
        console.log(err)
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
    conversations.sort((a, b) => a.length - b.length);

    let parts_ = data.parts
    let parts = parts_.map((key) => key.trim())

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
        .addNode('parser', parserNode)
        .addEdge('prompt','human')
        .addConditionalEdges(
            'human',
            routerAfterHuman,
            {
                finish: 'parser',
                retry: '__end__'
            }
        )
        .addEdge('__start__', 'prompt')
        .addEdge('parser', 'prompt')

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
