import { StateGraph, START, END, Annotation, messagesStateReducer, MemorySaver } from "@langchain/langgraph";
import { ChatOpenAI, convertReasoningSummaryToResponsesReasoningItem } from "@langchain/openai";
import { Question } from "./types.js";
import { make_router, system_user_prompt } from "./ai_utils/langchainHelpers.js";
import { promises as fs } from "fs";
import * as path from "path";
import { Conversation, Data, JsonFileConv, saveCache } from "./types.js";
import { z } from "zod"
import { tool } from "@langchain/core/tools";
import { QdrantClient } from "@qdrant/js-client-rest";
import { get_embedding } from "./ai_utils/langchainHelpers.js";
import { AIMessage, BaseMessage } from "@langchain/core/messages";
import * as readline from "node:readline/promises"
import {stdin, stdout } from "node:process";


const CONV_PATH = process.env['CONV_PATH'] as string;
const QDRANT_COLLECTION = process.env['QDRANT_COLLECTION'] as string;

const qdrantClient = new QdrantClient({
    url: process.env.QDRANT_URL,
    apiKey: process.env.QDRANT_API_KEY,
});
const model_o3 = new ChatOpenAI({
    model: 'o3'
})
const model_51 = new ChatOpenAI({
    model: 'gpt-5.1'
})
const model_5_nano = new ChatOpenAI({
    model: 'gpt-5-nano'
})
const researchTool = tool(
    async({query}) => {
        console.log('=========RESEARCH TOOL ARG=========')
        console.log(query)
        console.log('===================================')    
    },
    {
        name: 'research_query',
        description: 'Uzyj tego narzędzia, jeśli chcesz uzyskać dostęp do faktów, na temat osób / wydarzeń / lokacji wspomnianych w konwersacjach',
        schema: z.object({
            query: z.string().describe('Naturaline powiązane informacje, z tym co jest potrzebne. Mogą być fragmenty konwersacji, poszczególne informacje. Uzyj naturalnego języka.')
        })
    }
)
const proccedTool = tool(
    async({summary}) => {
        console.log('=========PROCEED TOOL ARG=========')
        console.log(summary)
        console.log('===================================')

    },
    {
        name: 'proceed_further_tool',
        description: 'Call this tool when you believe that all data needed to answer the question is collected and we are good to go further. Also call this tool, when you receive communication, that there is no more data related to this question.',
        schema: z.object({
            summary: z.string().describe('brief summary of why we do have all data necessary to answer the question')
        })
    }
)
function formatAppendedData(currentData: string, dataToAppend:string, dataDescription: string){
    
    let joinedDataArray: string = 'Nie znaleniono nowych danych. Musisz przejść do proceed_further_tool'

    if(dataToAppend.length > 0){
        joinedDataArray = dataToAppend;
    }

    const combinedData = [currentData, joinedDataArray].join(`\n Odpowiedź z wektorowej bazy danych faktów: \n ${dataDescription}`)
    return combinedData
}

function humandNode(state: typeof State.State){
    return{}
}

const State = Annotation.Root({
    answerPlan: Annotation<string>,
    keptInfo: Annotation<string>,
    questionId: Annotation<number>,
    startInfo: Annotation<string>,
    allQuestions: Annotation<Question[]>,
    conversations: Annotation<JsonFileConv[]>,
    currentContext: Annotation<string>,
    dataGatheringResponses: Annotation<BaseMessage[]>({
        reducer: messagesStateReducer,
        default: () => []
    }),
    isApproved: Annotation<boolean>
    
})


async function answerPlanNode(state: typeof State.State){
    
    const prompt = system_user_prompt(
    `
    **Rola:**
    Jesteś Definiatorem Celu (Goal Definer).
    Twoim jedynym zadaniem jest przeczytanie pytania i napisanie **jednego, prostego zdania** opisującego, co dokładnie ma być wynikiem.

    **Zadanie:**
    Zignoruj całą otoczkę pytania ("znajdź", "powiedz mi", "czy wiesz").
    Skup się na rzeczowniku/obiekcie, który jest odpowiedzią. Opisz go precyzyjnie.

    **Przykłady:**
    Pytanie: "Jeden z rozmówców skłamał podczas rozmowy. Kto to był?"
    Odpowiedź: Imię lub nazwisko osoby, która minęła się z prawdą.

    Pytanie: "Jaki jest prawdziwy endpoint do API podany przez osobę, która NIE skłamała?"
    Odpowiedź: Dokładny adres URL endpointu API.

    Pytanie: "Jakim przezwiskiem określany jest chłopak Barbary?"
    Odpowiedź: Przezwisko chłopaka Barbary.

    Pytanie: "Co zwraca API po wysłaniu hasła?"
    Odpowiedź: Treść odpowiedzi zwróconej przez API.

    **Ograniczenia:**
    - Odpisz tylko tym jednym zdaniem. Bez cudzysłowów, bez JSON-a, bez wstępów.
    - Używaj języka polskiego.
    `,
    `
    Pytanie: {question}
    `
)
       
    const chain = make_router(model_51, 'result', prompt)

    const result = await chain.invoke({
        question: state.allQuestions[state.questionId].question
    })

    console.log('=======ANSWER PLAN NODE=======')
    console.dir(result, { depth: null, colors: true });
    console.log('==============================')

    return{
        answerPlan: result.api_response.content
    }
}

async function dataGatheringNode(state: typeof State.State){

    const prompt = system_user_prompt(
        `
        **Rola:**
        Jesteś Weryfikatorem Celu (Target Verifier). Twoim zadaniem jest sprawdzenie, czy udało się już znaleźć konkretną informację zdefiniowaną w polu \`<cel_odpowiedzi>\`.

        **Kontekst Danych (Co czytasz?):**
        Otrzymujesz \`<zebrany_kontekst>\`, który składa się z dwóch warstw:
        1. **Konwersacje:** Zapisy rozmów ludzi. PAMIĘTAJ: Ludzie kłamią, mylą się i konfabulują. To nie są fakty.
        2. **Fakty (Baza Wektorowa):** Informacje, które już pobrałeś narzędziem \`research_query\`. To jest Twoje jedyne źródło prawdy.

        **Mapa Twojej Wiedzy (Gdzie szukać, jeśli brakuje danych?):**
        Baza faktów zawiera raporty i akta dotyczące:
        1. **Ludzi:** Profile psychologiczne, historie zatrudnienia, powiązania (np. Ragowski, Bomba, Azazel).
        2. **Miejsc:** Opisy techniczne sektorów fabryki (C, D), magazynów, zabezpieczeń.
        3. **Zdarzeń:** Raporty z incydentów, ucieczek i działań ruchu oporu.

        **Zadanie:**
        Porównaj \`<cel_odpowiedzi>\` (to, co musisz znaleźć) z \`<zebrany_kontekst>\` (to, co masz).

        **SCENARIUSZ 1: CEL NIEOSIĄGNIĘTY -> Użyj \`research_query\`**
        Wybierz to narzędzie, jeśli:
        - W kontekście brakuje informacji opisanej w \`<cel_odpowiedzi>\`.
        - Masz tylko poszlaki z rozmów (np. ktoś mówi "chyba jest w magazynie"), ale nie masz twardego faktu z bazy potwierdzającego, co jest w magazynie.
        - \`<cel_odpowiedzi>\` wymaga konkretu (np. "dokładny adres URL"), a Ty masz tylko nazwę serwisu.

        *Zasada Researchu:* Wpisz w zapytanie słowa kluczowe związane z brakującym elementem celu (np. "Sektor C przeznaczenie", "Rafał Bomba choroba").

        **SCENARIUSZ 2: CEL OSIĄGNIĘTY -> Użyj \`proceed_further_tool\`**
        Wybierz to narzędzie, jeśli:
        - \`<zebrany_kontekst>\` zawiera precyzyjną odpowiedź na opis z \`<cel_odpowiedzi>\`.
        - Masz pewność, że informacja pochodzi z wiarygodnego źródła (lub zweryfikowałeś kłamstwo rozmówcy faktami).
        - Otrzymałeś informację systemową, że w bazie nie ma więcej danych.

        **WAŻNE OGRANICZENIA:**
        - Nie generuj zwykłego tekstu. MUSISZ wywołać jedno z narzędzi.
        - Skup się wyłącznie na znalezieniu tego, co opisuje \`<cel_odpowiedzi>\`. Ignoruj poboczne wątki.
        `,
        `
        <pytanie>
        {question}
        </pytanie>

        <cel_odpowiedzi>
        {plan} 
        </cel_odpowiedzi>

        <zebrany_kontekst>
        {context}
        </zebrany_kontekst>
        `
    )

    const tools = [researchTool, proccedTool]
    const llm_with_tools = model_51.bindTools(tools)

    const chain = make_router(llm_with_tools, 'result', prompt)

    const result = await chain.invoke({
        question: state.allQuestions[state.questionId],
        plan: state.answerPlan,
        context: state.currentContext
    })

    console.log('=======DATA GATHERING NODE=======')
    console.dir(result, { depth: null, colors: true });
    console.log('=================================')
    
    return{
        dataGatheringResponses: [result.api_response]    
    }
}

async function qdrantResearchNode(state: typeof State.State) {
    
    const lastDataGatheringMessage = state.dataGatheringResponses[state.dataGatheringResponses.length - 1] as AIMessage;

    if(!(lastDataGatheringMessage?.tool_calls?.length)){
        throw new Error ('No tools have been called in qdrantResearchNode') 
    }

    const currentContext = state.currentContext
    const llmQuery = lastDataGatheringMessage.tool_calls[0].args.query;

    const vector = await get_embedding(llmQuery)
    const qdrantResults = await qdrantClient.search(QDRANT_COLLECTION, {
        vector: vector,
        limit: 1,
        with_payload: true
    });

    if(qdrantResults.length === 0){
        return {
            currentContext: formatAppendedData(currentContext, '', llmQuery)
        }
    }

    const qdrantBesstMatchFactId = qdrantResults[0].payload?.factId;

    const allFactPart = await qdrantClient.scroll(QDRANT_COLLECTION, {
        filter: {
            must:[
                {
                    key: 'factId',
                    match: {
                        value: qdrantBesstMatchFactId
                    }
                }
            ]
        },
        limit: 100,
        with_payload: true
    });

    const fullText = allFactPart.points
      .sort((a: any, b: any) => (a.payload.position - b.payload.position)) // Optional: if you saved 'position'
      .map(p => p.payload?.text)
      .join("\n\n");

    const newContext = formatAppendedData(currentContext, fullText, llmQuery)

    console.log('=======QDRANT SEARCH NODE=======')
    console.dir(newContext, { depth: null, colors: true });
    console.log('================================')

    return {
        currentContext: newContext
    }
}

// To do in the middle:
// finish gathering node - done
// proceed with Qdrant implementation - done


function dataGatheringRouter(state: typeof State.State){

    const lastDataGatheringMessage = state.dataGatheringResponses[state.dataGatheringResponses.length - 1] as AIMessage

    if(!(lastDataGatheringMessage?.tool_calls?.length)){
        throw new Error('Error in data gathering router node. The Data Gathering node did not return a tool call.')
    }
    const toolName = lastDataGatheringMessage.tool_calls[0].name;

    if(toolName === 'research_query'){
        return 'research'
    }
    if(toolName === 'proceed_further_tool'){
        return 'proceed'
    }
    throw new Error ('No matching tools have been called in dataGatheringNode')
}

function afterHumanRouter(state: typeof State.State){

    if(state.isApproved){
        return 'finish'
    }
    else{
        return 'retry'
    }
}




export async function invokeMainAgent(questions: Question[]) {

    const entries = await fs.readdir(CONV_PATH, {withFileTypes: true})
    const conversations:JsonFileConv[] = []

    for(const entry of entries){
        if(entry.isFile()){
            const fullPath = path.join(CONV_PATH, entry.name)
            const content = await fs.readFile(fullPath, 'utf-8')
            const json_content = JSON.parse(content) as JsonFileConv
            conversations.push(json_content)
        }
    }

    const initialState: typeof State.State = {
        answerPlan: '',
        keptInfo: '',
        questionId: 0,
        startInfo: '',
        allQuestions: questions,
        conversations: conversations,
        currentContext: conversations.map((conv: JsonFileConv)=> {
            return `Numer rozmowy: ${conv.convId}\n ${conv.conversation.join('\n')}`
        }).join('\n\n'),
        dataGatheringResponses:[], 
        isApproved: false
    }
    const config = { configurable: { thread_id: "cli-session-1" } };
    
    // checkpointer definition
    const checkpointer = new MemorySaver();

    // graph definition
    const workflow = new StateGraph(State)
        .addNode('plan', answerPlanNode)
        .addNode('qdrant', qdrantResearchNode)
        .addNode('gather', dataGatheringNode)
        .addNode('human', humandNode)
        .addEdge(START, 'plan')
        .addEdge('plan','gather')
        .addConditionalEdges(
            'gather',
            dataGatheringRouter,
            {
               research: 'qdrant',
               proceed: END // quick win - just answer the question :)
            }
        )
        .addEdge('qdrant', 'human')
        .addConditionalEdges(
            'human',
            afterHumanRouter,
            {
                finish: END,
                retry: 'gather'
            }
        )

    const app = workflow.compile({
        checkpointer,
        interruptBefore: ['human']
    });
    await app.invoke(initialState, config)

    const rl = readline.createInterface({ input: stdin, output: stdout });

    while(true){

        const snapshot = await app.getState(config)
                

        if(snapshot.next.length === 0) break;

        const answer = await rl.question("Approve? (y/n): ")

        let approval: boolean = false

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
    console.log(questions)
    rl.close();
}


// To do:
// implement other nodes:
// answerPlan - in progress
// data collection node 
// vector db implementation + chunking
// vector db retrieval node
// tool calling node
// sub agent API caller
// research node
// answer validation node
// admin task around Agent Invocation

// To Do: 

