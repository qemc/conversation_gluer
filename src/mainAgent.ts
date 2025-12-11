import { StateGraph, START, END, Annotation, messagesStateReducer, MemorySaver } from "@langchain/langgraph";
import { ChatOpenAI, convertReasoningSummaryToResponsesReasoningItem } from "@langchain/openai";
import { Question } from "./types.js";
import { make_router, system_user_prompt } from "./ai_utils/langchainHelpers.js";
import { promises as fs } from "fs";
import * as path from "path";
import { Conversation, Data, JsonFileConv, saveCache } from "./types.js";
import { z } from "zod"
import { tool } from "@langchain/core/tools";

const CONV_PATH = process.env['CONV_PATH'] as string;

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
        console.log(query)
    },
    {
        name: 'research_query',
        description: 'Call this tool when you want to add more data and what currently is collected, is not sufficient to answer the question or follow the steps defined.',
        schema: z.object({
            query: z.string().describe('The specific information needed to be extracted from Vector Database')
        })
    }
)

const proccedTool = tool(
    async({summary}) => {
        console.log(summary)
    },
    {
        name: 'proceed_further_tool',
        description: 'Call this tool when you believe that all data needed to answer the question is collected and we are good to go further. Also call this tool, when you receive communication, that there is no more data related to this question.',
        schema: z.object({
            summary: z.string().describe('brief summary of why we do have all data necessary to answer the question')
        })
    }
)


const State = Annotation.Root({
    answerPlan: Annotation<string>,
    keptInfo: Annotation<string>,
    questionId: Annotation<number>,
    startInfo: Annotation<string>,
    allQuestions: Annotation<Question[]>,
    conversations: Annotation<JsonFileConv[]>,
    currentContext: Annotation<string>
})


async function answerPlanNode(state: typeof State.State){
    
    const prompt = system_user_prompt(
    `
    **Rola:**
    Jesteś Architektem Rozwiązywania Problemów (Problem Solving Architect). Twoim zadaniem jest przeanalizowanie pytania użytkownika i stworzenie precyzyjnego planu działania, który zostanie wykonany przez Agenta wyposażonego w konkretne narzędzia.

    **Dostępne Narzędzia Agenta:**
    Twój wykonawca (Agent) posiada następujące możliwości w etapie zbierania wiedzy:
    1. \`research_query\`: Narzędzie do przeszukiwania bazy wektorowej (wiedzy/dokumentacji). Używane do znajdowania faktów, haseł, URL-i, imion itp.
    2. \`proceed_further_tool\`: Narzędzie sygnalizujące, że zgromadzono już komplet informacji i można przejść do właściwej odpowiedzi lub akcji.

    **Zadanie:**
    Dla każdego pytania wygeneruj wyjście składające się z dwóch sekcji:

    1. **Oczekiwany format:** Precyzyjny opis tego, jak ma wyglądać ostateczna odpowiedź (np. "Czysty ciąg znaków URL", "Obiekt JSON").
    2. **Kroki do podjęcia:** Lista ponumerowanych, atomowych czynności.

    **Zasady tworzenia kroków:**
    - **Mapowanie na Narzędzia:** Każdy krok wymagający zdobycia nowej informacji musi implikować użycie narzędzia \`research_query\`. Sformułuj to jasno, np. "Użyj research_query, aby znaleźć hasło...".
    - **Rozbijaj zależności:** Nie możesz "wysłać hasła", jeśli go nie masz. Najpierw zaplanuj krok znalezienia hasła (research), potem krok znalezienia adresu (research), a na końcu krok wykonania akcji.
    - **Logika i Weryfikacja:** Jeśli pytanie jest podchwytliwe (np. "kto nie kłamał"), dodaj krok analizy logicznej zebranych danych przed ostateczną odpowiedzią.

    **Format Wyjściowy:**
    Zwróć odpowiedź w ściśle określonym formacie:

    Oczekiwany format:
    [Opis formatu]

    Kroki do podjęcia:
    1. [Krok pierwszy - zazwyczaj research]
    2. [Krok drugi - research/analiza]
    ...
    `,
    `
    Pytanie do analizy:
    {question}
    `
)
       
    const chain = make_router(model_51, 'result', prompt)
    const result = await chain.invoke({
        question: state.allQuestions[state.questionId].question
    })

    console.log(result.api_response.content)

    return{
        answerPlan: result.api_response.content
    }
}

async function dataGatheringNode(state: typeof State.State){

    const tools = [researchTool, proccedTool]
    const llm_with_tools = model_51.bindTools(tools)


    const prompt = system_user_prompt(
        `
        **Rola:**
        Jesteś Audytorem Kompletności Danych (Data Completeness Auditor). Nie odpowiadasz bezpośrednio na pytania. Twoim jedynym zadaniem jest sterowanie przepływem (routing) poprzez wybór odpowiedniego narzędzia.

        **Zadanie:**
        Przeanalizuj dostarczone \`<pytanie>\`,  \`<zebrany_kontekst>\` oraz \`<plan>\` i podejmij jedną z dwóch decyzji:

        **SCENARIUSZ 1: BRAKUJE DANYCH -> Użyj \`research_query\`**
        Wybierz to narzędzie, jeśli:
        - W kontekście brakuje kluczowych faktów (np. imion, dat, nazw endpointów).
        - Pytanie wymaga akcji (np. "Wyślij hasło"), a Ty nie masz parametrów (nie znasz hasła lub URL).
        - Potrzebujesz doprecyzować informacje.
        
        *Zasada dla Researchu:* Zapytanie w \`query\` musi być precyzyjne i celować w brakujący element (np. "adres endpointu API", "hasło użytkownika X").

        **SCENARIUSZ 2: MAM KOMPLET DANYCH -> Użyj \`proceed_further_tool\`**
        Wybierz to narzędzie, jeśli:
        - Masz wszystkie fakty niezbędne do udzielenia odpowiedzi.
        - Masz wszystkie parametry niezbędne do wykonania zlecenia (np. masz URL i Hasło, żeby wykonać request).
        - Otrzymałeś informację, że w bazie nie ma więcej danych na ten temat (unikamy pętli).

        **WAŻNE OGRANICZENIA:**
        - Nie generuj zwykłego tekstu. MUSISZ wywołać jedno z narzędzi.
        - Bądź surowy. Jeśli masz wątpliwości, czy dane są kompletne, wybierz \`research_query\`.
        `,
        `
        <pytanie>
        {question}
        </pytanie>

        <plan>
        {plan}
        </plan>

        <zebrany_kontekst>
        {context}
        </zebrany_kontekst>
        `
    )

    const chain = make_router(llm_with_tools, 'result', prompt)
    const result = chain.invoke({
        questions: state.allQuestions[state.questionId],
        plan: state.answerPlan,
        context: state.currentContext
    })

    return{}
}

// To do in the middle:
// finish gathering node - in progress
// proceed with Qdrant implementation 

async function queryQdrantNode(){
    
    const prompt = system_user_prompt(
        `
        This node queries Vector DB based on the question, expected answer, and summary (?) TBD
        `,
        `
        Am I able to answer this questions based on conversation? 
        
        `
    )
    return{}
}



async function decideToolNode(){
    
    const prompt = system_user_prompt(
        `
        Based on the gathered knowledge, question and excpected answer, choose tool that needs to be invoked. Provide all necessay input data.  
        `,
        `
        Am I able to answer this questions based on conversation? 
        
        `
    )

    return{}
}


async function playApiAgentNode(){
    
    const prompt = system_user_prompt(
        `
        Placeholder for API invokation Agent
        Returns answer to be validated
        `,
        `        
        `
    )

    return{}
}


async function researchNode(){
    
    const prompt = system_user_prompt(
        `
        Research Node to be here implemented, 
        Returns answer to be validated
        `,
        `        
        `
    )

    return{}
}


async function validateAnswerNode(){
    
    const prompt = system_user_prompt(
        `
        Validates the answer by calling validating API.
        If the answer is correct, Agent proceeds further, 
        If the answer is incorrect, Agent goes back to the beginning or to the answer assessment node. TBD
        `,
        `        
        `
    )

    return{}
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
        }).join('\n\n')
    }
    const config = { configurable: { thread_id: "cli-session-1" } };
    
    // checkpointer definition
    const checkpointer = new MemorySaver();

    // graph definition
    const workflow = new StateGraph(State)
        .addNode('extract', answerPlanNode)
        .addEdge(START, 'extract')
        .addEdge('extract',END)

    const app = workflow.compile({
        checkpointer
    });
    await app.invoke(initialState, config)


    console.log(conversations)
    console.log(questions)
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

