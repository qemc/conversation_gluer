import { StateGraph, START, END, Annotation, messagesStateReducer, MemorySaver } from "@langchain/langgraph";
import { ChatOpenAI, convertReasoningSummaryToResponsesReasoningItem } from "@langchain/openai";
import { Answer, Question } from "./types.js";
import { make_router, system_user_prompt } from "./ai_utils/langchainHelpers.js";
import { promises as fs } from "fs";
import * as path from "path";
import { Response,JsonFileConv, ValidationPayload } from "./types.js";
import { success, z } from "zod"
import { tool } from "@langchain/core/tools";
import { QdrantClient } from "@qdrant/js-client-rest";
import { get_embedding } from "./ai_utils/langchainHelpers.js";
import { AIMessage, BaseMessage } from "@langchain/core/messages";
import * as readline from "node:readline/promises"
import {stdin, stdout } from "node:process";
import { invokeApiAgent } from "./apiAgent.js";
import { postPayload, extractQuestionId } from "./utils/utils.js";
import { finished } from "node:stream";


const CONV_PATH = process.env['CONV_PATH'] as string;
const QDRANT_COLLECTION = process.env['QDRANT_COLLECTION'] as string;
const AIDEVS_API_KEY = process.env['AIDEVS_API_KEY'] as string;
const AIDEVS_REPORT_URL = process.env['AIDEVS_REPORT_URL'] as string;



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
    model: 'gpt-5.1'
})
const researchTool = tool(
    async({query}) => {
        console.log('=========RESEARCH TOOL ARG=========')
        console.log(query)
        console.log('===================================')    
    },
    {
        name: 'research_query',
        description: 'Służy do weryfikacji faktów w bazie wektorowej. Mechanizm wykorzystuje podobieństwo kosinusowe (cosine similarity), dlatego formułując zapytanie, skup się na ogólnym sensie i kontekście semantycznym poszukiwanej informacji, a nie tylko na słowach kluczowych.',
        schema: z.object({
            query: z.string().describe(
                'Szczegółowe zdanie lub pytanie w języku naturalnym opisujące poszukiwany fakt. ' +
                'Zawrzyj w nim kluczowe imiona, nazwy miejsc i kontekst. ' +
                'Nie używaj pojedynczych słów kluczowych; opisz semantycznie o co pytasz, ' +
                'aby zmaksymalizować trafność dopasowania wektorowego.'
            )
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
        description: 'Uzyj tego narzędzia, jeśli uwazasz, ze aktualnie zgromadzone dane są wystsarczające aby odpowiedzieć na podane pytanie. ',
        schema: z.object({
            summary: z.string().describe('Maksymalnie 2 zdaniowe uzasadnienie, dlaczego uwazasz, ze aktualnie zebrany kontekst jest wystarczający, aby przejść dalej')
        })
    }
)
const apiAgentTool = tool (
    async() =>{
        console.log('=========PROCEED TOOL ARG=========')
        console.log('Proceeding with ApiAgent')
        console.log('===================================')
    },
    {
        name:'api_agent_tool',
        description: 'Uzyj tego narzędzia, jeśli uwazasz, ze zdobycie informacji potrzebnych do odpowiedzenia na pytanie wymaga wyslania zapytań do endpointu API'
    }
)
function formatAppendedData(currentData: string, dataToAppend:string, dataDescription: string){
    
    let joinedDataArray: string = 'Nie znaleniono nowych danych. Musisz przejść do proceed_further_tool'

    if(dataToAppend.length > 0){
        joinedDataArray = dataToAppend;
    }

    const combinedData = [currentData, joinedDataArray].join(`\n Odpowiedź z wektorowej bazy danych faktów na zapytanie: \n ${dataDescription} \n`)
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
    allAnswers: Annotation<Answer[]>,
    conversations: Annotation<JsonFileConv[]>,
    currentContext: Annotation<string>,
    dataGatheringResponses: Annotation<BaseMessage[]>({
        reducer: messagesStateReducer,
        default: () => []
    }),
    toolChoosingResponses: Annotation<BaseMessage[]>({
        reducer: messagesStateReducer,
        default: () => []
    }),
    isApproved: Annotation<boolean>,
    validationPayload: Annotation<ValidationPayload>,
    validationResult: Annotation<Response>,
    apiResponses: Annotation<Response[]>,
    currentApiCall: Annotation<number>, // reset on new iteration
    alreadyQueriedFacts: Annotation<number[]>,
    wrongAnswers: Annotation<string>
})


async function answerPlanNode(state: typeof State.State){
    
    const prompt = system_user_prompt(
    `
    **Rola:**
    Twoją rolą jest wyznaczanie odpowiedniego formatu odpowiedzi na podane pytanie. Podchodzisz chłodno do swojego zadania. Nie dodajesz zbędnych słów oraz komunikujesz wprost format odpowiedzi.

    **Zadanie:**
    Na podstawie otrzymanego pytania oraz swojej generalnej wiedzy zwróć format, jaki powinna mieć odpowiedź na zadane pytanie, uznana za poprawną. Głęboko myśl nad poprawnym rozwiązaniem zadania. 

    **Przykłady:**
    Pytanie: "Jeden z rozmówców skłamał podczas rozmowy. Kto to był?"
    Odpowiedź: Imię lub nazwisko osoby, która skłamała.

    Pytanie: "Jaki jest prawdziwy endpoint do API podany przez osobę, która NIE skłamała?"
    Odpowiedź:  Adres URL endpointu API.

    Pytanie: "Jakim przezwiskiem określany jest chłopak Barbary?"
    Odpowiedź: Przezwisko chłopaka Barbary.

    Pytanie: "Co zwraca API po wysłaniu hasła?"
    Odpowiedź: Treść odpowiedzi zwróconej przez API.

    **Rola2:**
    Twoją rolą jest anazliza swoich poprzenich działań.

    **Zadanie2:**    
    Po wyznaczeniu poprawnego formatu odpowiedzi, twoim zadaniem jest na jego podstawie przygotować podsumowanie w 1/2 zdaniach co musi być zrobione. Twoja odpowiedź zostanie przekazana do kolejnego modułu Agenta.

    **Przykłady2:**
    Pytanie: "Co odpowiada poprawny endpoint API po wysłaniu do niego hasła w polu "password" jako JSON?"
    Format odpowiedzi: "Treść odpowiedzi zwróconej przez API."
    Co nalezy wykonać: "Wysłać zapytanie do API z wpisanym hasłem w polu password."

    Pytanie: "Jeden z rozmówców skłamał podczas rozmowy. Kto to był?"
    Format odpowiedzi: "Imię lub nazwisko lub pseudonim osoby, która skłamała w konwersacji"
    Co nalezy wykonać: "Na podstawie konwersacji oraz danych w kontekście zidentyfikować osobę, która kłamała"

    Pytanie: "Jakim przezwiskiem określany jest chłopak Barbary?"
    Format odpowiedzi: "Przezwisko chłopaka Barbary."
    Co nalezy wykonać: "Przeszukać podany kontekst w poszukiawniu odpowiedzi"
    
    **Format Twojej Odpowiedzi:**
    Format odpowiedzi: **Format odpowiedzi**
    Co należy wykonać: **Czynności do wykonania**
    `,
    `
    Pytanie: {question}
    `
)
       
    const chain = make_router(model_5_nano, 'result', prompt)

    const result = await chain.invoke({
        question: state.allQuestions[state.questionId].question
    })

    console.log('=======ANSWER PLAN NODE=======')
    console.dir(result.api_response.content, { depth: null, colors: true });
    

    return{
        answerPlan: result.api_response.content
    }
}

async function dataGatheringNode(state: typeof State.State){

    const prompt = system_user_prompt(
        `
        **Rola:**
        Jesteś odpowiedzialny za weryfikację kompletności danych. Głęboko analizujesz otrzymane dane kontekście oraz w instrukcji systemowej aby poprawnie ocenić kompletność danych. 

        **Zadanie:**
        Weryfkacja, czy aktualnie zgromadzone dane, są wystarczające aby odpowiedzieć na pytanie zgodnie z podanym formatem oraz czynnościami, które nalezy wykonać.

        Zidentyfikuj imiona kazdego z rozmówców. Jeśli w konwersacji pojawia się agentka, musisz znaleźć jej imię. 

        Kontekst w pierwszej iteracji zawiera tylko konwersacje. Jeśli w kontekście po 'Informacje zebrane z Faktów:' będzie tekst, to oznacza, ze nie jest to pierwsza iteracja, będą tam się znajdować dopisane dane z bazy wektorowej z faktami. 

        Konwersacje powinny być Twoim głównym źródłem informacji, jeśli jednak będzie brakowało informacji potrzebnych do wypełnienia zadania, wywołaj narzędzie: 'research_query', które dopisze do kontekstu wynik, dopasowany do zapytania podanego w argumencie wywołania narzędzia. Pamiętaj, ze fakty są przechowywane w bazie wektorowej. Twoje zapytanie powinno brać pod uwagę charakterystykę rozumienia 'wektorów'. To znaczy, ze baza zamieni Twoje zapytanie na wektory oraz zwróci dopasowaną odpowiedź. Postaraj się zawrzeć słowa klucz. 
        
        Jeśli uznasz, ze aktualnie zgromadzone dane w polu <zebrany_kontekst> są wystarczające, aby poprawnie odpowiedzieć na pytanie w odpowiednim formacie oraz wykonać wszystkie potrzebne kroki aby odpowiedziec na pytanie wybierz narzędzie: 'proceed_further_tool'. W argumencie wywołania narzędzia podaj krótką odpowiedź na pytanie: 'dlaczego uwazasz, ze aktualnie zebrane dane są wystarczające aby odpowiedzieć na pytanie?'.         

        Pytanie:{question}
        {plan}

        `,
        `
        <zebrany_kontekst>
        {context}
        </zebrany_kontekst>
        `
    )
    const tools = [researchTool, proccedTool]
    const llmWithTools = model_5_nano.bindTools(tools)

    const chain = make_router(llmWithTools, 'result', prompt)
    const result = await chain.invoke({
        question: state.allQuestions[state.questionId],
        plan: state.answerPlan,
        context: state.currentContext
    })

    console.log('=======DATA GATHERING NODE=======')
    console.dir(result.api_response.tool_calls, { depth: null, colors: true });
    
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

    const qdrantBesstMatchFactId = (Number(qdrantResults[0].payload?.factId) ?? -1);
    
    let fullText = ''
    const alreadyQueriedFacts = state.alreadyQueriedFacts;

    if(!alreadyQueriedFacts.includes(qdrantBesstMatchFactId)){
        
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

        fullText = allFactPart.points
            .sort((a: any, b: any) => (a.payload.position - b.payload.position)) // Optional: if you saved 'position'
            .map(p => p.payload?.text)
            .join("\n\n");

        alreadyQueriedFacts.push(qdrantBesstMatchFactId);
    }

    const newContext = formatAppendedData(currentContext, fullText, llmQuery)

    console.log('=======QDRANT SEARCH NODE=======')
    console.dir(newContext, { depth: null, colors: true });

    return {
        currentContext: newContext
    }
}

async function toolChoosingNode(state: typeof State.State){

    const prompt = system_user_prompt(
        `
            **Rola**
            Jesteś odpowiedzialną osobą decyzyjną, która specjalizuje się w wyborze narzędzi dla Agentów AI. Głęboko zastanawiasz się nad poprawną odpowiedzią.

            **Zadanie**
            Na podstawie pytania, formy odpowiedzi oraz wymaganych czynności, wybierz odpowiednie narzędzie.
            Jeśli odpowiedź na pytanie wymaga posiadania odpowiedzi z endpointu API, wybierz 'api_agent_tool'. Wtedy kontekst zostanie przekazany do Agenta, który sprawdzi wszystkie endpointy api się tam znajdzujące.
            Jeśli odpowiedź na pytanie wymaga analizy konteksu oraz wyciągnięcia poprawnych wniosów. Wybierz tool o nazwie 'proceed_further_tool'. Wtedy kolejnym krokiem, będzie głęboki research na temat poprawnej odpowiedzi, który będzie opierać się na aktualnie zebranym konekście. 
            Pytanie:{question}
            {plan}

            `,
            `
            <zebrany_kontekst>
            {context}
            </zebrany_kontekst>
        `
    )

    const tools = [apiAgentTool, proccedTool]
    const llmWithTools = model_5_nano.bindTools(tools)

    const chain = make_router(llmWithTools, 'result', prompt)
    const result = await chain.invoke({
        question: state.allQuestions[state.questionId],
        plan: state.answerPlan,
        context: state.currentContext
    })
    console.log('=======TOOL CHOOSING NODE=======')
    console.dir(result.api_response.tool_calls, { depth: null, colors: true });
    return{
        toolChoosingResponses: [result.api_response]
    }
}

async function apiAgentNode(state: typeof State.State){

    // look up for already collected responses (if it is not the first loop)
    let apiResponses = state.apiResponses;
    const question = state.allQuestions[state.questionId];

    // if no responses (first loop) populate responses
    if(apiResponses.length === 0){
        apiResponses = await invokeApiAgent(state.currentContext);
    }
    // 
    const answer: Answer = {
        questionId: question.questionId,
        question: question.question,
        answer: apiResponses[state.currentApiCall].message   
    }
    const currentApiCall = state.currentApiCall + 1

    console.log('=======AGENTAPI NODE =======')
    console.dir(answer, { depth: null, colors: true })

    return{
        allAnswers: [...state.allAnswers, answer],
        apiResponses: apiResponses,
        currentApiCall: currentApiCall
    }
}

// adjust prompt
async function answerNode(state: typeof State.State){
    const prompt = system_user_prompt(
        `
            **Rola:**
            Jesteś zaawansowanym analitykiem tekstowym wyspecjalizowanym w odpowiadaniu na pytania dotyczące kontekstu. Myślisz głęboko oraz wypisujesz sobie kolejne etapy do przejścia.

            **Zadanie:**
            Odpowiedz na pytanie na podstawie podanego kontekstu w polu <kontekst>, formatu odpowiedzi oraz kroków do wykonania. 
            Zwracaj uwagę na fakty, i bierz pod uwagę kolejność konwersacji poniewaz tez jest kluczowa. Kolejność jest wskazana przez numer konwersacji. 
            Zwróć równiez uwagę na elementy w tagu <Poprzednie pytania>. Są tam informacje, które mogą przydać się w kolejnych pytaniach.
            Wypisz najwazniejsze elementy, które pomogą Tobie zidentyfikować prawidłową odpowiedź, zignoruj szum.
            
            Pytanie: {question}
            {plan}

            **Forma odpowiedzi**
            Twoja odpowiedź powinna być mozliwie jak najkrótsza, bez zadnych udziwnień ani wyjaśnień. Potrzebuję tylko bezpośredniej odpowiedzi. 
        `,
        `
            <kontekst>
            {context}
            Imię Agentki to Barbara!!!
            </kontekst>
        `
    )

    const question = state.allQuestions[state.questionId];

    const chain = make_router(model_5_nano, 'result', prompt)
    const result = await chain.invoke({
        question: question,
        plan: state.answerPlan,
        context: state.currentContext
    })

    const answer: Answer = {
        questionId: question.questionId,
        question:question.question,
        answer: result.result
    }
    console.log('=======ANSWER NODE =======')
    console.dir(answer, { depth: null, colors: true })
    return{
        allAnswers: [...state.allAnswers, answer]
    }

}

async function validateAnswerNode(state: typeof State.State) {

    const validationPayload = state.validationPayload;
    const validationPayloadAnswers = validationPayload.answer;

    for(let i = state.allAnswers.length - 1; i < state.allQuestions.length; i++){
        
        const toBeFilled = state.allAnswers[i]?.answer ?? 'null'
        const index = (i+1).toString().padStart(2, '0')
        validationPayloadAnswers[index] = toBeFilled;
    }
    
    validationPayload.answer = validationPayloadAnswers

    const validationAnswer = await postPayload<ValidationPayload, Response>(
        validationPayload, 
        AIDEVS_REPORT_URL
    );
    console.log('=======VALIDATE ANSWER NODE =======')
    console.dir(validationAnswer, { depth: null, colors: true })

    return{
        validationPayload: validationPayload,
        validationResult: validationAnswer
    }
}



async function explainApiErrorNode(state: typeof State.State){

    const cleanAnswers = state.allAnswers.slice(0,-1);
    let currentApiCall = state.currentApiCall;
    let apiResponses = state.apiResponses;
    if (currentApiCall >= state.apiResponses.length) currentApiCall = 0, apiResponses = [];

    return{
        apiResponses: apiResponses,
        currentApiCall: currentApiCall,
        allAnswers: cleanAnswers
    }
}


async function explainErrorNode(state: typeof State.State){

    const wrongAnswer = state.wrongAnswers + '\n' + state.allAnswers[state.allAnswers.length - 1].answer;
    const cleanAnswers = state.allAnswers.slice(0,-1);

    const previousAnswers = state.allAnswers.map((answer)=>{
        return `Poprzednie pytanie: ${answer.question}\n Odpowiedź na poprzednie pytanie: ${answer.answer}\n`
    })
    

    const currentContext = state.conversations.map((conv: JsonFileConv)=> {
        return `Numer rozmowy: ${conv.convId}\n ${conv.conversation.join('\n')}`
    }).join('\n\n') + `Błędna odpowiedzi na aktualne pytanie: ${wrongAnswer}, nie mozesz juz odpowiedzieć w ten sposób ` + `<Poprzednie pytania> ${previousAnswers} </Poprzednie pytania>` + 
    '\n\nInformacje zebrane z Faktów: '

    
    return{
        allAnswers: cleanAnswers,      
        currentContext: currentContext,  
        answerPlan: '', 
        apiResponses: [],
        currentApiCall: 0,
        alreadyQueriedFacts: [],       
        validationResult: { code: 600, message: '' },
        wrongAnswers: wrongAnswer
    }
}

async function formatCorrectAnswerNode(state: typeof State.State){

    const previousAnswers = state.allAnswers.map((answer)=>{
        return `Poprzednie pytanie: ${answer.question}\n Odpowiedź na poprzednie pytanie: ${answer.answer}\n`
    })
    
    const currentContext = state.conversations.map((conv: JsonFileConv)=> {
            return `Numer rozmowy: ${conv.convId}\n ${conv.conversation.join('\n')}`
        }).join('\n\n') + `<Poprzednie pytania> ${previousAnswers} </Poprzednie pytania>` + '\n\nInformacje zebrane z Faktów: '

    return{
        currentContext: currentContext,
        apiResponses: [],
        currentApiCall: 0,
        alreadyQueriedFacts: [],
        answerPlan: '',
        questionId: state.questionId + 1,
        validationResult: {code: 600, message:''},
        wrongAnswers:''
    }
}

function toolChoosingRouter(state: typeof State.State){

    const lastToolChoosingMessage = state.toolChoosingResponses[state.toolChoosingResponses.length - 1] as AIMessage

    if(!(lastToolChoosingMessage?.tool_calls?.length)) throw new Error('Error in data gathering router node. The Data Gathering node did not return a tool call.')

    const toolName = lastToolChoosingMessage.tool_calls[0].name;

    if(toolName === 'api_agent_tool')return 'agent'
    if(toolName === 'proceed_further_tool')return 'proceed'
    throw new Error ('No matching tools have been called in dataGatheringNode')
}

function dataGatheringRouter(state: typeof State.State){

    const lastDataGatheringMessage = state.dataGatheringResponses[state.dataGatheringResponses.length - 1] as AIMessage

    if(!(lastDataGatheringMessage?.tool_calls?.length)) throw new Error('Error in data gathering router node. The Data Gathering node did not return a tool call.')
    
    const toolName = lastDataGatheringMessage.tool_calls[0].name;

    if(toolName === 'research_query') return 'research'
    if(toolName === 'proceed_further_tool')return 'proceed'
    throw new Error ('No matching tools have been called in dataGatheringNode')
}

function afterHumanRouter(state: typeof State.State){
    return state.isApproved ? 'finish' : 'retry';
}
function validationRouter(state: typeof State.State) {
    
    const responseQuestionId = Number((extractQuestionId(state.validationResult.message) ?? 0));
    
    if(state.allAnswers.length === state.allQuestions.length && state.validationResult.code !== -350) return 'finish';
    else if(state.allQuestions[state.questionId].questionId < responseQuestionId) return 'success';
    else if(state.allQuestions[state.questionId].questionId === responseQuestionId && state.currentApiCall > 0)return 'retryApi';
    else return 'retry';
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

    const validationPayload: ValidationPayload = {
        task:'phone',
        apikey: AIDEVS_API_KEY,
        answer:{}
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
        }).join('\n\n') + '\n\nInformacje zebrane z Faktów: ',
        dataGatheringResponses:[], 
        toolChoosingResponses:[],
        allAnswers: [],
        isApproved: false,
        validationPayload: validationPayload, 
        validationResult: {code: 600, message:''},
        apiResponses: [],
        currentApiCall:0,
        alreadyQueriedFacts:[],
        wrongAnswers: ''
    }
    const config = { configurable: { thread_id: "cli-session-1" }, recursionLimit: 100 };
    
    // checkpointer definition
    const checkpointer = new MemorySaver();

    // graph definition
    const workflow = new StateGraph(State)
        .addNode('plan', answerPlanNode)
        .addNode('qdrant', qdrantResearchNode)
        .addNode('gather', dataGatheringNode)
        .addNode('choose', toolChoosingNode)
        .addNode('human', humandNode)
        .addNode('api', apiAgentNode)
        .addNode('answer', answerNode)
        .addNode('validate', validateAnswerNode)
        .addNode('error', explainErrorNode)
        .addNode('apiError', explainApiErrorNode)
        .addNode('format', formatCorrectAnswerNode)
        .addEdge(START, 'plan')
        .addEdge('plan','gather')
        .addConditionalEdges(
            'gather',
            dataGatheringRouter,
            {
               research: 'qdrant',
               proceed: 'choose' 
            }
        )
        .addConditionalEdges(
            'choose',
            toolChoosingRouter,
            {
                agent: 'api',
                proceed: 'answer'
            }
        )
        .addEdge('api', 'validate')
        .addEdge('answer', 'validate')
        .addEdge('qdrant', 'human')
        .addConditionalEdges(
            'human',
            afterHumanRouter,
            {
                finish: 'gather',
                retry: END
            }
        )
        .addConditionalEdges(
            'validate',
            validationRouter,
            {
                finish: END,
                success: 'format',
                retryApi: 'apiError',
                retry: 'error'
            }
        )
        .addEdge('apiError', 'api')
        .addEdge('format', 'plan')
        .addEdge('error', 'plan')


    const app = workflow.compile({
        checkpointer,
        interruptBefore: ['human']
    });
    let result = await app.invoke(initialState, config)

    const rl = readline.createInterface({ input: stdin, output: stdout });
    

    while(true){

        const snapshot = await app.getState(config)
            
        if(snapshot.next.length === 0) break;

        const answer = await rl.question("Continue? (y/n): ")
        let approval: boolean = false

        if(answer === 'y') approval = true;
        if(answer === 'q') break;
        
        await app.updateState(config, {
            isApproved: approval
        });
        result = await app.invoke(null, config);
    }
    console.log(questions)
    console.log(result)
    rl.close();
}


// To do:
// implement other nodes:
// answerPlan - done
// data collection node - done
// vector db implementation + chunking - done
// vector db retrieval node - done
// tool calling node - done
// sub agent API caller - done
// research node - done 
// answer validation node - done
// admin task around Agent Invocation - done

// To Do: 
// 
