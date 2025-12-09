import { StateGraph, START, END, Annotation, messagesStateReducer, MemorySaver } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { Question } from "./types.js";
import { make_router, system_user_prompt } from "./ai_utils/langchainHelpers.js";
import { promises as fs } from "fs";
import * as path from "path";
import { Conversation, Data, JsonFileConv, saveCache } from "./types.js";


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

const State = Annotation.Root({
    answerPlan: Annotation<string>,
    keptInfo: Annotation<string>,
    questionId: Annotation<number>,
    question: Annotation<string>,
    startInfo: Annotation<string>,
    allQuestions: Annotation<Question[]>,
    conversations: Annotation<JsonFileConv[]>

})

async function extractInfoNode(state: typeof State.State){

    const defineOneShotItems = system_user_prompt(
        `**Rola:**
        Jesteś Analitykiem Słów Kluczowych i Ekstrakcji Danych (Query Decomposition Engine). Twoim jedynym zadaniem jest przeanalizowanie listy pytań użytkownika i wyodrębnienie z nich konkretnych "Celów Informacyjnych" (faktów, encji, nazw własnych), które należy znaleźć w bazie tekstowej.

        **Kontekst:**
        Otrzymasz listę pytań w formacie JSON. Niektóre pytania są logiczne i złożone, inne proste.Kade z nich ma swoje ID, które, moze się przydać. Twoim zadaniem nie jest odpowiadanie na nie, a jedynie stworzenie "listy zakupów" - spisu konkretnych informacji, które trzeba wyszukać w tekście źródłowym. 

        **Instrukcje Systemowe:**
        1. **Analiza:** Przeczytaj każde pytanie z listy wejściowej.
        2. **Ekstrakcja:** Zidentyfikuj konkretny obiekt, nazwę lub daną techniczną, o którą pyta użytkownik. Szukaj takich elementów jak:
        - Artefakty techniczne (np. "endpoint API", "adres IP", "struktura JSON").
        - Tożsamości (np. "imiona rozmówców", "przezwisko", "nazwa firmy").
        - Dane uwierzytelniające (np. "hasło", "klucz dostępu").
        3. **Uproszczenie:** Odrzuć całą otoczkę gramatyczną pytania (np. słowa "Jaki jest...", "Kto to...", "Podaj mi..."). Zostaw tylko sam przedmiot wyszukiwania.
        4. **Normalizacja:** Jeśli pytanie jest złożone (np. "Kto kłamał?"), wyciągnij ogólną kategorię, która pomoże to ustalić (np. "imiona rozmówców", "zawód osoby").


        **Ograniczenia:**
        - Wynik ma być **płaską listą oddzieloną przecinkami**.
        - Nie numeruj wyników.
        - Nie dodawaj żadnych opisów ani wstępów.
        - Używaj mianownika (np. zamiast "chłopaka Barbary" wyciągnij "chłopak Barbary" lub frazę kluczową "przezwisko chłopaka Barbary" w formie umożliwiającej wyszukiwanie).

        **Przykłady (Few-Shot):**
        *Input:* "Jaki jest prawdziwy endpoint do API?"
        *Output:* endpoint API

        *Input:* "Jakie dwie osoby rozmawiają ze sobą w pierwszej (convId: 1) rozmowie?"
        *Output:* imiona rozmówców

        *Input:* "Co odpowiada poprawny endpoint API po wysłaniu hasła?"
        *Output:* endpoint API, hasło

        **Format Wyjściowy:**
        Jeden ciąg tekstowy oddzielony przecinkami.
        Przykład: \`endpoint API, hasło, imiona rozmówców w pierwszej rozmowie \``,
        `Pytania: {questions}`
    )
    const mainPrompt = system_user_prompt(
        `
        **Rola:**
        Jesteś Specjalistą ds. Ekstrakcji Danych (Data Retrieval Specialist). Twoim zadaniem jest przeszukanie dostarczonych transkrypcji rozmów i znalezienie konkretnych wartości dla listy zdefiniowanych obiektów.

        **Kontekst:**
        Otrzymasz dwa zestawy danych:
        1. \`<lista_celow>\`: Lista obiektów, haseł i informacji, które musisz znaleźć (wygenerowana przez poprzedni proces).
        2. \`<kontekst>\`: Pełna treść 5 rozmów, w których ukryte są odpowiedzi.

        **Instrukcje Systemowe:**
        1. **Wczytaj Cele:** Przeanalizuj każdy element z \`<lista_celow>\`. To są Twoje priorytety.
        2. **Skanuj Kontekst:** Przeszukaj \`<kontekst>\` (rozmowy) w poszukiwaniu konkretnych wartości odpowiadających celom.
        - Jeśli celem jest "endpoint API", znajdź URL.
        - Jeśli celem jest "hasło", znajdź ciąg znaków hasła.
        - Jeśli celem jest "imię", znajdź nazwę własną.
        3. **Weryfikacja:** Upewnij się, że wyciągnięta wartość jest bezpośrednio powiązana z szukanym obiektem w tekście. Nie zgaduj.
        4. **Formatowanie:** Zwróć wynik w formacie JSON, gdzie kluczem jest szukany obiekt z listy, a wartością jest to, co znalazłeś w tekście.
        5. Musisz być pewnym, ze odpowiedzi znajdują się w konwersacjach. Jeśli się nie znajdują, wpisz \`null\` lub \`Brak danych\`.

        **Ograniczenia:**
        - Ignoruj informacje, które nie znajdują się na \`<lista_celow>\`.
        - Jeśli nie możesz znaleźć wartości dla danego celu, wpisz \`null\` lub \`Brak danych\`.
        - Wyciągaj dane w formacie surowym (np. jeśli w tekście jest "hasło to: rurki123", wyciągnij tylko \`rurki123\`).
        - Pomiń znaki interpunkcyjne na końcu wyciągniętych fraz (kropki, przecinki).

        **Format Wyjściowy:**
        Czysty obiekt JSON.
        Przykład:
        \`\`\`json
        {{
        "endpoint API": "[https://api.przyklad.pl](https://api.przyklad.pl)",
        "hasło": "tajneHaslo123",
        "przezwisko chłopaka Barbary": "Basiula"
        }}`
    ,
        `
        <lista_celow>
        {itemsToFocus}
        </lista_celow>

        <kontekst>
        {convs}
        </kontekst>
    
        `
    )
    const oneShotItems = make_router(model_51, 'itemsToFocus', defineOneShotItems)
    const mainOneShot = make_router(model_51, 'result', mainPrompt)

    const chain = oneShotItems.pipe(mainOneShot)
    const result = await chain.invoke({
        questions: state.allQuestions, 
        convs: state.conversations
    })
    console.log(result)
    return{}
}

async function answerPlanNode(state: typeof State.State){
    
    const prompt = system_user_prompt(
        `
        Jesteś Architektem Rozwiązywania Problemów (Problem Solving Architect). Twoim zadaniem jest przeanalizowanie pytania użytkownika i stworzenie precyzyjnego planu działania dla autonomicznego agenta.

        Dla każdego pytania musisz wygenerować wyjście składające się z dwóch sekcji:

        1. **Oczekiwany format:** Precyzyjny opis tego, jak ma wyglądać ostateczna odpowiedź (np. "Czysty ciąg znaków URL", "Obiekt JSON", "Pojedyncze imię").
        2. **Kroki do podjęcia:** Lista ponumerowanych, atomowych czynności, które należy wykonać, aby uzyskać odpowiedź.

        **Zasady tworzenia kroków:**
        - **Rozbijaj zależności:** Jeśli pytanie wymaga użycia narzędzia (np. API), musisz najpierw uwzględnić kroki znalezienia niezbędnych parametrów (np. "Znajdź URL", "Znajdź hasło"), a dopiero potem krok wykonania akcji ("Wyślij zapytanie").
        - **Logika i Dedukcja:** Jeśli pytanie zawiera warunki logiczne (np. "podany przez osobę, która nie kłamała"), musisz dodać krok analizy merytorycznej (np. "Przeanalizuj kontekst, aby wykluczyć kłamcę").
        - **Precyzja:** Kroki muszą być instrukcjami typu "Znajdź", "Przeanalizuj", "Oblicz", "Wyślij".

        **Format Wyjściowy:**
        Zwróć odpowiedź w ściśle określonym formacie:

        Oczekiwany format:
        [Opis formatu]

        Kroki do podjęcia:
        1. [Krok pierwszy]
        2. [Krok drugi]
        ...
        `,
        `
        Pytanie do analizy:
        {question}
        `
    )
    const chain = make_router(model_51, 'result', prompt)
    const result = await chain.invoke({
        question: state.question
    })

    return{
        answerPlan: result.api_response.content
    }
}


async function dataGatherinNode(){
    
    const prompt = system_user_prompt(
        `
       
        `,
        `

        `
    )

    return{}
}

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
        question: '',
        startInfo: '',
        allQuestions: questions,
        conversations: conversations
    }
    const config = { configurable: { thread_id: "cli-session-1" } };
    
    // checkpointer definition
    const checkpointer = new MemorySaver();

    // graph definition
    const workflow = new StateGraph(State)
        .addNode('extract', extractInfoNode)
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

