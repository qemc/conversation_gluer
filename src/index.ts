import 'dotenv/config';
import { fetch_url, fetch_env, parse_json_to_array, saveToFile, normalize_to_compare } from './utils/utils.js';
import { Conversation, Data, Part_embeddings, Part_cosine_similarity, ChainIO, ConvDetails, Question} from './types.js';
import { cosineSimilarity } from '@langchain/core/utils/math';
import { make_router, system_user_prompt, get_embedding, LangChainOpenAImodel, make_human_router } from './ai_utils/langchainHelpers.js';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { RunnableLambda } from '@langchain/core/runnables';
import { JsonOutputParser } from '@langchain/core/output_parsers';
import { invokeAgent } from './langgraphAgent.js';
import { number } from 'zod/v3';
import { invokeMainAgent } from './mainAgent.js';
import { uploadPreCalculatedChunks } from './ai_utils/loadChunkedFacts.js';


const model_5nano = LangChainOpenAImodel();
const model_5mini = LangChainOpenAImodel('gpt-5-mini');
const model_o3 = LangChainOpenAImodel('o3');

const parser = new JsonOutputParser();

async function process_converstaion_data(url: string):Promise<Data> {
    const data = await fetch_url(url)
    
    let con1 = data.rozmowa1 as Conversation
    let con2 = data.rozmowa2 as Conversation
    let con3 = data.rozmowa3 as Conversation
    let con4 = data.rozmowa4 as Conversation
    let con5 = data.rozmowa5 as Conversation

    con1.id = 1
    con2.id = 2
    con3.id = 3
    con4.id = 4
    con5.id = 5

    let conversations = [con1, con2, con3, con4, con5]
    let parts = data.reszta 

    return {
        conversations: conversations,
        parts: parts
    } as Data
}
const url = fetch_env('AIDEVS_URL_ZAD51')
// const processedConversations = await process_converstaion_data(url)

const questionsUrl = fetch_env('AIDEVS_URL_ZAD51_QUESTIONS')
async function getQuestions(url: string): Promise<Question[]>{
    const questions = await fetch_url(url);
    
    const questionList: Question[] = Object.entries(questions).map(([id,text])=> ({
        questionId: Number(id),
        question: String(text)
    })) 
    return questionList
}

const questions: Question[] = await getQuestions(questionsUrl)

// The idea was to test how embeddings would work with completing the conversation. 
// It definietley did not fulfil the requirements. 
// Although it was fun to test it. 
async function try_embeddings(
    url: string
){
    const data = await process_converstaion_data(url);
    if(!data) throw new Error('Error while reading data from the url')
    
    let part_embeddings: Part_embeddings[] = [];
    
    for(let i = 0; i<data.parts.length; i++){

        let part: string = data.parts[i];
        const embeddings: number[] = await get_embedding(part);

        const part_embedding = {
            text: part,
            embeddings:embeddings
        } as Part_embeddings

        part_embeddings.push(part_embedding)
    }
    
    let conversations = data.conversations;
    conversations.sort((a, b) => a.length - b.length);

    for(let i = 0; i < conversations.length; i++){
        console.log(conversations[i].length)
        const conversation_arr_str: string[] = [data.conversations[i].start]

        for(let x = 0; x<conversations[i].length - 2; x++){

            const string_conv:string = conversation_arr_str.join('\n');
            const current_conversation_embeddding: number[] = await get_embedding(string_conv);
            
            let cosine_sim_current_conv:Part_cosine_similarity[] = await Promise.all(part_embeddings.map(async part => {
                const cosine_sim: number = cosineSimilarity(
                    [current_conversation_embeddding], 
                    [part.embeddings]
                )[0][0] 
                
                return {
                    text: part.text,
                    cosine_sim: cosine_sim
                } as Part_cosine_similarity
            }))
            const best_match = cosine_sim_current_conv.reduce((prev, current) => 
                current.cosine_sim > prev.cosine_sim ? current : prev
            );

            part_embeddings = part_embeddings.filter(
                part => part.text !== best_match.text
                );
            conversation_arr_str.push(best_match.text)
        }
        console.log(conversation_arr_str)
    }

}
// The idea is to ask LLM (some more thoughtful models like o4-mini or o3) to complete the conversations.
// The input would be start and end of the conversation and the number of sentences. 
// LLM would need to return sequenced array-like structure with chosen parts.
// It would be crucial to remove them from array
async function try_llm(
    url: string
){
    const data = await process_converstaion_data(url);
    if(!data) throw new Error('Error while reading data from the url')
    
    let conversations:Conversation[] = data.conversations;
    
    let parts_:string[] = data.parts;
    let parts = parts_.map((key) => key.trim())

    // Prompt 1 (main prompt)           o3, o4-mini, o4-mini-deep-research	
    const main_prompt: ChatPromptTemplate = system_user_prompt(
        
        `
            Jesteś ekspertem lingwistycznym i logikiem. Twoim zadaniem jest rekonstrukcja poprawnej kolejności konwersacji na podstawie rozsypanych fragmentów.

            ### DANE WEJŚCIOWE:
            1. Zdanie Początkowe: {{PIERWSZE_ZDANIE}}
            2. Zdanie Końcowe: {{OSTATNIE_ZDANIE}}
            3. Oczekiwana całkowita liczba zdań (N): {{LICZBA_N}}
            4. Kandydaci na środek (lista nieuporządkowana): 
            {{LISTA_KANDYDATÓW}}

            ### INSTRUKCJA:
            Twoim celem jest ułożenie spójnej konwersacji składającej się dokładnie z N zdań (wliczając początek i koniec).

            Kroki logiczne, które musisz wykonać (w myśli):
            1. Przeanalizuj "Zdanie Początkowe" i poszukaj w kandydatach jego naturalnego następstwa (pytanie->odpowiedź, powitanie->reakcja).
            2. Przeanalizuj "Zdanie Końcowe" i znajdź, co musi wystąpić bezpośrednio przed nim.
            3. Dopasuj pozostałe zdania, zwracając uwagę na:
            - Zaimki (np. "ona" musi odnosić się do kogoś wymienionego wcześniej).
            - Następstwo czasowe.
            - Logikę pytania i odpowiedzi.
            4. Upewnij się, że użyłeś odpowiedniej liczby zdań środkowych, aby suma wyniosła N.

            ### RESTRYKCJE (KRYTYCZNE):
            - Wyjście musi zawierać WYŁĄCZNIE ponumerowaną listę.
            - Nie dodawaj żadnych wstępów, wyjaśnień ani podsumowań.
            - Nie zmieniaj ani jednego znaku w treści zdań (zachowaj oryginalną interpunkcję, wielkość liter, błędy, myślniki).
            - Każde zdanie z listy kandydatów może być użyte tylko raz.

            ### FORMAT WYJŚCIA:
            1. [Zdanie Początkowe]
            2. [Zdanie środkowe 1]
            ...
            N. [Zdanie Końcowe]
        `,
        `
        Zdanie Początkowe: {first_sentence}  
        Zdanie końcowe: {last_sentence}  
        Oczekiwana całkowita liczba zdań (N): {length}  
        Kandydaci na środek (lista nieuporządkowana): {parts}
        `
    )
    const prompt1_router: RunnableLambda<ChainIO,ChainIO> = make_router(
        model_o3,
        'chosen_options',
        main_prompt
    ) 

    // Prompt 2 (llm parser)            5-nano, 5-mini
    const llm_parser_prompt: ChatPromptTemplate = system_user_prompt(
        `
        Jesteś odpowiedzialny za analizę odpowiedzi. Otrzymaną listę ponumerowaną listę, którą zwracasz w formacie json. 
        {{
            "1":"dokładnie to co było napisane w pierwszym punkcie",
            "2":"dokładnie to co było napisane w drugim punkcie",
            "3":"dokładnie to co było napisane w trzecim punkcie",
            "N":"dokładnie to co było napisane w N-tm punkcie",
        }}

        gdzie N to numer ostatniego elementu i stanowi liczbę wszystkich punktów.
        Zwracasz tylko jsona nic innego. Nie dodajesz nic od siebie. 
        Nie zmieniasz równie zadnych znaków, które zostały ci przekazane w liście. Jeśli jest tam myślnik, jest tam po coś. 
        `,
        `
        Lista:
        {chosen_options}
        `
    )
    const prompt2_router: RunnableLambda<ChainIO,ChainIO> = make_router(
        model_5nano, 
        'parsed_data',
        llm_parser_prompt
    )

    const human_router: RunnableLambda<ChainIO,ChainIO> = make_human_router(
        'Check coversation before proceeding.',
        'human_result'
    )

    conversations.sort((a, b) => a.length - b.length);
    for(let i = 0; i < conversations.length; i++){

        const parts_string: string = parts.join('\n');
        
        const chain = prompt1_router.pipe(prompt2_router).pipe(human_router);

        const result = await chain.invoke({
            first_sentence: conversations[i].start,
            last_sentence: conversations[i].end,
            length: conversations[i].length,
            parts: parts_string
        })


        const json_array_string: string = String(result?.parsed_data)
        console.log(json_array_string)

        const json_data = await parser.invoke(json_array_string)
        console.log(json_data)

        const raw_used_parts = parse_json_to_array(json_data);
        const summaryConversationPrompt: ChatPromptTemplate = system_user_prompt(
        `
        Jesteś wyspecjalizowany w tworzeniu podsumowywań konwersacji. 
        Na podstawie kilku zdań jesteś w stanie stwierdzić kto z kim rozmawia. 
        Wyłapujesz specyficzne tematy rozmowy. 
        Kazdy z tematów poruszonych w rozmowie jesteś stanie wyłapać. 
        Zwracasz dane w następującym formacie: 
        {{
            "rozmowcy": ["rozmówca1", "rozmówca1"],
            "wspomnieni":["Osoba wspomniana1", "Osoba wspomniana2"...]
            "wyrazenia_klucze": ["wyrazenie1", "wyrazenie2", "wyrazenie3"...]
        }}
        
        rozmowcy to są osoby, które ze sobą rozmawiają
        wspomnieni to są osoby, które zostały wspomniane w tekście
        wyrazenia_klucze to wyrazenia opisujące tekst, maksymalnie 10 najbardziej charakterystycznych. 
        Wyrazenia powinny być bardzo specyficzne dla danej konwersacji, nie powinny się w nich znajdowac osoby, poniwaz zostaly wspomniane w innych polach. 
        
        Dane wyzej będą słuzyć jako wskazówka dla llmu, który tekst wybrać, zeby znaleźć odpowiedzi na swoje pytanie.

        np:
        Konwersacja:
        - Samuelu! helooo?! Słyszysz mnie teraz? Zadzwoniłem ponownie, bo chyba znowu z zasięgiem jest u Ciebie jakiś problem...
        - tak Zygfryd, słyszę Cię teraz dobrze. Przepraszam, gdy poprzednio dzwoniłeś, byłem w fabryce. Wiesz, w sektorze D, gdzie się produkuje broń i tutaj mają jakąś izolację na ścianach dodatkową. Telefon gubi zasięg. Masz jakieś nowe zadanie dla mnie?
        - tak. Mam dla Ciebie nowe zadanie. Skontaktuj się z Tomaszem. On pracuje w Centrali. Może pomóc Ci włamać się do komputera tego gościa. Masz już endpoint API?
        Odpowiedź:
        {{
            "rozmówcy": ["Samuel", "Zygfryd"],
            "Wspomnieni":["Samuel", "Zygfryd", "Tomasz"]
            "slowa_klucze": ["Ponowne połączenie", "Sektor D", "Produkcja broni", "Endpoint API"]
        }}

        Nie zwracasz nic innego, tylko strukturę wyglądającą jak json.
        `,
        `
        Konwersacja:
        {data}
        `
        
        )

        const sumary_router: RunnableLambda<ChainIO,ChainIO> = make_router(
            model_5mini, 
            'summary',
            summaryConversationPrompt
        )

        const res = parts.filter(bigItem => {
            const bigNorm = normalize_to_compare(bigItem);
            const existsInSmall = raw_used_parts.some(smallItem => normalize_to_compare(smallItem) === bigNorm);
            return !existsInSmall; 
        });

        parts = res;

        if(!result?.human_result) throw new Error(`Not approved by human user at conversation nr ${i+1} \n`)

        
        const conversation_to_save = raw_used_parts.join("\\n"); 
        const summaryResult = await sumary_router.invoke({data: conversation_to_save})
        const summaryResultJson = await parser.invoke(String(summaryResult?.summary))
        
        const convDetails = {
            text: conversation_to_save,
            details: summaryResultJson
        } as ConvDetails

        await saveToFile(process.env['CONV_PATH']!,`rozmowa${i+1}`, convDetails);
        console.log(convDetails)


    }
}


//await invokeAgent(processedConversations)

await invokeMainAgent(questions)

