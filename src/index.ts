import { fetch_env, getQuestions } from './utils/utils.js';
import { Question } from './types.js';
import { invokeMainAgent } from './mainAgent.js';


const questionsUrl = fetch_env('AIDEVS_URL_ZAD51_QUESTIONS')
const questions: Question[] = await getQuestions(questionsUrl)

await invokeMainAgent(questions)
