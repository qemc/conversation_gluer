import 'dotenv/config'

export type AIVendor = 'openai' | 'anthropic' | 'googleai';

const API_KEY_MAP: Record<AIVendor, string> = {
    openai: 'OPENAI_API_KEY',
    anthropic: 'ANTHROPIC_API_KEY',
    googleai: 'GOOGLEAI_API_KEY'
}

export function loadApiKey(vendor: AIVendor): string{
    const envVar = API_KEY_MAP[vendor];
    const apiKey = process.env[envVar];

    if(!apiKey){
        throw new Error(`Missing ${envVar} in .env for vendor "${vendor}".`);
    }
    return apiKey;
}