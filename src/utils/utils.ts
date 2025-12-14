import { promises as fs } from "fs";
import * as path from 'path';
import axios from 'axios'

export async function fetch_url(url: string){

    const response = await fetch(url);
    if(!response.ok) throw new Error(`HTTP ${response.status} - failed to fetch data`)
    
    const data = await response.json()
    return data
}

export function fetch_env(ENV_VAR:string){

    const ENV_VAR_ = process.env[ENV_VAR];
    if(!ENV_VAR_) throw new Error(`Error in fetching ${ENV_VAR_} from .env`)
    return ENV_VAR_
}


export function parse_json_to_array(jsonObj: Record<string, string>): string[] {
  if (!jsonObj || typeof jsonObj !== "object") {
    throw new Error("Invalid JSON object provided to parse_json_to_array");
  }

  const sortedKeys = Object.keys(jsonObj).sort((a, b) => Number(a) - Number(b));
  return sortedKeys.map((key) => jsonObj[key].trim());
}

// await saveToFile("/Users/you/Desktop", "my-data", data);

export async function saveToFile(rootPath: string, fileName:string, value: any){
  const finalFileName = fileName.endsWith(".json") ? fileName : `${fileName}.json`;
  const filePath = path.join(rootPath, finalFileName);

  await fs.mkdir(rootPath, { recursive: true });
  const json = JSON.stringify(value, null, 2);
  await fs.writeFile(filePath, json, "utf8");
} 

export async function readFromFile<T>(filePath: string): Promise<T> {
  
  const text = await fs.readFile(filePath, "utf8");
  const data = JSON.parse(text);

  return data as T;
}

export function normalize_to_compare(str: string): string {
  return str
    .toLowerCase()
    .replace(/\s+/g, '')           // Remove all whitespace
    .replace(/[^\p{L}\p{N}]/gu, ''); // Remove non-letters/non-numbers (Unicode safe for Polish)
}

export function filterOutItems(sourceArray: string[], itemsToRemove: string[]): { filteredList: string[], removedCount: number } {
  const exclusionSet = new Set(
    itemsToRemove.map((item) => normalize_to_compare(item))
  );

  const filteredList = sourceArray.filter((item) => {
    const normalizedItem = normalize_to_compare(item);
    return !exclusionSet.has(normalizedItem);
  });

  return {
    filteredList,
    removedCount: sourceArray.length - filteredList.length
  };
}


export async function saveJsonToFile(
  fileName: string, 
  rootPath: string, 
  data: any
): Promise<void> {

  const fullPath = path.join(rootPath, fileName);
  const jsonContent = JSON.stringify(data, null, 2);

  await fs.writeFile(fullPath, jsonContent, 'utf-8');
  console.log(`Saved data to ${fullPath}`);
}

export async function postPayload<Payload>(payload: Payload, url: string){
  try {
    // Axios automatically stringifies body and sets headers
    const { data } = await axios.post<Payload>(url, payload);
    console.log(data)
    return data

  } catch (error) {
    console.error(error);
  }
}