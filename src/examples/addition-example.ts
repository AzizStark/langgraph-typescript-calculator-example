// Import from index.ts first to ensure dotenv is configured
import './index';

import { HumanMessage } from '@langchain/core/messages';
import { ChatOpenAI } from '@langchain/openai';
import { Annotation, StateGraph, START, END } from '@langchain/langgraph';

// Define a simple state annotation for our addition calculator
export const AdditionStateAnnotation = Annotation.Root({
  // Input values
  a: Annotation<number>({
    reducer: (_, val) => val,
  }),
  b: Annotation<number>({
    reducer: (_, val) => val,
  }),
  // Result of addition
  result: Annotation<number | undefined>({
    reducer: (_, val) => val,
    default: () => undefined,
  }),
});

// Define the type for our state
export type IAdditionStateAnnotation = typeof AdditionStateAnnotation;

const model = new ChatOpenAI({
  modelName: 'gpt-4o-mini',
  temperature: 0,
  apiKey: process.env.OPENAI_API_KEY,
});

const llmAddNode = async (
  state: IAdditionStateAnnotation['State']
): Promise<Partial<IAdditionStateAnnotation['State']>> => {
  const prompt = `I need you to add two numbers: ${state.a} and ${state.b}. 
Please respond with only the numerical result, nothing else.`;

  const response = await model.invoke([new HumanMessage(prompt)]);

  const content = response.content.toString().trim();
  const calculatedResult = parseInt(content, 10);

  return { result: calculatedResult };
};

export const createAdditionGraph = (
  llmAddNode: (
    state: IAdditionStateAnnotation['State']
  ) => Promise<Partial<IAdditionStateAnnotation['State']>>
) => {
  const builder = new StateGraph(AdditionStateAnnotation)
    .addNode('llm_add', llmAddNode)
    .addEdge(START, 'llm_add')
    .addEdge('llm_add', END);

  return builder.compile();
};

export const runAddition = async (a: number, b: number) => {
  const graph = createAdditionGraph(llmAddNode);
  const initialState = { a, b };
  const result = await graph.invoke(initialState);

  console.log(`Addition via LLM: ${a} + ${b} = ${result.result}`);
  return result;
};

async function main() {
  await runAddition(12315, 551233);
}

main();
