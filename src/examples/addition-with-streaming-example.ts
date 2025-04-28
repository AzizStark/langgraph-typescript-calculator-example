import { Annotation, StateGraph, START, END } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage } from '@langchain/core/messages';
import 'dotenv/config';

/* state */
const Calc = Annotation.Root({
  a: Annotation<number>(),
  b: Annotation<number>(),
  result: Annotation<number>(),
});
type S = (typeof Calc)['State'];

/* model (streaming on) */
const model = new ChatOpenAI({
  modelName: 'gpt-4o-mini',
  temperature: 0,
  streaming: true, // ← enable token stream
  apiKey: process.env.OPENAI_API_KEY,
});

/* add node with live output */
const addNode = async ({ a, b }: S) => {
  const stream = await model.stream([new HumanMessage(`Only return ${a + b}`)]);

  let full = '';
  for await (const chunk of stream) {
    const text = typeof chunk.content === 'string' ? chunk.content : '';
    process.stdout.write(text); // live tokens
    full += text;
  }
  console.log(); // newline after stream
  return { result: Number(full) };
};

/* graph */
const calculator = new StateGraph(Calc)
  .addNode('add', addNode)
  .addEdge(START, 'add')
  .addEdge('add', END)
  .compile();

/* run */
(async () => {
  const { result } = await calculator.invoke({ a: 5, b: 3 });
  console.log('→', result);
})();
