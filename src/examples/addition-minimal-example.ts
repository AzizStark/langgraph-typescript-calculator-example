import { Annotation, StateGraph, START, END } from '@langchain/langgraph';
import { HumanMessage } from '@langchain/core/messages';
import { openAiModel } from '../llm/models';

/* ---------- 1. Graph state ---------- */
const State = Annotation.Root({
  a: Annotation<number>(), // input 1
  b: Annotation<number>(), // input 2
  result: Annotation<number>(), // output
});
type S = (typeof State)['State']; // inferred state type  :contentReference[oaicite:0]{index=0}

const addNode = async ({ a, b }: S) => {
  const reply = await openAiModel().invoke([new HumanMessage(`Only return ${a + b}`)]);
  return { result: Number(reply.content) };
};

/* ---------- 3. Graph ---------- */
const calculator = new StateGraph(State)
  .addNode('add', addNode)
  .addEdge(START, 'add')
  .addEdge('add', END)
  .compile();

/* ---------- 4. Run ---------- */
(async () => {
  const { result } = await calculator.invoke({ a: 5, b: 3 });
  console.log(`5 + 3 =`, result); // → 8
})();
