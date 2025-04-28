import { Annotation, StateGraph, START, END } from '@langchain/langgraph';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import { addTool, subtractTool } from '../tools/arithmetic-tools';
import { bedrockModel, openAiModel } from '../llm/models';

//graph state
const State = Annotation.Root({
  messages: Annotation<any[]>({
    default: () => [],
    reducer: (curr, a) => [...curr, ...a],
  }),
  result: Annotation<number | null>({
    default: () => null,
    reducer: (_, v) => v,
  }),
});

type S = (typeof State)['State'];

//llm node (already bound to tools)
const model = openAiModel().bindTools([addTool, subtractTool]);
const SYS = new SystemMessage('You are a calculator. Use the add & subtract tools when needed.');

const llmNode = async ({ messages }: S) => {
  const reply = await model.invoke([SYS, ...messages]);

  // Bedrock quirk: strip duplicate tool_use fragments
  if (typeof reply.content !== 'string') {
    reply.content = reply.content.filter((c) => c.type !== 'tool_use');
  }
  return { messages: [reply] };
};

// route for further tool calls or finish
const route = ({ messages }: S) => (messages.at(-1)?.tool_calls?.length ? 'tools' : 'extract');

// pull the numeric answer out of recent tool messages
const extractNode = ({ messages }: S) => {
  const toolMsg = [...messages]
    .reverse()
    .find((m) => m.tool_call_id || m.name === 'add' || m.name === 'subtract');

  return { result: Number(toolMsg?.content ?? NaN) };
};

// build & compile graph
const calculator = new StateGraph(State)
  .addNode('llm', llmNode)
  .addNode('tools', new ToolNode([addTool, subtractTool]))
  .addNode('extract', extractNode)
  .addEdge(START, 'llm')
  .addConditionalEdges('llm', route, { tools: 'tools', extract: 'extract' })
  .addEdge('tools', 'llm') // loop back after tool use
  .addEdge('extract', END)
  .compile();

// example usage
(async () => {
  // 1️⃣ Initialize empty history and state
  let history: any[] = [];
  let state: { messages: any[]; result: number | null };

  const ask = async (q: string) => {
    // 2️⃣ Append the new user turn
    history.push(new HumanMessage(q));

    // 3️⃣ Invoke the same graph, feeding in the entire history
    state = await calculator.invoke({ messages: history });

    // 4️⃣ Log everything
    console.log(`\nUser        → ${q}`);
    const llmMsg = state.messages.at(-1)!;
    console.log('Assistant   →', llmMsg.content);
    state.messages
      .filter((m) => m.name === 'add' || m.name === 'subtract')
      .forEach((t) => console.log(`Tool:${t.name.padEnd(8)}→ ${t.content}`));
    console.log('Result      →', state.result);
    console.log('────────────────────────────────');
  };

  await ask('Calculate 5 + 3 - 2');
  await ask('Add 10 to previous result');
})();
