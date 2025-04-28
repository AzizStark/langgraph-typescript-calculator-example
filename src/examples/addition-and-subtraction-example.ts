import { Annotation, StateGraph, START, END } from '@langchain/langgraph';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import { addTool, subtractTool } from '../tools/arithmetic-tools';
import { bedrockModel } from '../llm/models';

/* ── 1. Graph state ───────────────────────────────────────────── */
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

/* ── 2. LLM node (already bound to tools) ─────────────────────── */
const model = bedrockModel().bindTools([addTool, subtractTool]);
const SYS = new SystemMessage('You are a calculator. Use the add & subtract tools when needed.');

const llmNode = async ({ messages }: S) => {
  const reply = await model.invoke([SYS, ...messages]);

  // Bedrock quirk: strip duplicate tool_use fragments
  if (typeof reply.content !== 'string') {
    reply.content = reply.content.filter((c) => c.type !== 'tool_use');
  }
  return { messages: [reply] };
};

/* ── 3. Should we call tools or finish? ───────────────────────── */
const route = ({ messages }: S) => (messages.at(-1)?.tool_calls?.length ? 'tools' : 'extract');

/* ── 4. Pull the numeric answer out of recent tool messages ───── */
const extractNode = ({ messages }: S) => {
  const toolMsg = [...messages]
    .reverse()
    .find((m) => m.tool_call_id || m.name === 'add' || m.name === 'subtract');

  return { result: Number(toolMsg?.content ?? NaN) };
};

/* ── 5. Build & compile graph ─────────────────────────────────── */
const calculator = new StateGraph(State)
  .addNode('llm', llmNode)
  .addNode('tools', new ToolNode([addTool, subtractTool]))
  .addNode('extract', extractNode)
  .addEdge(START, 'llm')
  .addConditionalEdges('llm', route, { tools: 'tools', extract: 'extract' })
  .addEdge('tools', 'llm') // loop back after tool use
  .addEdge('extract', END)
  .compile();

/* ── 6. Demo ─────────────────────────────────────────────────── */
(async () => {
  const ask = async (q: string) => {
    const state = await calculator.invoke({ messages: [new HumanMessage(q)] });

    // 1. show the question
    console.log(`\nUser        → ${q}`);

    // 2. show the assistant reply that may include tool calls
    const llmMsg = state.messages.at(-1);
    console.log('Assistant   →', llmMsg.content);

    // 3. show every tool message produced in this turn
    const toolMsgs = state.messages.filter((m) => m.name === 'add' || m.name === 'subtract');
    for (const t of toolMsgs) {
      console.log(`Tool:${t.name.padEnd(8)}→ ${t.content}`);
    }

    // 4. final extracted numeric answer
    console.log('Result      →', state.result);
    console.log('────────────────────────────────');
  };

  await ask('Calculate 5 + 3 - 2');
})();
