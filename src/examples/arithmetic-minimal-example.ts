import { Annotation, StateGraph, START, END, messagesStateReducer } from '@langchain/langgraph';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import { addTool, subtractTool, multiplyTool, divideTool } from '../tools/arithmetic-tools';
import { openAiModel, bedrockModel } from '../llm/models';
import { MemorySaver } from '@langchain/langgraph';

// 1️⃣ Only two bits of state: the chat history and the last numeric result
const State = Annotation.Root({
  messages: Annotation<any[]>({
    default: () => [],
    reducer: messagesStateReducer,
  }),
  result: Annotation<number | null>({
    default: () => null,
    reducer: (_, v) => v,
  }),
});

type S = (typeof State)['State'];

const tools = [addTool, subtractTool, multiplyTool, divideTool];

// 2️⃣ LLM node that uses all four tools
const llmNode = async ({ messages }: S) => {
  const system = new SystemMessage(
    'You are a calculator. If you can’t use the tools, return exactly “TOOL_FAIL” so we can catch it.'
  );
  const model = bedrockModel().bindTools(tools);
  const reply = await model.invoke([system, ...messages]);

  // Bedrock quirk: strip duplicate tool_use fragments
  if (Array.isArray(reply.content)) {
    reply.content = reply.content.filter((c) => c.type !== 'tool_use');
  }

  return { messages: [reply] };
};

// 3️⃣ Tool vs. extract routing
const route = (state: S) => (state.messages.at(-1)?.tool_calls?.length ? 'tools' : 'extract');

// 4️⃣ Pull the last tool output as your numeric result
const extractNode = ({ messages }: S) => {
  const lastTool = [...messages]
    .reverse()
    .find((m) => m.tool_call_id || tools.map((t) => t.name).includes(m.name));
  const num = Number(lastTool?.content ?? NaN);
  return { result: isNaN(num) ? null : num };
};

// 5️⃣ Build your graph with a simple in‑memory saver
const calculator = new StateGraph(State)
  .addNode('llm', llmNode)
  .addNode('tools', new ToolNode(tools))
  .addNode('extract', extractNode)
  .addEdge(START, 'llm')
  .addConditionalEdges('llm', route, { tools: 'tools', extract: 'extract' })
  .addEdge('tools', 'llm')
  .addEdge('extract', END)
  .compile({ checkpointer: new MemorySaver() });

// 6️⃣ Multi‑turn usage: just pass the same thread_id each time
// example usage
(async () => {
  const thread = 'calc-session'; // your persistent session key

  const ask = async (q: string) => {
    // invoke with persistence
    const state = await calculator.invoke(
      { messages: [new HumanMessage(q)] },
      { configurable: { thread_id: thread } }
    );

    console.log(`\nUser        → ${q}`);

    // 1️⃣ Last assistant reply is always the final message in state.messages
    const llmMsg = state.messages.at(-1)!;
    console.log('Assistant   →', llmMsg.content);

    // 2️⃣ Grab every tool call
    const toolNames = ['add', 'subtract', 'multiply', 'divide'];
    const toolMsgs = state.messages.filter((m) => toolNames.includes(m.name));
    for (const t of toolMsgs) {
      console.log(`Tool:${t.name.padEnd(8)}→ ${t.content}`);
    }

    // 3️⃣ Log the numeric result
    console.log('Result      →', state.result);

    console.log('────────────────────────────────');
  };

  await ask('Calculate 24 * 3 + 15');
  await ask('Divide the previous result by 9');
  await ask('Multiply this value by itself');
  await ask('Subtract 48 from the result');
  await ask('Add the square root of 225 to this number');
  await ask('Summarise what we have done so far');
  await ask('find the log of 123 to base 2');
})();
