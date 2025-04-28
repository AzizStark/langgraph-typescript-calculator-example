import { Annotation, StateGraph, START, END } from '@langchain/langgraph';
import { HumanMessage, SystemMessage, AIMessage, AIMessageChunk } from '@langchain/core/messages';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import { addTool, subtractTool } from '../tools/arithmetic-tools';
import { openAiModel } from '../llm/models';

// State definition
const State = Annotation.Root({
  messages: Annotation<any[]>({ default: () => [], reducer: (curr, a) => [...curr, ...a] }),
  result: Annotation<number | null>({ default: () => null, reducer: (_, v) => v }),
});

type S = (typeof State)['State'];

// LLM node
const model = openAiModel(true).bindTools([addTool, subtractTool]);

const SYS = new SystemMessage('You are a calculator.');

const llmNode = async ({ messages }: S, options?: any): Promise<{ messages: AIMessageChunk[] }> => {
  const onStream = options?.onStream;
  const handler = {
    handleLLMNewToken(token: string) {
      onStream?.(new AIMessageChunk(token));
    },
    handleLLMEnd() {
      onStream?.(new AIMessage(''));
    },
  };

  // @ts-ignore
  const reply = await model.invoke([SYS, ...messages], handler);
  if (Array.isArray(reply.content))
    reply.content = reply.content.filter((c) => c.type !== 'tool_use');
  return { messages: [reply as AIMessageChunk] };
};

// Routing
const route = ({ messages }: S) => (messages.at(-1)?.tool_calls?.length ? 'tools' : 'extract');

// Extract result
const extractNode = ({ messages }: S) => {
  const m = [...messages]
    .reverse()
    .find((m) => m.tool_call_id || ['add', 'subtract'].includes(m.name));
  return { result: Number(m?.content ?? NaN) };
};

// Graph setup
const calculator = new StateGraph(State)
  .addNode('llm', llmNode)
  .addNode('tools', new ToolNode([addTool, subtractTool]))
  .addNode('extract', extractNode)
  .addEdge(START, 'llm')
  .addConditionalEdges('llm', route, { tools: 'tools', extract: 'extract' })
  .addEdge('tools', 'llm')
  .addEdge('extract', END)
  .compile();

// Demo
(async () => {
  const ask = async (q: string) => {
    const state = await calculator.invoke({ messages: [new HumanMessage(q)] });
    console.log(`\nUser → ${q}`);
    process.stdout.write('Assistant → ');
    state.messages.forEach(
      (m) => m instanceof AIMessageChunk && process.stdout.write(m.content as string)
    );
    console.log();
    state.messages
      .filter((m) => ['add', 'subtract'].includes(m.name))
      .forEach((t) => console.log(`Tool:${t.name} → ${t.content}`));
    console.log('Result →', state.result);
  };
  await ask('Calculate 5 + 3 - 2 + 103');
  await ask('Calculate 4 + 2 - 2 + 103');
})();
