import { Annotation, StateGraph, START, END, messagesStateReducer } from '@langchain/langgraph';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import { addTool, subtractTool, multiplyTool, divideTool } from '../tools/arithmetic-tools';
import { openAiModel } from '../llm/models';
import { MemorySaver } from '@langchain/langgraph';
import dotenv from 'dotenv';
dotenv.config();

// Configuration constants
const TOOLS = [addTool, subtractTool, multiplyTool, divideTool];
const SUMMARIZE_EVERY = 4;
const KEEP_LAST_N = 2;

type Message = any;

// State definition
const State = Annotation.Root({
  messages: Annotation<Message[]>({ default: () => [], reducer: messagesStateReducer }),
  result: Annotation<number | null>({ default: () => null, reducer: (_, v) => v }),
  summary: Annotation<string | null>({ default: () => null, reducer: (_, v) => v }),
  count: Annotation<number>({ default: () => 0, reducer: (prev) => prev + 1 }),
});
type S = (typeof State)['State'];

// Utility: simplify messages for summarization
const simplifyMessage = (msg: Message) => {
  const toolNames = TOOLS.map((t) => t.name);
  if (msg.tool_call_id || toolNames.includes(msg.name)) {
    return new HumanMessage('A calculation was performed.');
  }
  if (msg.tool_calls?.length) {
    return new HumanMessage('The assistant requested a calculation.');
  }
  if (typeof msg.content === 'string') {
    return new HumanMessage(msg.content);
  }
  if (Array.isArray(msg.content)) {
    const text = msg.content
      .filter((i: any) => i.type === 'text')
      .map((i: any) => i.text)
      .join('\n');
    return new HumanMessage(text || 'Complex message without text');
  }
  return new HumanMessage('Message with unknown format');
};

// Node: summarization logic
async function summaryNode({ messages, summary, count }: S) {
  if (count % SUMMARIZE_EVERY === 0 && count > 0 && messages.length > KEEP_LAST_N) {
    const sanitized = messages.map(simplifyMessage);
    try {
      const resp = await openAiModel().invoke([
        new SystemMessage('Summarize the conversation in 2-3 concise sentences.'),
        ...sanitized,
      ]);

      const summaryText = typeof resp.content === 'string' ? resp.content : '';
      console.log(`[Calculator] Summary generated: ${summaryText}`);

      const pruned = messages.slice(-KEEP_LAST_N);
      console.log(
        `[Calculator] 🔪 Pruned messages: before=${messages.length}, after=${pruned.length}`
      );

      return {
        summary: summaryText,
        messages: pruned,
        count: 0,
      };
    } catch (err) {
      console.error('[Calculator] Error generating summary', err);
      return { count };
    }
  }

  return { count };
}

// Node: main LLM interaction
async function llmNode({ messages, summary }: S) {
  const sysParts = ['You are a calculator.'];
  if (summary) sysParts.push(`Previous conversation: ${summary}`);
  const ctx = [new SystemMessage(sysParts.join('\n\n')), ...messages];

  const model = openAiModel().bindTools(TOOLS);
  const reply = await model.invoke(ctx);

  if (Array.isArray(reply.content)) {
    const validSet = new Set(TOOLS.map((t) => t.name));
    reply.content = reply.content.filter(
      (c: any) => c.type !== 'tool_use' || validSet.has(c.tool_name)
    );
  }
  return { messages: [reply] };
}

// Node: extract numeric result (and carry full state forward)
function extractNode({ messages, summary, count }: S) {
  const toolNames = TOOLS.map((t) => t.name);
  const last = [...messages].reverse().find((m) => m.tool_call_id || toolNames.includes(m.name));
  const num = Number(last?.content);
  return {
    result: isNaN(num) ? null : num,
    messages,
    summary,
    count,
  };
}

// Build and export the graph
export const calculator = new StateGraph(State)
  .addNode('summarize', summaryNode)
  .addNode('llm', llmNode)
  .addNode('tools', new ToolNode(TOOLS))
  .addNode('extract', extractNode)
  .addEdge(START, 'summarize')
  .addEdge('summarize', 'llm')
  .addConditionalEdges('llm', (s) => (s.messages.at(-1)?.tool_calls?.length ? 'tools' : 'extract'))
  .addEdge('tools', 'summarize')
  .addEdge('extract', END)
  .compile({ checkpointer: new MemorySaver() });

// Convenience session creator
export const createSession = (thread: string) => {
  const ask = async (q: string) => {
    const state = await calculator.invoke(
      { messages: [new HumanMessage(q)] },
      { configurable: { thread_id: thread } }
    );

    console.log(`→ After turn: messages.length=${state.messages.length}, count=${state.count}`);
    console.log(`User → ${q}`);
    const msg = state.messages.at(-1)!;
    console.log('Assistant →', msg.content);
    console.log('Result →', state.result);
    if (state.summary) console.log('Summary →', state.summary);
    console.log('─'.repeat(50));

    return state;
  };
  return ask;
};

// Example usage
(async () => {
  const ask = createSession('calc-session');
  await ask(
    'Perform the following operations step by step: calculate 24 × 3 + 15; divide the result by 9; square that result; subtract 48.'
  );
  await ask(
    'Now take that value: add the square root of 225; add 124 and multiply the sum by 2; finally, divide the result by 3.'
  );
})();
