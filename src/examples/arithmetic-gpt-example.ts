import { Annotation, StateGraph, START, END } from '@langchain/langgraph';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import { addTool, divideTool, multiplyTool, subtractTool } from '../tools/arithmetic-tools';
import { bedrockModel, openAiModel } from '../llm/models';
import { MemorySaver } from '@langchain/langgraph'; // Import the MemorySaver

// Create a memory checkpointer
const memoryCheckpointer = new MemorySaver();

// Enhance graph state to track previous results
const State = Annotation.Root({
  messages: Annotation<any[]>({
    default: () => [],
    reducer: (curr, a) => [...curr, ...a],
  }),
  result: Annotation<number | null>({
    default: () => null,
    reducer: (_, v) => v,
  }),
  previousResults: Annotation<number[]>({
    default: () => [],
    reducer: (curr, newResult) => {
      if (Array.isArray(newResult)) {
        return [...curr, ...newResult];
      }
      return newResult !== null && newResult !== undefined ? [...curr, newResult] : curr;
    },
  }),
  threadId: Annotation<string>({
    default: () => 'default-thread',
    reducer: (_, id) => id,
  }),
});

type S = (typeof State)['State'];

// Update system message to indicate memory capability
const SYS = new SystemMessage(
  'You are a calculator with memory. Use the arithmetic tools when needed. ' +
    'You can reference previous results in your calculations.'
);

// LLM node (enhanced to know about previous results)
const llmNode = async (state: S) => {
  const { messages, previousResults } = state;

  // Add context about previous results if they exist
  let contextMessages = [SYS];
  if (previousResults.length > 0) {
    const previousResultsMessage = new SystemMessage(
      `Previous calculations: [${previousResults.join(', ')}]. ` +
        `Most recent result: ${previousResults[previousResults.length - 1]}`
    );
    contextMessages.push(previousResultsMessage);
  }

  const model = openAiModel().bindTools([addTool, subtractTool, multiplyTool, divideTool]);
  const reply = await model.invoke([...contextMessages, ...messages]);

  // Bedrock quirk: strip duplicate tool_use fragments
  if (typeof reply.content !== 'string') {
    reply.content = reply.content.filter((c) => c.type !== 'tool_use');
  }

  return { messages: [reply] };
};

// Route for further tool calls or finish
const route = ({ messages }: S) => (messages.at(-1)?.tool_calls?.length ? 'tools' : 'extract');

// Fix: Correctly typed extract node
const extractNode = (state: S) => {
  const { messages } = state;
  const toolMsg = [...messages]
    .reverse()
    .find(
      (m) =>
        m.tool_call_id ||
        m.name === 'add' ||
        m.name === 'subtract' ||
        m.name === 'multiply' ||
        m.name === 'divide'
    );

  const result = Number(toolMsg?.content ?? NaN);

  // Only add valid results to history
  if (!isNaN(result)) {
    return {
      result,
      previousResults: [result],
    };
  }

  return { result };
};

// Build & compile graph with checkpointer
const calculator = new StateGraph(State)
  .addNode('llm', llmNode)
  .addNode('tools', new ToolNode([addTool, subtractTool, multiplyTool, divideTool]))
  .addNode('extract', extractNode)
  .addEdge(START, 'llm')
  .addConditionalEdges('llm', route, { tools: 'tools', extract: 'extract' })
  .addEdge('tools', 'llm')
  .addEdge('extract', END)
  .compile({
    checkpointer: memoryCheckpointer,
  });

// Enhanced example usage with persistent session
(async () => {
  // Create a unique thread ID for this session
  const threadId = 'calc-thread-' + Date.now();

  const ask = async (q: string) => {
    // Important: Include the thread_id in the configurable options
    const config = {
      configurable: {
        thread_id: threadId, // This is the key fix - thread_id, not threadId
      },
    };

    const state = await calculator.invoke(
      {
        messages: [new HumanMessage(q)],
        threadId, // This goes into our state
      },
      config // This contains the thread_id for the checkpointer
    );

    console.log(`\nUser        → ${q}`);

    const llmMsg = state.messages.at(-1);
    console.log('Assistant   →', llmMsg.content);

    const toolMsgs = state.messages.filter((m) => m.name === 'add' || m.name === 'subtract');
    for (const t of toolMsgs) {
      console.log(`Tool:${t.name.padEnd(8)}→ ${t.content}`);
    }

    console.log('Result      →', state.result);
    console.log('Previous Results →', state.previousResults);
    console.log('────────────────────────────────');

    return state;
  };

  await ask('Calculate 24 * 3 + 15');
  await ask('Divide the previous result by 9');
  await ask('Multiply this value by itself');
  await ask('Subtract 48 from the result');
  await ask('Add the square root of 225 to this number');
  // correct answer should be 60.44444444444443
})();
