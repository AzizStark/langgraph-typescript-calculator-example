import { Annotation, StateGraph, START, END, messagesStateReducer } from '@langchain/langgraph';
import {
  HumanMessage,
  SystemMessage,
  RemoveMessage,
  isSystemMessage,
  isAIMessage,
} from '@langchain/core/messages';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import { addTool, subtractTool, multiplyTool, divideTool } from '../tools/arithmetic-tools';
import { bedrockModel, openAiModel } from '../llm/models';
import { MemorySaver } from '@langchain/langgraph';
import dotenv from 'dotenv';
dotenv.config();

// Core configuration
interface MessageSummaryConfig {
  maxMessages: number;
  retainLastN: number;
}

const DEFAULT_SUMMARY_CONFIG: MessageSummaryConfig = {
  maxMessages: 7,
  retainLastN: 2,
};

// State definition
const State = Annotation.Root({
  messages: Annotation<any[]>({
    default: () => [],
    reducer: messagesStateReducer,
  }),
  result: Annotation<number | null>({
    default: () => null,
    reducer: (_, v) => v,
  }),
  conversationSummary: Annotation<string | null>({
    default: () => null,
    reducer: (_, v) => v,
  }),
});

type S = (typeof State)['State'];
type CalcConfig = {
  configurable?: {
    thread_id?: string;
    trace?: any;
  };
};

const tools = [addTool, subtractTool, multiplyTool, divideTool];

const isToolMessage = (message: any): boolean => {
  return message.tool_call_id || tools.map((t) => t.name).includes(message.name);
};

const buildSummarizeNode = (config: MessageSummaryConfig = DEFAULT_SUMMARY_CONFIG) => {
  return async (state: S, runnableConfig?: CalcConfig) => {
    const { messages } = state;
    if (!messages || messages.length <= config.maxMessages) {
      return {};
    }

    console.log(`[Calculator] ========== Summarizing ${messages.length} messages...`);

    const currentSummary = state.conversationSummary || '';
    const summaryPrompt = currentSummary
      ? `This is the summary of the calculations so far: ${currentSummary}\n\nExtend the summary by taking into account the new calculations above, showing the mathematical steps and results:`
      : 'Create a concise summary of the calculations performed above, showing the mathematical steps and results:';

    // Create sanitized messages without tool content
    const sanitizedMessages = messages.map((msg) => {
      if (msg.tool_call_id || tools.map((t) => t.name).includes(msg.name)) {
        return new HumanMessage(`Calculation result: ${msg.name || 'operation'} = ${msg.content}`);
      }

      if (msg.tool_calls) {
        return new HumanMessage(
          `Requested calculation: ${
            typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)
          }`
        );
      }

      if (isSystemMessage(msg)) {
        return msg;
      }

      return new HumanMessage(
        typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)
      );
    });

    sanitizedMessages.push(new HumanMessage(summaryPrompt));

    const model = openAiModel();
    const response = await model.invoke(sanitizedMessages);
    const newSummary =
      typeof response.content === 'string' ? response.content : JSON.stringify(response.content);

    // Determine which messages to keep
    let keepStartIndex = messages.length - config.retainLastN;
    let systemPromptOffset = 0;

    if (messages.length > 0 && isSystemMessage(messages[0])) {
      systemPromptOffset = 1;
      keepStartIndex = Math.max(1, keepStartIndex);
    }

    if (keepStartIndex > systemPromptOffset) {
      const firstKeptMessage = messages[keepStartIndex];
      if (isToolMessage(firstKeptMessage)) {
        for (let i = keepStartIndex - 1; i >= systemPromptOffset; i--) {
          if (isAIMessage(messages[i])) {
            keepStartIndex = i;
            break;
          }
        }
      }
    }

    const deleteMessages = [];
    for (let i = systemPromptOffset; i < keepStartIndex; i++) {
      if (messages[i] && messages[i].id) {
        deleteMessages.push(new RemoveMessage({ id: messages[i].id }));
      }
    }

    return {
      conversationSummary: newSummary,
      messages: deleteMessages,
    };
  };
};

const llmNode = async (state: S, runnableConfig?: CalcConfig) => {
  let messages = state.messages;

  // Handle conversation summary if available
  if (state.conversationSummary) {
    if (isSystemMessage(messages[0])) {
      messages = [
        messages[0],
        new HumanMessage(
          `Summary of the calculations before the following messages: ${state.conversationSummary}`
        ),
        ...messages.slice(1),
      ];
    } else {
      messages = [
        new SystemMessage(
          `You are a calculator assistant. Summary of previous calculations: ${state.conversationSummary}`
        ),
        ...messages,
      ];
    }
  } else if (!messages.some(isSystemMessage)) {
    messages = [
      new SystemMessage(
        'You are a calculator assistant that performs calculations step by step using the available arithmetic tools.'
      ),
      ...messages,
    ];
  }

  const model = openAiModel().bindTools(tools);
  const response = await model.invoke(messages);

  // Fix for Bedrock's duplicate tool_use entries
  if (Array.isArray(response.content)) {
    response.content = response.content.filter((c) => c.type !== 'tool_use');
  }

  return { messages: [response] };
};

const route = (state: S) => {
  const lastMessage = state.messages.at(-1);
  return lastMessage?.tool_calls?.length ? 'tools' : 'extract';
};

const extractNode = ({ messages }: S) => {
  const toolNames = tools.map((t) => t.name);
  const lastTool = [...messages]
    .reverse()
    .find((m) => m.tool_call_id || toolNames.includes(m.name));

  const num = lastTool ? Number(lastTool.content) : NaN;
  return { result: isNaN(num) ? null : num };
};

// Build the graph
const calculator = new StateGraph(State)
  .addNode('summarize', buildSummarizeNode())
  .addNode('llm', llmNode)
  .addNode('tools', new ToolNode(tools))
  .addNode('extract', extractNode)
  .addEdge(START, 'summarize')
  .addEdge('summarize', 'llm')
  .addConditionalEdges('llm', route, { tools: 'tools', extract: 'extract' })
  .addEdge('tools', 'summarize')
  .addEdge('extract', END)
  .compile({ checkpointer: new MemorySaver() });

// Calculator session creator
export const createCalculatorSession = (threadId = 'calculator-session') => {
  const calculate = async (query: string) => {
    const state = await calculator.invoke(
      { messages: [new HumanMessage(query)] },
      { configurable: { thread_id: threadId } }
    );

    console.log(`User        → ${query}`);

    const llmMsg = state.messages.at(-1)!;
    console.log(
      'Assistant   →',
      typeof llmMsg.content === 'string' ? llmMsg.content : JSON.stringify(llmMsg.content)
    );

    // Log tool usage
    const toolNames = tools.map((t) => t.name);
    const toolMsgs = state.messages.filter((m) => m.tool_call_id || toolNames.includes(m.name));

    for (const tool of toolMsgs) {
      console.log(`Tool:${(tool.name || '').padEnd(8)}→ ${tool.content}`);
    }

    console.log('Result      →', state.result);
    if (state.conversationSummary) {
      console.log('Summary     →', state.conversationSummary);
    }
    console.log('Message count →', state.messages.length);
    console.log('─'.repeat(50));

    return state;
  };

  return calculate;
};

// Example usage
(async () => {
  try {
    const calculate = createCalculatorSession();

    await calculate(
      'Perform the following operations step by step: calculate 24 × 3 + 15; divide the result by 9; square that result; subtract 48. Consider the result as A'
    );
    await calculate(
      'Mutiple 2 by 22; add the square root of 225; add 124 and multiply the sum by 2; finally, divide the result by 3. Consider the result as B'
    );
    await calculate(
      'Add 400 to 323, divide the result by 2, and subtract 100. Consider the result as C'
    );
    await calculate('Add A, B, and C;');
  } catch (error) {
    console.error('Error in calculation sequence:', error);
  }
})();
