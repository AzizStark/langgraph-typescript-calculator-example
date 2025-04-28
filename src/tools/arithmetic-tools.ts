import { DynamicStructuredTool } from '@langchain/core/tools';

export const addTool = new DynamicStructuredTool({
  name: 'add',
  description: 'Add two numbers together',
  schema: {
    type: 'object',
    properties: {
      a: { type: 'number', description: 'First number' },
      b: { type: 'number', description: 'Second number' },
    },
    required: ['a', 'b'],
  },
  func: async (input: { a: number; b: number }) => {
    console.log('Adding numbers:', input.a, input.b);
    return input.a + input.b;
  },
});

export const subtractTool = new DynamicStructuredTool({
  name: 'subtract',
  description: 'Subtract second number from first number',
  schema: {
    type: 'object',
    properties: {
      a: { type: 'number', description: 'First number' },
      b: { type: 'number', description: 'Second number' },
    },
    required: ['a', 'b'],
  },
  func: async (input: { a: number; b: number }) => {
    console.log('Subtracting numbers:', input.a, input.b);
    return input.a - input.b;
  },
});

export const multiplyTool = new DynamicStructuredTool({
  name: 'multiply',
  description: 'Multiply two numbers together',
  schema: {
    type: 'object',
    properties: {
      a: { type: 'number', description: 'First number' },
      b: { type: 'number', description: 'Second number' },
    },
    required: ['a', 'b'],
  },
  func: async (input: { a: number; b: number }) => {
    console.log('Multiplying numbers:', input.a, input.b);
    return input.a * input.b;
  },
});

export const divideTool = new DynamicStructuredTool({
  name: 'divide',
  description: 'Divide first number by second number',
  schema: {
    type: 'object',
    properties: {
      a: { type: 'number', description: 'First number (dividend)' },
      b: { type: 'number', description: 'Second number (divisor)' },
    },
    required: ['a', 'b'],
  },
  func: async (input: { a: number; b: number }) => {
    if (input.b === 0) {
      throw new Error('Cannot divide by zero');
    }
    console.log('Dividing numbers:', input.a, input.b);
    return input.a / input.b;
  },
});

// Export all tools as a collection for easy import
export const mathTools = [addTool, subtractTool, multiplyTool, divideTool];
