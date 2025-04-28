# TypeScript Starter with dotenv

A TypeScript starter project with environment variable support using dotenv.

## Environment Variables

This project uses `dotenv` to manage environment variables. To set up your environment:

1. Copy the example environment file:
   ```
   cp .env.example .env
   ```

2. Edit the `.env` file and add your actual API keys and configuration:
   ```
   OPENAI_API_KEY=your-actual-api-key-here
   ```

The `.env` file is excluded from version control to keep your sensitive information private.

## Available Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key for accessing language models |

## Running the Project

1. Install dependencies:
   ```
   npm install
   ```

2. Run the TypeScript code:
   ```
   npm start
   ```

## Development

- Environment variables are loaded automatically when the application starts
- Add new environment variables to both `.env` and `.env.example` (without real values)
- Always access environment variables via `process.env.VARIABLE_NAME`
