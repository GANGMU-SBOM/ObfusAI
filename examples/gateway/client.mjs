// JavaScript / TypeScript: the official SDKs, protected by changing only the base URL.
//   npm i openai @anthropic-ai/sdk
//   node client.mjs            (GATEWAY defaults to http://127.0.0.1:8787)
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";

const GATEWAY = process.env.GATEWAY ?? "http://127.0.0.1:8787";
const SYSTEM = "Rule: BUY when edge_bps > 35. Codename project_falcon.";
const TOOLS = [{ type: "function", function: { name: "get_signal", description: "Reads edge_bps for a ticker.",
  parameters: { type: "object", properties: { metric: { type: "string" } } } } }];
const messages = [{ role: "system", content: SYSTEM }, { role: "user", content: "go" }];

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY ?? "demo", baseURL: `${GATEWAY}/v1` });

const reply = await openai.chat.completions.create({ model: "gpt-4o", messages, tools: TOOLS });
const call = reply.choices[0].message.tool_calls[0].function;
console.log("openai chat      :", reply.choices[0].message.content, "| tool args:", call.arguments);

let streamed = "";
for await (const chunk of await openai.chat.completions.create({ model: "gpt-4o", messages, tools: TOOLS, stream: true })) {
  streamed += chunk.choices[0]?.delta?.content ?? "";
}
console.log("openai chat sse  :", streamed);

const resp = await openai.responses.create({ model: "gpt-4o", instructions: SYSTEM, input: "go" });
console.log("openai responses :", resp.output_text);

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY ?? "demo", baseURL: GATEWAY });
const msg = await anthropic.messages.create({
  model: "claude-sonnet-5-5", max_tokens: 100, system: SYSTEM, messages: [{ role: "user", content: "go" }],
  tools: [{ name: "get_signal", description: "Reads edge_bps for a ticker.", input_schema: { type: "object", properties: {} } }],
});
console.log("anthropic        :", msg.content[0].text, "| tool input:", JSON.stringify(msg.content[1].input));
const stream = anthropic.messages.stream({ model: "claude-sonnet-5-5", max_tokens: 100, system: SYSTEM, messages: [{ role: "user", content: "go" }] });
let text = "";
for await (const event of stream) if (event.type === "content_block_delta" && event.delta.type === "text_delta") text += event.delta.text;
console.log("anthropic sse    :", text);
