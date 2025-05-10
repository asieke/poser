import { BaseLLMProvider } from './base';
import { GoogleGenAI, Content, Part, GroundingMetadata } from "@google/genai";

import {
  LLMOptions,
  LLMRequestNonStreaming,
  LLMRequestStreaming,
  RequestMessage,
  ContentPart as OpenApiContentPart,
} from '../../types/llm/request';
import {
  LLMResponseNonStreaming,
  LLMResponseStreaming,
} from '../../types/llm/response';
import { LLMProvider, LLMProviderType } from '../../types/provider.types';
import { ChatModel } from '../../types/chat-model.types';

export type SiekeProviderObject = Extract<LLMProvider, { type: 'sieke' }>;

export class SiekeLLMProvider extends BaseLLMProvider<SiekeProviderObject> {

  private client: GoogleGenAI
  private apiKey: string


  constructor(provider: SiekeProviderObject) {
    super(provider);
    this.apiKey = provider.apiKey ?? ''
    this.client = new GoogleGenAI({ apiKey: this.apiKey });

    console.log(`SiekeLLMProvider constructor called with provider: ${JSON.stringify(provider)}`);
  }

  messagesToContents(messages: RequestMessage[]): Content[] {
    const googleContents: Content[] = [];

    for (const message of messages) {
      switch (message.role) {
        case 'user':
          const userParts: Part[] = [];
          if (typeof message.content === 'string') {
            userParts.push({ text: message.content });
          } else {
            for (const part of message.content) {
              if (part.type === 'text') {
                userParts.push({ text: part.text });
              }
            }
          }
          if (userParts.length > 0) {
            googleContents.push({ role: 'user', parts: userParts });
          }
          break;

        case 'assistant':
          if (message.content && typeof message.content === 'string') {
            googleContents.push({ role: 'model', parts: [{ text: message.content }] });
          }
          break;

        case 'system':
          break;

        case 'tool':
          break;
      }
    }
    return googleContents;
  }

  generateFootnoteMarkdown(input: string, grounding: GroundingMetadata | undefined): string {
    if (
      !grounding ||
      !grounding.groundingSupports ||
      grounding.groundingSupports.length === 0 ||
      !grounding.groundingChunks ||
      grounding.groundingChunks.length === 0
    ) {
      return input;
    }

    const uriToFootnoteId = new Map<string, number>();
    let nextFootnoteId = 1;
    const textParts: string[] = [];
    let lastProcessedEndIndex = 0;

    const validSupports = grounding.groundingSupports
      .filter(
        (s) =>
          s.segment &&
          typeof s.segment.startIndex === 'number' &&
          typeof s.segment.endIndex === 'number' &&
          s.segment.text
      )
      .sort((a, b) => a.segment!.startIndex! - b.segment!.startIndex!);

    for (const support of validSupports) {
      const segmentStartIndex = support.segment!.startIndex!;
      const segmentEndIndex = support.segment!.endIndex!;

      if (segmentStartIndex > lastProcessedEndIndex) {
        textParts.push(input.substring(lastProcessedEndIndex, segmentStartIndex));
      }

      const currentSegmentText = input.substring(segmentStartIndex, Math.min(segmentEndIndex, input.length));
      textParts.push(currentSegmentText);

      let segmentFootnoteMarkers = "";

      if (support.groundingChunkIndices && support.groundingChunkIndices.length > 0) {
        for (const chunkIndex of support.groundingChunkIndices) {
          if (chunkIndex >= 0 && chunkIndex < grounding.groundingChunks.length) {
            const chunk = grounding.groundingChunks[chunkIndex];
            if (chunk && chunk.web && chunk.web.uri && chunk.web.title) {
              const uri = chunk.web.uri;

              let footnoteId: number;
              if (uriToFootnoteId.has(uri)) {
                footnoteId = uriToFootnoteId.get(uri)!;
              } else {
                footnoteId = nextFootnoteId++;
                uriToFootnoteId.set(uri, footnoteId);
              }
              segmentFootnoteMarkers = `<sup><a href="${uri}">${footnoteId}</a></sup>`;
              break;
            }
          }
        }
      }
      textParts.push(segmentFootnoteMarkers);

      lastProcessedEndIndex = Math.max(lastProcessedEndIndex, segmentEndIndex);
    }

    if (lastProcessedEndIndex < input.length) {
      textParts.push(input.substring(lastProcessedEndIndex));
    }

    return textParts.join('');
  }


  async generateResponse(
    model: ChatModel,
    request: LLMRequestNonStreaming,
    _opts?: LLMOptions,
  ): Promise<LLMResponseNonStreaming> {
    console.log(`SiekeLLMProvider received messages: ${JSON.stringify(request.messages)} `);

    console.log('Contents: ', this.messagesToContents(request.messages));

    const config = {
      tools: [
        { googleSearch: {} },
      ],
      responseMimeType: 'text/plain',
      systemInstruction: [
        {
          text: `
            You are an intelligent assistant to help answer any questions that the user has, particularly about editing and organizing markdown files in Obsidian.
            1. Please keep your response as concise as possible. Avoid being verbose.
            2. Do not lie or make up facts.
            3. Format your response in markdown.
            4. Respond in the same language as the user's message.
            5. When writing out new markdown blocks, also wrap them with <smtcmp_block> tags. For example:\\n<smtcmp_block language=\\"markdown\\">\\n{{ content }}\\n</smtcmp_block>
            6. When providing markdown blocks for an existing file, add the filename and language attributes to the <smtcmp_block> tags. Restate the relevant section or heading, so the user knows which part of the file you are editing. For example:\\n<smtcmp_block filename=\\"path/to/file.md\\" language=\\"markdown\\">\\n## Section Title\\n...\\n{{ content }}\\n...\\n</smtcmp_block>
            7. When the user is asking for edits to their markdown, please provide a simplified version of the markdown block emphasizing only the changes. Use comments to show where unchanged content has been skipped. Wrap the markdown block with <smtcmp_block> tags. Add filename and language attributes to the <smtcmp_block> tags. For example:\\n<smtcmp_block filename=\\"path/to/file.md\\" language=\\"markdown\\">\\n<!-- ... existing content ... -->\\n{{ edit_1 }}\\n<!-- ... existing content ... -->\\n{{ edit_2 }}\\n<!-- ... existing content ... -->\\n</smtcmp_block>\\nThe user has full access to the file, so they prefer seeing only the changes in the markdown. Often this will mean that the start/end of the file will be skipped, but that's okay! Rewrite the entire file only if specifically requested. Always provide a brief explanation of the updates, except when the user specifically asks for just the content.
            8. Default to attempting to search Google, unless it is clear it is not a research/information/factual question.`,

        }
      ],
    };

    const response = await this.client.models.generateContent({
      model: "gemini-2.5-flash-preview-04-17",
      contents: this.messagesToContents(request.messages),
      config: config,
    });

    console.log('Response: ', response.text);
    console.log('GROUNDING: ', response?.candidates?.[0]?.groundingMetadata);

    const text = response.text || '';

    const markdown = this.generateFootnoteMarkdown(text, response?.candidates?.[0]?.groundingMetadata);

    return {
      id: `sieke - resp - ${Date.now()} `,
      created: Math.floor(Date.now() / 1000),
      object: 'chat.completion',
      model: model.model,
      choices: [
        {
          message: {
            role: 'assistant',
            content: markdown,
          },
          finish_reason: 'stop',
        },
      ],
      usage: {
        prompt_tokens: request.messages.reduce((acc, curr) => acc + (curr.content?.length || 0), 0),
        completion_tokens: text.length,
        total_tokens: request.messages.reduce((acc, curr) => acc + (curr.content?.length || 0), 0) + text.length,
      },
    };
  }

  async streamResponse(
    model: ChatModel,
    request: LLMRequestStreaming,
    opts?: LLMOptions,
  ): Promise<AsyncIterable<LLMResponseStreaming>> {
    const nonStreamingRequest: LLMRequestNonStreaming = {
      ...request,
      messages: request.messages,
      stream: false,
    };

    const fullResponse = await this.generateResponse(model, nonStreamingRequest, opts);

    const responseId = fullResponse.id;
    const responseCreated = fullResponse.created;
    const responseModelStr = fullResponse.model;
    const fullContent = fullResponse.choices[0]?.message?.content ?? "";

    async function* generateStream(): AsyncIterable<LLMResponseStreaming> {
      const chunkDelayMs = 1;

      if (fullContent === "") {
        yield {
          id: responseId,
          object: 'chat.completion.chunk',
          created: responseCreated,
          model: responseModelStr,
          choices: [{ delta: { role: 'assistant', content: "" }, finish_reason: null }],
        };
        if (chunkDelayMs > 0) await new Promise(resolve => setTimeout(resolve, chunkDelayMs));
        yield {
          id: responseId,
          object: 'chat.completion.chunk',
          created: responseCreated,
          model: responseModelStr,
          choices: [{ delta: {}, finish_reason: 'stop' }],
        };
        return;
      }

      const words = fullContent.split(' ');

      for (let i = 0; i < words.length; i++) {
        const word = words[i];
        const isFirstWord = i === 0;
        const isLastWord = i === words.length - 1;

        let contentPiece = word;
        if (!isLastWord) {
          contentPiece += ' ';
        }

        const delta: { role?: 'assistant'; content?: string } = {};
        if (isFirstWord) {
          delta.role = 'assistant';
        }
        delta.content = contentPiece;

        yield {
          id: responseId,
          object: 'chat.completion.chunk',
          created: responseCreated,
          model: responseModelStr,
          choices: [{
            delta: delta,
            finish_reason: null
          }]
        };
        if (chunkDelayMs > 0 && !isLastWord) await new Promise(resolve => setTimeout(resolve, chunkDelayMs));
      }

      yield {
        id: responseId,
        object: 'chat.completion.chunk',
        created: responseCreated,
        model: responseModelStr,
        choices: [{
          delta: {},
          finish_reason: 'stop'
        }]
      };
    }

    return generateStream();
  }

  async getEmbedding(
    _model: string,
    _text: string,
  ): Promise<number[]> {
    throw new Error('Embeddings are not supported by Sieke provider.');
  }
} 