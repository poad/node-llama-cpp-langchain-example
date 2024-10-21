import { ChatLlamaCpp } from "@langchain/community/chat_models/llama_cpp";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
// import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { ChatPromptTemplate, PromptTemplate } from "@langchain/core/prompts";
import {
  type BaseMessage,
} from "@langchain/core/messages";
import { RecursiveUrlLoader } from "@langchain/community/document_loaders/web/recursive_url";
import { compile } from "html-to-text";

export async function run() {
  const embeddings = new HuggingFaceTransformersEmbeddings({
    modelName: "nomic-ai/nomic-embed-text-v1.5-GGUF",
  });

  const vectorstore = new MemoryVectorStore(embeddings);

  const compiledConvert = compile({ wordwrap: 130 }); // returns (text: string) => string;

  console.info('Loading HTMLs');
  const loader = new RecursiveUrlLoader("https://langchain.com/", {
    extractor: compiledConvert,
    maxDepth: 1,
    excludeDirs: ["/docs/api/"],
  });

  const docs = await loader.load();

  console.info('Split chunks');

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  });

  const splitDocs = await splitter.splitDocuments(docs);
  console.info('Add to vector store');
  await vectorstore.addDocuments(splitDocs);

  const model = new ChatLlamaCpp({ modelPath: "./models/RakutenAI-7B-chat-q2_K.gguf" });
  const responseChainPrompt = ChatPromptTemplate.fromMessages<{
    context: string;
    chat_history: BaseMessage[];
    question: string;
  }>([
    ["system", `You are an experienced researcher, expert at interpreting and answering questions based on provided sources. Using the provided context, answer the user's question to the best of your ability using the resources provided.
Generate a concise answer for a given question based solely on the provided search results. You must only use information from the provided search results. Use an unbiased and journalistic tone. Combine search results together into a coherent answer. Do not repeat text.
If there is nothing in the context relevant to the question at hand, just say "Hmm, I'm not sure." Don't try to make up an answer.
Anything between the following \`context\` html blocks is retrieved from a knowledge bank, not part of the conversation with the user.
<context>
{context}
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm not sure." Don't try to make up an answer. Anything between the preceding 'context' html blocks is retrieved from a knowledge bank, not part of the conversation with the user.`],
    ["placeholder", "{chat_history}"],
    ["user", `{input}`],
  ]);

  const documentChain = await createStuffDocumentsChain({
    llm: model,
    prompt: responseChainPrompt,
    documentPrompt: PromptTemplate.fromTemplate(
      `<doc>\n{page_content}\n</doc>`,
    ),
  });

  // const historyAwarePrompt = ChatPromptTemplate.fromMessages([
  //   ["placeholder", "{chat_history}"],
  //   ["user", "{input}"],
  //   [
  //     "user",
  //     "Given the above conversation, generate a natural language search query to look up in order to get information relevant to the conversation. Do not respond with anything except the query.",
  //   ],
  // ]);

  // const historyAwareRetrieverChain = await createHistoryAwareRetriever({
  //   llm: model,
  //   retriever: vectorstore.asRetriever(),
  //   rephrasePrompt: historyAwarePrompt,
  // });

  const retrievalChain = await createRetrievalChain({
    combineDocsChain: documentChain,
    retriever: vectorstore.asRetriever(),
  });

  console.info('start RAG invoke');
  const events = await retrievalChain.streamEvents({ input: "RAG を作りたい。" }, { version: "v2" });
  for await (const ev of events) {
    const eventType = ev.event;
    // console.info(JSON.stringify({eventType}));
    if (eventType === 'on_chat_model_stream') {
      console.info(ev.data.chunk ?? '');
    }
  }
}

await run();