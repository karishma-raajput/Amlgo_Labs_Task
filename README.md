# Amlgo_Labs_Task

RAG Chatbot - eBay User Agreement QA Assistant
This is a Retrieval - Augmented Generation (RAG) Chatbot designed to answer questions based on a long document- the eBay User Agreemnt. It uses document chunking, sementic embeddings, FAISS for retrieval, and GPT-3.5  to generate factual response grounded in the document.

PDF -> clean & chunks -> Embed chunks -> store in FAISS -> User Query -> Retrieve Top K chunks -> GPT-3.5 Response

##stpes to run
#install Dependencies

2. Preprocess & Chunks PDF
   Load and cleean the PDF using PyMuPDF
   Tokenize into sentences with NLTK
   Group  sentences into chunks(100-300 words)

   #code
   doc= fitz.open("AI Training Document.pdf")
   text= extract_clean_text(doc)
   chunks = chunks_sentences(text)

3 Generate Embeddings and Build FAISS Index

#code
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks,show_progress_bar=True)
index = faiss.read_index("faiss_index.index")
with open("faiss_chunks.pkl","rb") as f:
  chunks = pickle.load(f)


4 Run Retrieval + GPT Query
index = faiss.read_index("faiss_index.index")
with open("faiss_chunks.pkl","rb") as f:
  chunks = pickle.load(f)


  def ask_anything(query, k=3):
  print(f"\n Query : {query}")
  query_vec= model.encode([query]).astype("float32")
  D,I= index.search(query_vec,k)
  context = "\n".join([chunks[i] for i in I[0]])
  prompt= f"""Hi, I am your ebay assistant bot. I've read the user agreement carefully.
Using the information below, I'll do my best to answer your question clearly and If the answer isn't there,
say "SORRY, I could not find the answer in the document"

Context:
{context}

Question :
{query}

Answer:"""

  response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[{"role": "user", "content": prompt}],
      temperature=0.2
  )

  answer = response['choices'][0]['message']['content']

  print("\nAnswer:")
  print(answer)
  print("\nContext Used:")
  print(context[:1000])


ask_anything("Can I tranfer my eBay account to somewhere else ?")


NOTE L Due to OpenAI linitation, funal GPT response were not genrated live


LIMITATION and NOTES
1 GPT response couldn't be generated in the final run due to OpenAI limitation.
2 Retrieval and prompt logic are fully functional and tested.
3 Hallucination are reduced through prompt intructions and fallback responses
4 Real time straming and UI not Implmented due to time + quota but logic is ready.


Folder Structure
Amlgo _Labs_Task.ipynb     -> Main Notebook with all logic
AI Task.pdf                -> Assignemnt Instruction
AI Training Document.pdf   -> Input Document
faiss_chunks.pkl           -> Stored Chunks
faiss_index.index           -> Vector Index
requirement.text            -> Requirement Package
README.md                  ->  This file
