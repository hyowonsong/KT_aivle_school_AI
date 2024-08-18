# 🌟 에이블스쿨 QA Chatbot 서비스(7차 미니프로젝트)

---

## **📊 프로젝트 개요**

- **목적 :  에이블스쿨 지원자들을 위한 QA Chatbot 서비스** **개발**
- **기간 :** 2024.06.03 ~ 2024.06.13
- **사용도구**
    - **AI 모델**: LLM, RAG 기반 챗봇 모델
    - **H/W**: 클라우드 웹 서버
    - **DB**: 벡터 DB on SQLite3
    - **S/W**: 장고 웹 프레임워크

---

## 🖥 데이터 수집 및 요구사항

### 1. 데이터 수집

### 기본

- **에이블스쿨 홈페이지 Q&A**:
    - 홈페이지 FAQ 데이터를 수집하여 데이터셋 구성
    - 질문과 답변을 하나의 chunk로 구성

### 추가

- **데이터 추가 수집**:
    - 에이블러들의 블로그 글 및 기타 의견 등
    - 지원자 소개 글에서 내용 chunk로 구성

### DB 구성

- **RAG 용 Vector DB** 및 **사용 기록 DB** 구성

---

### 2. 기능 요구사항: 질문 답변 기능

### 기본

- **질문 답변 기능**:
    - **모델 사용**: OpenAI LLM (gpt-3.5-turbo), Embedding (text-embedding-ada-002)
    - **Vector DB**를 Retriever로 이용
    - **단발성 질문 답변**

### 추가

- **대화가 이어지는 기능**:
    - 대화 흐름 유지 및 다양한 시도 통해 정확한 답변 제공

---

## 🛠️ **기술 구현**

### 1. 환경 준비

### (1) 라이브러리 Import

```python
import pandas as pd
import numpy as np
import os
import sqlite3
import openai

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

```

### (2) OpenAI API Key 확인

- **환경변수**에서 키 불러오기

```python
api_key = os.getenv('OPENAI_API_KEY')
print(api_key)
```

### 2. Vector DB 만들기

### 데이터 로딩

- 에이블스쿨 홈페이지 FAQ 데이터 CSV로 저장
- **DataFrame**으로 로딩

```python
data = pd.read_csv('aivleschool_qa.csv', encoding='utf-8')
data.head()
```

### 벡터 데이터베이스

- **Embedding 모델**: text-embedding-ada-002
- **DB 경로**: ./database

```python
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
database = Chroma(persist_directory="./database", embedding_function=embeddings)
```

### 데이터 입력

- **문서 객체로 변환 및 추가**

```python
documents = [Document(page_content=text) for text in data['QA'].tolist()]
database.add_documents(documents)
```

### 입력된 데이터 조회

```python
database.get()
```

### 3. RAG + LLM 모델

### 모델 선언: ConversationalRetrievalChain

- **LLM 모델**: gpt-3.5-turbo
- **Retriever**: 벡터 DB
- **Memory**: ConversationBufferMemory

```python
chat = ChatOpenAI(model="gpt-3.5-turbo")
k = 3
retriever = database.as_retriever(search_kwargs={"k": k})

memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", output_key="answer", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(llm=chat, retriever=retriever, memory=memory, return_source_documents=True, output_key="answer")
```

### 모델 사용하기

- **질문 예시:**

```python
query = '지원하는데 나이 제한이 있나요?'
result = qa(query)
print(result["answer"])
```

### 4. Retrieval Augmented Generation (RAG)

### 데이터베이스 구성 및 입력

- **Chroma DB 구성**

```python
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
database = Chroma(persist_directory="./data", embedding_function=embeddings)
```

- **문서 추가 및 조회**

```python
input_list = ['test 데이터 입력1', 'test 데이터 입력2']
ind = database.add_texts(input_list)
database.get(ind)
```

### 유사도 높은 문서 조회

- **유사도 점수와 문서 내용**

```python
query = "오늘 낮 기온은?"
k = 3
result = database.similarity_search_with_score(query, k=k)
for doc in result:
    print(f"유사도 점수: {round(doc[1], 5)}, 문서 내용: {doc[0].page_content}")
```

### 질문에 대한 답변 받기

- **RetrievalQA 사용**

```python
chat = ChatOpenAI(model="gpt-3.5-turbo")
retriever = database.as_retriever()
qa = RetrievalQA.from_llm(llm=chat, retriever=retriever, return_source_documents=True)

result = qa(query)
print(result["result"])
```

---

## 👀 서비스 화면

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/36358b89-fde5-4b16-95d9-7decef74047e/3a49c456-627f-4e27-b5c9-cafc2a458d49/image.png)

- 홈페이지 화면으로 CSV 업로드, Vector DB 확인, 채팅 기록 확인, 회원 가입, 로그인 기능으로 이동이 가능합니다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/36358b89-fde5-4b16-95d9-7decef74047e/114200bf-e7c6-4cc5-8b6d-beed60c43939/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/36358b89-fde5-4b16-95d9-7decef74047e/4154c1a8-6c78-40c3-a429-c3b60121c512/image.png)

- vector db 기록 확인 및 상담 내용이 기록됩니다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/36358b89-fde5-4b16-95d9-7decef74047e/f356c435-c5ef-4bd1-9c6a-bd66c90eb431/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/36358b89-fde5-4b16-95d9-7decef74047e/73daf8ed-e5ad-4899-8661-d18cbb1fba2c/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/36358b89-fde5-4b16-95d9-7decef74047e/72e2be51-9268-43e1-8d5e-a7193d0e826b/image.png)

- 회원 가입 및 로그인 후 챗봇 사용이 가능합니다.
