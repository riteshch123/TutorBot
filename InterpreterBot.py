from dotenv import load_dotenv
import os
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import logging
import requests
import json
import argparse
import TutorBot
# Set up logging
logging.basicConfig(filename='logs/interpreter_bot.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
API_KEY = os.getenv("API_KEY")

logging.info("InterpreterBot application started.")

# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

logging.info("Session states initialized.")

# Define function to get input


def interpeterBot(previous_messages: list, userMessage: str):

    template = (
        "You are an interpreter to a conversation between user and assistant. Provided is the conversation between both. Your job is to understand the intent of the user while he is asking question and provide relevant semantically related keywords. "
    )

    system_message_prompt = SystemMessagePromptTemplate.from_template(
        template)
    human_template = 'Now, your job as an assistant is to interpret the entire conversation, and understand the intent of the conversation and the question and then provide the semantically relevant keywords to query the database which are more relevant to the question. Ensure to provide ONLY the keywords in your response. The question is "{userMessage}"'
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        human_template)

    # create previous messages from the provided list
    previous_message_prompts = []
    for message in previous_messages:
        if message['role'] == 'user':  # this is a user message
            previous_message_prompts.append(
                HumanMessagePromptTemplate.from_template(message['content']))
        elif message['role'] == 'assistant':  # this is an assistant message
            previous_message_prompts.append(
                AIMessagePromptTemplate.from_template(message['content']))

    # add the system message and the user message to the list
    previous_message_prompts += [system_message_prompt,
                                 human_message_prompt]

    chat_prompt = ChatPromptTemplate.from_messages(
        previous_message_prompts)

    response = chat(chat_prompt.format_prompt(
        userMessage=userMessage).to_messages())
    return response.content


previous_messages = [{
    'role': 'assistant',
    'content': 'Photosynthesis is essentially the life-sustaining process by which green plants, algae, and some bacteria convert sunlight into energy. This process involves the use of sunlight to produce oxygen and stored chemical energy in the form of glucose, which is a type of sugar.Here\'s a quick run-down of the process: The plant takes in carbon dioxide and water from its surroundings. In the plant cells, the water loses electrons and transforms into oxygen, and the carbon dioxide gains electrons, turning into glucose. The plant then releases the oxygen back into the air, while the energy is stored within the glucose molecules for later use.The magic behind this transformation is a pigment inside plant cells called chlorophyll, which absorbs sunlight and gives plants their green color. The entire photosynthesis process is brimming with fascinating details. Here are some things we could explore further: - Would you like to discuss in-depth the role of chlorophyll in photosynthesis?- Are you interested in understanding more about the light-dependent and light-independent reactions in photosynthesis?- Or would you like to discuss the different types of photosynthesis, like C3 and C4 photosynthesis?'
},
    {
        'role': 'user',
        'content': 'ok, got it'
}]


st.title("InterpreterBot by CustomGPT")

logging.info(f"API running")
chat = ChatOpenAI(model_name='gpt-3.5-turbo',
                  openai_api_key=API_KEY,
                  temperature=0)

# Get input
input = st.text_input(
    "Input:", value=st.session_state["input"], key="input1")

# Generate response
if input:
    logging.info(f"User input received: {input}.")
    response = interpeterBot(previous_messages, input)
    logging.info(f"Generated response: {response}.")
    st.session_state.generated.append(response)
    st.session_state.past.append(input)

with st.expander("Conversation"):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(f"Input: {st.session_state.past[i]}")
        st.info(f"Response: {st.session_state['generated'][i]}")
        st.write("")
        logging.debug(
            f"Displayed conversation - Input: {st.session_state.past[i]}, Response: {st.session_state['generated'][i]}.")


def query_api(text_query):
    '''
    text_query = interpeterBot(previous_messages, input)
    '''
    logging.info('Initiating query to the API with text: %s', text_query)
    url = 'https://customplugin.customplugin.ai/query'
    headers = {
        'Content-Type': 'application/json',
    }
    data = {
        '''"text": interpeterBot.response.content'''
        "text": text_query
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    logging.info('API responded with status code: %s', response.status_code)

    if response.status_code == 200:

        logging.info('API responded successfully with a 200 status code.')
        response_data = response.json()
        context = response_data.get('context', '')

        start_marker = "--START OF CONTEXT--"
        end_marker = "--END OF CONTEXT--"

        start_index = context.find(start_marker) + len(start_marker)
        end_index = context.find(end_marker)

        context = context[start_index:end_index].strip()
        return f'{context}'

    else:
        logging.error('API request failed with status code: %s',
                      response.status_code)
        return f'Request failed with status code: {response.status_code}'


interpreter_response = interpeterBot(
    previous_messages, input)


def model():
    previous_messages = [{
        'role': 'assistant',
        'content': 'Photosynthesis is essentially the life-sustaining process by which green plants, algae, and some bacteria convert sunlight into energy. This process involves the use of sunlight to produce oxygen and stored chemical energy in the form of glucose, which is a type of sugar.Here\'s a quick run-down of the process: The plant takes in carbon dioxide and water from its surroundings. In the plant cells, the water loses electrons and transforms into oxygen, and the carbon dioxide gains electrons, turning into glucose. The plant then releases the oxygen back into the air, while the energy is stored within the glucose molecules for later use.The magic behind this transformation is a pigment inside plant cells called chlorophyll, which absorbs sunlight and gives plants their green color. The entire photosynthesis process is brimming with fascinating details. Here are some things we could explore further: - Would you like to discuss in-depth the role of chlorophyll in photosynthesis?- Are you interested in understanding more about the light-dependent and light-independent reactions in photosynthesis?- Or would you like to discuss the different types of photosynthesis, like C3 and C4 photosynthesis?'
    },
        {
        'role': 'user',
        'content': 'ok, got it'
    }]

    context_from_api = query_api(interpreter_response)

    tutor_response = TutorBot.tutorBot(context_from_api, input)

    return tutor_response


result = model()
print(result)
