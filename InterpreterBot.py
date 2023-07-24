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


def interpreterBot(previous_messages: list, userMessage: str):

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
    # Convert response to comma-separated keywords
    keywords = ", ".join(
        [word for word in response.content.split() if len(word) > 2])
    return keywords


previous_messages = [{
    'role': 'assistant',
    'content': 'Hi, How are you doing today?'
},
    {
        'role': 'user',
        'content': 'doing good.'
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
    response = interpreterBot(previous_messages, input)
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

    logging.info('Initiating query to the API with text: %s', text_query)
    url = 'https://levinbot.customplugin.ai/query'
    headers = {
        'Content-Type': 'application/json',
    }
    data = {
        "text": text_query
    }

    try:

        response = requests.post(url, headers=headers, data=json.dumps(data))
        logging.info('API responded with status code: %s',
                     response.status_code)

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

    except Exception as e:
        logging.error('API request failed with exception: %s',
                      response.status_code)
        return f'Request failed with exception: {response.status_code}'


def model():

    user_input = input
    response = None

    if user_input:
        previous_messages = [{'role': 'assistant', 'content': 'Hi, How are you doing today?'},
                             {'role': 'user', 'content': 'doing good.'}]

        # Step 1: Use InterpreterBot to get keywords
        keywords = interpreterBot(previous_messages, user_input)

        # Step 2: Get the context from the API using the keywords
        context = query_api(keywords)

        # Step 3: Use TutorBot to generate a response using the context and user's input
        systemMessage = '''

        You are an AI tutor and your expertise is based on the data provided. The user is someone who wants a deeper understanding
        of the data.

        Rules:
        1. Being an AI tutor, your tone should be conversational, and not like a conventional Q&A bot.
        2. Always you need to answer in 1st person.
        3. The inital response that you generate needs to have 3 additional questions at the end in the form of a numbered bulleted list,
        to keep the conversation with the user and to ensure that the user has choice.
        4. For every further response, you need to include 1 follow up topic in a conversational tone, based on the response you provided,
        to keep the conversation flowing.
        5. If the user is unwilling to go ahead with the follow up topic or if he is confused, acknowledge it and provide 3 additional questions
        for the user to choose, from the content provided.
        6. You need to ensure that every response and question is only from the content provided. Do not use outside knowledge and don't let
        your responses be open-ended.
        7. If the user asks anything that is beyond the scope of the data provided below, let the user know in your response that the question
        is out of scope.

        '''
        # Define your system message here
        response = TutorBot.tutorBot(context, user_input, systemMessage)

    return response


result = model()
print(result)
