import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from streamlit_chat import message
import logging
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("API_KEY")

logging.info('TutorBot application started.')


# Initialize session states

# Set up logging
logging.basicConfig(filename='logs/tutor_bot.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info('Setting up initial context.')


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + \
            st.session_state['responses'][i+1] + "\n"
    return conversation_string


logging.info('Initial context set up.')

logging.debug('Checking if responses and requests are in session state.')


def tutorBot(context, input, systemMessage):

    st.title("TutorBot by CustomGPT")

    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["How can I assist you?"]

    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    logging.info('Setting up ChatOpenAI and ConversationBufferWindowMemory.')

    llm = ChatOpenAI(model_name="gpt-4-0613",
                     openai_api_key=API_KEY,
                     temperature=0,
                     max_tokens=2048
                     )

    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(
            k=10, return_messages=True)

    system_msg_template = SystemMessagePromptTemplate.from_template(
        template=systemMessage + context)
    human_msg_template = HumanMessagePromptTemplate.from_template(
        template="{input}")

    prompt_template = ChatPromptTemplate.from_messages(
        [system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

    conversation = ConversationChain(
        memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

    logging.info('ChatOpenAI and ConversationBufferWindowMemory set up.')

    # container for chat history
    response_container = st.container()
    # container for text box
    textcontainer = st.container()

    with textcontainer:
        query = st.text_input("Query: ", key="input2")
        if query:
            logging.debug('Received user query: %s', query)
            with st.spinner("typing..."):
                conversation_string = get_conversation_string()
                response = conversation.predict(
                    input=f"Context:\n {conversation_string} \n\n Query:\n{query}")
                logging.info(
                    'Processed and generated response for user query.')

            st.session_state.requests.append(query)
            st.session_state.responses.append(response)
            logging.debug('Added user query and response to session state.')

    with response_container:
        if st.session_state['responses']:
            logging.debug('Displaying responses and requests for the session.')
            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i], key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i],
                            is_user=True, key=str(i) + '_user')
            logging.info('Session ended.')
