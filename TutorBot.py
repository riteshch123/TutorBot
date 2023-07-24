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


def tutorBot(context, input):

    context = '''
    Most life on Earth depends on photosynthesis.The process is carried out by plants, algae, and some types of bacteria, which capture energy
    from sunlight to produce oxygen (O2) and chemical energy stored in glucose (a sugar). Herbivores then obtain this energy by eating plants,
    and carnivores obtain it by eating herbivores.

    The process

    During photosynthesis, plants take in carbon dioxide (CO2) and water (H2O) from the air and soil. Within the plant cell, the water is oxidized,
    meaning it loses electrons, while the carbon dioxide is reduced, meaning it gains electrons. This transforms the water into oxygen and the
    carbon dioxide into glucose. The plant then releases the oxygen back into the air, and stores energy within the glucose molecules.

    Chlorophyll

    Inside the plant cell are small organelles called chloroplasts, which store the energy of sunlight. Within the thylakoid membranes of the
    chloroplast is a light-absorbing pigment called chlorophyll, which is responsible for giving the plant its green color. During photosynthesis,
    chlorophyll absorbs energy from blue- and red-light waves, and reflects green-light waves, making the plant appear green.

    Light-dependent reactions vs. light-independent reactions

    While there are many steps behind the process of photosynthesis, it can be broken down into two major stages: light-dependent reactions
    and light-independent reactions. The light-dependent reaction takes place within the thylakoid membrane and requires a steady stream of
    sunlight, hence the name light-dependent reaction. The chlorophyll absorbs energy from the light waves, which is converted into chemical
    energy in the form of the molecules ATP and NADPH. The light-independent stage, also known as the Calvin Cycle, takes place in the stroma,
    the space between the thylakoid membranes and the chloroplast membranes, and does not require light, hence the name light-independent reaction.
    During this stage, energy from the ATP and NADPH molecules is used to assemble carbohydrate molecules, like glucose, from carbon dioxide.

    C3 and C4 photosynthesis

    Not all forms of photosynthesis are created equal, however. There are different types of photosynthesis, including C3 photosynthesis
    and C4 photosynthesis. C3 photosynthesis is used by the majority of plants. It involves producing a three-carbon compound called
    3-phosphoglyceric acid during the Calvin Cycle, which goes on to become glucose. C4 photosynthesis, on the other hand, produces a
    four-carbon intermediate compound, which splits into carbon dioxide and a three-carbon compound during the Calvin Cycle. A benefit
    of C4 photosynthesis is that by producing higher levels of carbon, it allows plants to thrive in environments without much light or water.
    '''

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

    st.title("TutorBot using OpenAI")

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
