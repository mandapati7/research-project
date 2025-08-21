# app.py
# Streamlit app for Deep Research
# This app follows the flow of deep_research.py and is modularized for clarity.
# Comments are provided for learning purposes.

import streamlit as st
import os
import json
import itertools
from openai import OpenAI



# Helper function to set OpenAI API key from secrets or .env
def set_api_key_env():
    api_key = None
    # Try to get from Streamlit secrets, but handle missing secrets.toml gracefully
    try:
        if hasattr(st, 'secrets') and 'openai_api_key' in st.secrets:
            api_key = st.secrets['openai_api_key']
    except Exception:
        pass
    # Try to get from environment
    if not api_key:
        api_key = os.environ.get('OPENAI_API_KEY')
    # Try to load from .env if not found
    if not api_key and os.path.exists('.env'):
        with open('.env') as f:
            for line in f:
                if line.startswith('OPENAI_API_KEY='):
                    api_key = line.strip().split('=', 1)[1]
                    break
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
    return api_key

# Function to initialize OpenAI client
@st.cache_resource(show_spinner=False)
def get_openai_client():
    return OpenAI()

# Function to ask clarifying questions using the LLM
@st.cache_data(show_spinner=False)
def get_clarifying_questions(_client, topic, developer_message, model_mini):
    prompt_to_clarify = f"""
Ask 5 numbered clarifying question to the user about the topic: {topic}.
The goal of the questions is to understand the intended purpose of the research.
Reply only with the questions.
"""
    clarify = _client.responses.create(
        model=model_mini,
        input=prompt_to_clarify,
        instructions=developer_message
    )
    questions = clarify.output[0].content[0].text.split("\n")
    return questions, clarify.id

# Function to get research goal and queries
@st.cache_data(show_spinner=False)
def get_goal_and_queries(_client, answers, questions, topic, developer_message, model, clarify_id):
    prompt_goals = f"""
Using the user answers {answers} to questions {questions}, write a goal sentence and 5 web search queries for the research about {topic}
Output: A json list of 5 web search queries and a goal sentence that wil reach it
Format: {{\"goal\":\"...\", \"queries\":[\"q1\",.....]}}
"""
    goal_and_queries = _client.responses.create(
        model=model,
        input=prompt_goals,
        previous_response_id=clarify_id,
        instructions=developer_message
    )
    plan = json.loads(goal_and_queries.output[0].content[0].text)
    return plan, goal_and_queries.id

# Function to run a web search query
@st.cache_data(show_spinner=False)
def run_search(_client, q, developer_message, model, tools):
    web_search = _client.responses.create(
        model=model,
        input=f"seach: {q}",
        instructions=developer_message,
        tools=tools
    )
    # Defensive: check output length and handle errors
    if len(web_search.output) > 1 and hasattr(web_search.output[1], 'id') and hasattr(web_search.output[1], 'content'):
        return {
            "query": q,
            "resp_id": web_search.output[1].id,
            "research_output": web_search.output[1].content[0].text
        }
    else:
        # Return error info for debugging
        return {
            "query": q,
            "resp_id": None,
            "research_output": f"Error: Unexpected response format: {web_search.output}"
        }

# Function to evaluate if the research goal is met
@st.cache_data(show_spinner=False)
def evaluate(_client, collected, goal, developer_message, model):
    review = _client.responses.create(
        model=model,
        input=[
            {"role": "developer", "content": f"Reasearch goal: {goal}"},
            {"role": "assistant", "content": json.dumps(collected)},
            {"role": "user", "content": "Does this information will fully satisfy the goal? Answer Yes or No only!"}
        ],
        instructions=developer_message
    )
    return "yes" in review.output[0].content[0].text.lower()

# Function to get more queries if needed
@st.cache_data(show_spinner=False)
def get_more_queries(_client, collected, goal, developer_message, model, goal_and_queries_id):
    more_searches = _client.responses.create(
        model=model,
        input=[
            {"role": "assistant", "content": f"Current data: {json.dumps(collected)}"},
            {"role": "developer", "content": f"Reasearch goal: {goal}. write 5 other web searchs to achieve the goal"},
        ],
        instructions=developer_message,
        previous_response_id=goal_and_queries_id
    )
    return json.loads(more_searches.output[0].content[0].text)

# Function to write the final report
@st.cache_data(show_spinner=False)
def write_report(_client, collected, goal, developer_message, model):
    report = _client.responses.create(
        model=model,
        input=[
            {"role": "developer", "content": (f"Write a complete and detailed report about reasearch goal: {goal} "
            "Cite sources inline using  [n] and append a reference "
            "list mapping [n] to url")},
            {"role": "assistant", "content": json.dumps(collected)},
        ],
        instructions=developer_message
    )
    return report.output[0].content[0].text

# Streamlit UI
st.title('Deep Research Streamlit App')
st.write('This app helps you perform deep research using OpenAI and web search tools.')



# Read API key from secrets, env, or .env
api_key = set_api_key_env()
if not api_key:
    st.error('OpenAI API key not found. Please set it in Streamlit secrets, as an environment variable, or in a .env file.')
    st.stop()

client = get_openai_client()

MODEL = "gpt-4.1"
MODEL_MINI = "gpt-4.1-mini"
TOOLS = [{"type": "web_search"}]
developer_message = """
You are an expert Deep Researcher.
You provide complete and in depth research to the user.
"""

# Step 1: Get research topic from user
with st.form('topic_form'):
    topic = st.text_input('Enter the research topic:')
    submitted = st.form_submit_button('Submit Topic')

if 'topic' not in st.session_state:
    st.session_state['topic'] = ''
if submitted and topic:
    st.session_state['topic'] = topic

if st.session_state['topic']:
    st.success(f"Research Topic: {st.session_state['topic']}")
    # Step 2: Get clarifying questions
    with st.spinner('Generating clarifying questions...'):
        questions, clarify_id = get_clarifying_questions(client, st.session_state['topic'], developer_message, MODEL_MINI)
    st.write('Please answer the following clarifying questions:')
    answers = []
    for i, q in enumerate(questions):
        ans = st.text_input(f"Q{i+1}: {q}", key=f"answer_{i}")
        answers.append(ans)
    if all(answers):
        # Step 3: Get goal and queries
        with st.spinner('Generating research goal and queries...'):
            plan, goal_and_queries_id = get_goal_and_queries(client, answers, questions, st.session_state['topic'], developer_message, MODEL, clarify_id)
        goal = plan["goal"]
        queries = plan["queries"]
        st.info(f"Research Goal: {goal}")
        st.write('Web Search Queries:')
        for q in queries:
            st.write(f"- {q}")
        # Step 4: Run web searches and collect data
        if st.button('Run Research'):
            collected = []
            progress = st.progress(0, text='Running web searches...')
            for idx, q in enumerate(queries):
                st.write(f"Collecting data for query: {q}")
                with st.spinner(f'Running web search for: {q}'):
                    collected.append(run_search(client, q, developer_message, MODEL, TOOLS))
                progress.progress((idx + 1) / len(queries), text=f'Completed {idx + 1} of {len(queries)} searches')
            progress.empty()
            # Step 5: Evaluate if enough information is collected
            with st.spinner('Evaluating if enough information is collected...'):
                enough = evaluate(client, collected, goal, developer_message, MODEL)
            if not enough:
                st.warning('Not enough information. Generating more queries...')
                with st.spinner('Generating more queries...'):
                    queries = get_more_queries(client, collected, goal, developer_message, MODEL, goal_and_queries_id)
                for idx, q in enumerate(queries):
                    st.write(f"Collecting data for query: {q}")
                    with st.spinner(f'Running web search for: {q}'):
                        collected.append(run_search(client, q, developer_message, MODEL, TOOLS))
            # Step 6: Write and display the final report
            with st.spinner('Writing final research report...'):
                report = write_report(client, collected, goal, developer_message, MODEL)
            # Show Q&A summary as plain text for printing/analysis
            st.markdown('### Clarifying Questions and Answers')
            for i, (q, a) in enumerate(zip(questions, answers)):
                st.markdown(f"**Q{i+1}: {q}**  <br>**A{i+1}:** {a}", unsafe_allow_html=True)
            st.markdown('### Final Research Report')
            st.markdown(report, unsafe_allow_html=True)
            # Add print-specific CSS to expand all textareas and inputs for printing
            st.markdown("""
                <style>
                @media print {
                  textarea, input[type="text"] {
                    height: auto !important;
                    min-height: 40px !important;
                    max-height: none !important;
                    overflow: visible !important;
                    white-space: pre-wrap !important;
                  }
                  .stTextInput>div>div>input {
                    width: 100% !important;
                    min-width: 300px !important;
                  }
                }
                </style>
            """, unsafe_allow_html=True)
            # Show print instructions instead of a print button
            st.info('To print or save the report, use your browser\'s print feature: press Ctrl+P (Windows) or Cmd+P (Mac).')
