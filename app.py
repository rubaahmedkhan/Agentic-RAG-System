import streamlit as st
import asyncio
from agents import Runner
from agent import agent, config  # Import your agent and config
from openai.types.responses import ResponseTextDeltaEvent

st.set_page_config(page_title="AI Assistant", page_icon="🤖")
st.title("Agentic RAG Assistant")
st.markdown("Ask anything about OpenAI Agents SDK or the Agentic AI video.")

query = st.text_input("💬 Enter your question", placeholder="e.g. What is agentic AI?")
submit = st.button("🔍 Ask")

if submit and query:
    st.info("💡 Streaming response...")
    output_placeholder = st.empty()

    async def stream_answer():
        result = Runner.run_streamed(agent, input=query, run_config=config)
        final_answer = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                final_answer += event.data.delta
                output_placeholder.markdown(final_answer)

    asyncio.run(stream_answer())
