from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from abc import ABC, abstractmethod

load_dotenv()

# Initialize the LLM
llm = init_chat_model(model_provider="openai", model="gpt-4.1")


# Define the state of the graph
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None
    next: str | None


# Define the message classifier
class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        default=...,
        description="Classify if the message requires an emotional (therapist) or logical response."
    )


# Define the message classifier
def classify_message(state: State):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke(
        [
            {
                "role": "system",
                "content": """
                Classify the user message as either:
                - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
                - 'logical': if it asks for facts, information, logical analysis, or practical solutions
                """,
            },
            {
                "role": "user",
                "content": last_message.content
            },
        ]
    )
    return {"message_type": result.message_type}


def router(state: State):
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "therapist"}
    if message_type == "logical":
        return {"next": "scientist"}


# Define the base agent class
class BaseAgent(ABC):
    """Base class for all agents in the system."""

    def __init__(self, llm, name):
        self.llm = llm
        self.name = name

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass

    def __call__(self, state: State) -> dict:
        """Process the state and return the agent's response."""
        last_message = state["messages"][-1]

        messages = [
            {
                "role": "system",
                "content": self.get_system_prompt(),
            },
            {
                "role": "user",
                "content": last_message.content,
            },
        ]

        reply = self.llm.invoke(messages)
        return {"messages": [{"role": "assistant", "content": f"({self.name}) {reply.content}"}]}


class TherapistAgent(BaseAgent):
    """Therapist agent focused on emotional support and empathy."""

    def get_system_prompt(self) -> str:
        return """
        You are a compassionate therapist. Focus on the emotional aspects of the user's message.
        Show empathy, validate their feelings, and help them process their emotions.
        Ask thoughtful questions to help them explore their feelings more deeply.
        Avoid giving scientific solutions unless explicitly asked.
        """


class ScientistAgent(BaseAgent):
    """Scientist agent focused on facts and logical analysis."""

    def get_system_prompt(self) -> str:
        return """
        You are a scientist. Focus only on facts and information.
        Provide clear, concise answers based on logic and evidence.
        Do not address emotions or provide emotional support.
        Be direct and straightforward in your responses.
        """


# Create agent instances
therapist_agent = TherapistAgent(llm=llm, name="Therapist")
scientist_agent = ScientistAgent(llm=llm, name="Scientist")

# Building the graph: nodes and edges
# Always start with all nodes then add all the edges

graph_builder = StateGraph(State)

graph_builder.add_node(node="classifier", action=classify_message)
graph_builder.add_node(node="router", action=router)
graph_builder.add_node(node="therapist-agent", action=therapist_agent)
graph_builder.add_node(node="scientist-agent", action=scientist_agent)

graph_builder.add_edge(start_key=START, end_key="classifier")
graph_builder.add_edge(start_key="classifier", end_key="router")

graph_builder.add_conditional_edges(
    source="router",
    path=lambda state: state.get("next"),
    path_map={"therapist": "therapist-agent", "scientist": "scientist-agent"},
)

graph_builder.add_edge(start_key="therapist-agent", end_key=END)
graph_builder.add_edge(start_key="scientist-agent", end_key=END)

graph = graph_builder.compile()

# print(graph.get_graph().draw_mermaid())


def run():
    state = {"messages": [], "message_type": None}

    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("Bye")
            break

        state["messages"] = state.get("messages", []) + [{"role": "user", "content": user_input}]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")


if __name__ == "__main__":
    run()
