import os
import getpass

from langchain_community.tools import DuckDuckGoSearchRun # Get search tool from langchain
from crewai import Agent, Task, Crew, Process # Crew AI imports


os.environ["OPENAI_API_KEY"] = getpass.getpass("OPENAI API KEY: ")

# Get search tool from langchain
search_tool = DuckDuckGoSearchRun()

# Create agents

# Define agent for research
researcher = Agent(
    role='Educational Researcher',
    goal='Develop ideas for teaching some one a new topic',
    backstory=""" You are an expert researcher. Who is focused on reasearching topics for educational purposes""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool]
)

# Define agent for writing
writer = Agent(
    role='Educational Content Writer',
    goal='Craft compelling content that simplifies a topic ',
    backstory="""You are a renowned Content Writer, known for transforming complex concepts into compelling narratives.""",
    verbose=True,
    allow_delegation=True
)

# Define examiner agent
examiner = Agent(
    role='Educational examiner',
    goal='Create questions that will test the knowledge of the student.',
    backstory="""You are a renowned Examiner you are really good at generating questions that can properly test the knowledge of a student.""",
    verbose=True,
    allow_delegation=True
)


def run_task(topic):
    # Create tasks for researcher

    task1 = Task(
        description=f"""Develop comprehensive research on this topic: {topic}. If you encounter any thing you are not sure of use your search engine.""",
        expected_output="A comprehensive research on the topic.",
        agent=researcher
    )

    # Create tasks for writer

    task2 = Task(
        description="""Using the insights provided, write an article that simplifies the topic even further making it beginner friendly.""",
        expected_output="A simplified but detailed article on the topic.",
        agent=writer
    )

    # Create tasks for examiner

    task3 = Task(
        description="""Using the article provided generate questions that test the reader's knowledge.""",
        expected_output="2 to 3 questions from the article.",
        agent=examiner
    )
    
    # Instantiate your crew with a sequential process

    crew = Crew(
        agents=[researcher, writer, examiner],
        tasks=[task1, task2, task3],
        Process=Process.sequential, # You can use different processes: https://docs.crewai.com/core-concepts/Managing-Processes/
        verbose=2, # You can set it to 1 or 2 to different logging levels
    )

    # Get your crew to work!
    crew.kickoff()
    
    return task2.output.result, task3.output.result

def main():
    topic = input("Provide the topic you wish to learn about: ")

    article, question = run_task(topic)

    print(article)
    
    input("Press any key to see question: ")

    print(question)

if __name__ == '__main__':
    main()
