import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key not found! Please set OPENAI_API_KEY in .env.")

class Agent:
    def __init__(self, name="Agent"):
        self.name = name
        self.status = "idle"

    def generate_response(self, user_input):
        try:
            cleaned_input = user_input.strip()
            if not cleaned_input:
                return "No input provided!"

            print("\nProcessing...")
            self.status = "planning"
            print("Status: Planning response...")
            plan_text = self._create_plan(cleaned_input)

            if not plan_text:
                raise ValueError("Plan creation returned no result.")

            self.status = "executing"
            print("Status: Executing plan...")
            partial_results = self._execute_plan(plan_text, cleaned_input)

            self.status = "completed"
            final_answer = self._synthesize_results(partial_results)
            return final_answer

        except Exception as e:
            self.status = "error"
            return "Agent encountered an error: " + str(e)

    def _create_plan(self, user_input):
        prompt = (
            "I am an AI agent. The user input is:\n"
            f"'{user_input}'\n\n"
            "Please list 1-3 steps (each on a new line) that I should do to handle this input.\n"
            "Example:\n"
            "1. Understand the user's question\n"
            "2. Research the topic\n"
            "3. Provide a clear answer\n"
        )
        plan_response = self._call_llm(prompt)
        return plan_response

    def _parse_plan(self, plan_text):
        lines = plan_text.split("\n")
        steps = []
        for line in lines:
            step = line.strip()
            if step:
                steps.append(step)
        return steps

    def _execute_plan(self, plan_text, user_input):
        steps = self._parse_plan(plan_text)
        results = []

        for step in steps:
            try:
                step_prompt = (
                    f"Step: {step}\n\n"
                    f"User Input: '{user_input}'\n\n"
                    "Please do this step and provide the result."
                )
                result = self._call_llm(step_prompt)
                results.append(f"Step: {step}\nResult: {result}")
            except Exception as e:
                results.append(f"Step: {step}\nResult: Error executing this step: {e}")
        return results

    def _synthesize_results(self, partial_results):
        synthesis_prompt = (
            "Here are the partial results from each step:\n\n"
            + "\n\n".join(partial_results)
            + "\n\nProvide a single, clear paragraph that synthesizes these results into a direct and complete response." 
        )
        final_answer = self._call_llm(synthesis_prompt)
        return final_answer

    def _call_llm(self, prompt):
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful AI agent."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()

def main():
    agent = Agent(name="Agent")
    os.system('cls' if os.name == 'nt' else 'clear')
    print("""Welcome to the AI Agent Interface!
--------------------------------
This agent can help answer your questions and assist with various tasks.
Simply type your question or request, and the agent will respond.
Examples:
- 'What is Python?'
- 'How do I start learning programming?'
- 'Explain what an API is'
Type 'quit' to exit the program.
--------------------------------""")

    while True:
        user_input = input("User: ")
        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        response = agent.generate_response(user_input)
        print(f"\n{agent.name}: {response}\n(Agent status: {agent.status})")

if __name__ == "__main__":
    main()
