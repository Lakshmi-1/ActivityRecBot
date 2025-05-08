# ActivityRecBot
## Overview
ActivityRecBot is designed to assist users in discovering suitable activities in various Texas cities. This system operates under a closed-world assumption and uses Prolog-based querying over a structured knowledge base. This approach ensures that all recommendations are grounded in actual database entries, effectively preventing hallucinations or fabricated responses often seen in purely language-model-driven systems. As a result, users can trust the chatbot’s suggestions to be accurate, relevant, and based solely on the available data. In its current implentation, the chatbot leverages a curated knowledge base of 50 options, each characterized by properties such as ID, name, city, address, tags, price range, time of day, reservation requirements, description, URL, and age group.

## How To Run
1) Install [SWI-Prolog](https://www.swi-prolog.org/download/stable) 
2) Get an Free API key from [Groq](https://console.groq.com/keys)
3) Clone the repo
4) Configure a .env file with your GROQ_API_KEY
5) Run 'pip install requirements.txt'
6) Run 'python LLM.py'
7) Happy Chatting :)

## Features
Information Querying - Users can ask the chatbot for general information about the activities stored in its knowledge base or inquire about specific activities. This allows the chatbot to serve as an interactive guide, providing details such as descriptions, price ranges, time of day, reservation requirements, and age suitability for each activity.

Personalized Recommendations - Users can describe their preferences in natural language to receive tailored suggestions that closely align with their needs.

Relaxation Suggestions - If the chatbot cannot find activities that meet all the specified criteria, it can suggest which criteria the user might consider relaxing to expand the pool of available options. This makes the system both flexible and user-friendly, helping users avoid dead ends and increasing the likelihood of finding a good match.


