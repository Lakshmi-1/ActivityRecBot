from pyswip import Prolog
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
import ast
import re

load_dotenv()
MESSAGE_HISTORY_LIMIT = 6

prolog = Prolog()
prolog.consult("project.pl")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

state = {
    'location': 'unknown',
    'price_range': 'unknown',
    'tags': 'unknown',
    'time_of_day': 'unknown',
    'reservation': 'unknown',
    'age_group': 'unknown',
    'retry': False
}

shown_ids = set()
prev_suggestions = set()

messages = []

def safe_eval(arg):
    try:
        val = ast.literal_eval(arg)
        if isinstance(val, list):
            return val
        elif isinstance(val, tuple):
            return list(val)
        else:
            return [val]
    except Exception:
        return [arg.strip()]

def add_message(msg):
    messages.append(msg)
    return messages[-MESSAGE_HISTORY_LIMIT:]

def parse_predicates(response_text):
    global state
    try:
        predicate_str = response_text.strip()
        match = re.findall(r'(\w+\([^\)]*\))', predicate_str)

        for p in match:
            if p.startswith("location("):
                state['location'] = p[9:-1]
            elif p.startswith("price_range("):
                state['price_range'] = p[12:-1]
            elif p.startswith("tags("):
                state['tags'] = p[5:-1]
            elif p.startswith("time_of_day("):
                state['time_of_day'] = p[12:-1]
            elif p.startswith("reservation("):
                state['reservation'] = p[12:-1]
            elif p.startswith("age_group("):
                state['age_group'] = p[10:-1]
    except Exception as e:
        print("Bot: Sorry, I couldn't understand that properly.")
        return None

def call_prolog_relax(messages):
    excluded_ids_str = "[" + ",".join([str(i) for i in shown_ids]) + "]"
    prev_suggestions_str = "[" + ",".join([str(i) for i in prev_suggestions]) + "]"
    query = f"check_relax({state['location']}, {state['price_range']}, {state['tags']}, {state['time_of_day']}, {state['reservation']}, {state['age_group']}, {excluded_ids_str}, {prev_suggestions_str}, Results)"
    try:
        results = list(prolog.query(query))
        prev_suggestions.add(results[0]['Results'])
        response = print_relax_results(results[0]['Results'])
        print(f"Bot: {response.content}")
        messages = []
    except Exception as e:
        match = re.search(r"relax\(error\)", str(e))
        if match:
            response = "I'm sorry, looks like I showed you all the options available."
            print(f"Bot: {response}")
            prev_suggestions.clear()
            messages = []
        else:
            print(f"Bot: Error occurred: {e}")

    return messages

def call_prolog_query(messages, q):
    query = f"query({q[0]}, {q[1]}, Results)"
    try:
        results = list(prolog.query(query))
        response = print_query_results("Query = " + str(query) + " + Results = " + str(results[0]['Results']))
        print(f"Bot: {response.content}")
        messagess = add_message(AIMessage(content=response.content))
    except Exception as e:
        match = re.search(r"query\(error\)", str(e))
        if match:
            response = "I'm sorry, I need you to provide me a specific activity ID for me to lookup that information."
            print(f"Bot: {response}")
            messages = []
        else:
            print(f"Bot: Error occurred: {e}")

    return messages

def call_prolog_recommend(messages):
    global state
    if state['retry'] == True:
        excluded_ids_str = "[" + ",".join([str(i) for i in shown_ids]) + "]"
        query = f"get_recommendations({state['location']}, {state['price_range']}, {state['tags']}, {state['time_of_day']}, {state['reservation']}, {state['age_group']}, {excluded_ids_str}, Results)"
        state['retry'] = False
    else:
        query = f"get_recommendations({state['location']}, {state['price_range']}, {state['tags']}, {state['time_of_day']}, {state['reservation']}, {state['age_group']}, Results)"
    try:
        results = list(prolog.query(query))
        if results[0]['Results'][0] != []:
            messages = []
            print(f"Bot: Here are some matches to your search criteria. To learn more about any one of them, use the respective ID. You can also ask for more options.")
            for m in results[0]['Results'][0]:
                shown_ids.add(m)
            print("IDs | Names")
            for item in results[0]['Results'][1:]:
                id = item[0]
                name = item[1]
                print(f" {id}   {name}")
        else:
            print("Bot: It seems that you search criteria did not produce any results. Would you like me to help adjust the search to find more options?")
            messages = []
            state['retry'] = False
            shown_ids.clear()
    except Exception as e:
        match1 = re.search(r"missing_input\((\w+)\)", str(e))
        match2 = re.search(r"unavailable_[a-z]*", str(e))
        if match1:
            response = prompt_missing_info(match1.group(0))
            print(f"Bot: {response.content}")
            messages = add_message(AIMessage(content=response.content))
        elif match2:
            response = prompt_unavailble(match2.group(0))
            print(f"Bot: {response.content}")
            messages = add_message(AIMessage(content=response.content))
        else:
            print(f"Bot: Error occurred: {e}")
    return messages

def control_logic(messages):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """===========================================================
                    CONTROL RESPONSES (ONLY WHEN STRICTLY NECESSARY)
                    -----------------------------------------------------------
                    - 'irrelevant': If the user's message is completely irrelevant to the items you are tasked with extracting.
                    - 'thank': If the user expresses gratitude (e.g., “Thanks!”).
                    - 'greeting': If the user greets (e.g., “Hello”).
                    - 'quit': If the user wants to end the conversation.
                    - 'retry': if the user wants to get new options or retry the query.
                    - 'relax': if the user wants help relaxing search criteria.
                    - 'predicate' : If the message is asking for info (e.g., activity details, locations, tags) or if the message involves preferences (e.g., city, time, price, interests, reservation, age group).

                    Examples:
                    - "Spaceships are cool." → 'irrelevant'  
                    - "Thank you so much!" → 'thank'  
                    - "Hey there!" → 'greeting'  
                    - "I want to quit now." → 'quit'
                    - "Can you show me something else?" → 'retry'
                    - "Do you have other activities?" → 'retry'
                    - "I would like help adjusting the criteria" → 'relax'
                    - "Help me relax the criteria" → 'relax'
                    - AI Message: It seems that you search criteria did not produce any results. Would you like me to help adjust the search to find more options? User: Yes please → 'relax'
                    - "Is the place family-friendly?" → 'predicate'
                    - 'I like food and bars" → 'predicate'
                    - 'I'd like to try houston instead" → 'predicate'
                    - "I'd like to visit cheap places in dfw" → 'predicate'
                    - "Okay can you try houston" → 'predicate'

                    Only give a one-word output with irrelevent, thank, greetin, quit, retry, relax, or predicate.
                    Use the 'predicate' control when the user want to subtitute any of the options or try a different option, NOT the 'retry' control."""
            ),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm

    return chain.invoke(messages)

def extract_predicates(messages):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Your job is to extract structured predicates based on the user’s preferences. These predicates represent key parameters about their desired activity. Only extract predicates if the user clearly expresses a preference for them. Do not infer or assume information.

===========================================================
PREDICATE FORMAT
-----------------------------------------------------------
Each predicate must be formatted as:

- location([...])  
- tags([...])  
- price_range(cheap/moderate/expensive/luxury/any)  
- time_of_day(morning/afternoon/evening/any)  
- reservation(yes/no/any)  
- age_group(adult/teen/all)
- query([activityIds], ['city'/'tags'/'price_range'/'time_of_day'/'reservation'/'age_group'/'description'/'url'/'address'])

Multi-word locations should be combined (e.g., "fort worth" → "fortworth").

===========================================================
WHEN TO EXTRACT PREDICATES
-----------------------------------------------------------
- Only extract a predicate if the user explicitly specifies it.
- If they explicitly say they have no preference, use `any` for that predicate.
- Never include a predicate unless it was mentioned or updated in the most recent user message.
- Do not assume defaults.

Examples:
- Bot: What is your budget? User: I don’t care. → [price_range(any)]
- Bot: What time would you like? User: I’m free any time. → [time_of_day(any)]
- Bot: What are some of your interests? User: I don't have any in particular → [tags(any)]
- User: I am okay with any location. → [location(any)]
- User: I'd like to visit cheap places. → [price_range(cheap)]
- User: I'm down to do anything. → [tags(any)]

Do not include a list with tags unless the user explicitly provides multiple tags.
**Do not extract `tags(any)` unless the user explicitly says they are okay with any type of interest.**

===========================================================
UPDATING PREDICATES
-----------------------------------------------------------
If the user changes their mind, only return the updated predicate.

Example flow:
User: I'd like to visit cheap places to swim in Austin.  
→ [location(austin), tags(swimming), price_range(cheap)]

User: Okay, nature and fun instead.  
→ [tags([nature, fun])]

User: Make that any price range.  
→ [price_range(any)]

User: Can you change the price to cheap?
→ [price_range(cheap)]

User: Actually, I’d like to go in DFW instead.  
→ [location(dfw)]

User: Actually, I'd like food and museums.
-> [tags([food, museum])]

===========================================================
EXAMPLES – CORRECT EXTRACTIONS
-----------------------------------------------------------
User: I want to go shopping in Austin with a budget of $20 in the afternoon.  
→ [location(austin), tags(shopping), price_range(moderate), time_of_day(afternoon)]

User: Actually, I’d prefer evening.  
→ [time_of_day(evening)]

User: I want to go to places in houston or austin.
→ [location([houston,austin])]

User: Can we do Dallas instead of Austin?  
→ [location(dallas)]

User: Never mind the shopping, let’s do museums.  
→ [tags(museum)]

User: I don’t care about the time of day.  
→ [time_of_day(any)]

User: I want to do something fun. I'm free in the afternoon or evening.
-> [tags(fun), time_of_day([afternoon,evening])]

User: I want to do something fun.  
→ [tags(fun)]

User: I don’t care when or how much it costs, just something adventurous.  
→ [tags(adventure), time_of_day(any), price_range(any)]

User: I want cheap food in Austin.  
→ [location(austin), tags(food), price_range(cheap)]

User: Make that moderate.  
→ [price_range(moderate)]

User: I’m taking my kids too.  
→ [age_group(all)]

User: I’d like something family-friendly.
→ [age_group(all)]

===========================================================
AMBIGUOUS BUT MANAGEABLE CASES
-----------------------------------------------------------
User: I want something outdoorsy in DFW.  
→ [location(dfw), tags(nature)]

User: Something fun for teens under $50.  
→ [tags(fun), age_group(teen), price_range(moderate)]

===========================================================
DO NOT OVER-EXTRACT
-----------------------------------------------------------
User: I want to do fun stuff.  
→ [tags(fun)]  

User: Can I do something fun in the morning?
→ [tags(fun), time_of_day(morning)]

User: I'd like cheap activities with my kids.
→ [price_range(cheap), age_group(all)]

Do not add 'any' predicates unless the user explicitly states they don't care.

===========================================================
PRICE RANGE EXAMPLES
-----------------------------------------------------------
User: 50 dollars
→ [price_range(moderate)]

User 20 dollars
→ [price_range(cheap)]

User: I'd like to spend $100.
→ [price_range(expensive)]

User: I'm willing to spend 70 dollars.
→ [price_range(expensive)]

In general: cheap is $0-$20, moderate is $21-$50, expensive is $51-$99, luxury is $100+.

===========================================================
TAG EXTRACTION
-----------------------------------------------------------
User: I am thinking of going to a museum or an aquarium.
→ [tags([museum, aquarium])]

User I enjoy amusement parks, escape rooms, and bowling.
→ [tags([amusement_park, escape_room, bowling])]

User: I like to go on nature walks and do yoga.
→ [tags([nature_walk, yoga])]

User: I like spas and relaxing activities.
→ [tags([spa, relaxing])]

Available tags:
[nature, outdoors, park, hiking, kayaking, paddleboarding, swimming, adventure, scenic, relaxing, recreation]
[games, fun, arcade, bowling, escape_room, mini_golf, golf]
[culture, art, museum, gallery, theater, history, live_music]
[nightlife, bar, club, dancing, music_venue]
[wellness, yoga, meditation, spa, sauna, nature_walk]
[food, dining, restaurant, cafe, food_truck, brewery, wine_tasting]
[tours, food_tour, museum_tour, city_tour, walking_tour, boat_tour]
[shopping, boutiques, markets, vintage, mall]
[entertainment, amusement_park, zoo, aquarium, theme_park, movies]
[romantic, couples, date_night, intimate, cozy, sunset_view]

Keep the plurality of the user's requests consistent with the tags provided.
Tags should not be a list unless there are multiple interests provided by the user.

**Do not extract `tags(any)` unless the user explicitly says they have no interest preferences.**

===========================================================
THE QUERY PREDICATE
-----------------------------------------------------------
User: What locations do you have?
→ query([], ['city'])

User: What are some things I can choose from?
→ query([], ['city','tags'])

User: What activities do you have?
→ query([], ['tags'])

User: What city is activity 20 in?
→ query([20],['city'])

User: Can you give me a description and website for ids 17 and 23?
→ query([17,23],['description','url'])

User: Can you give me some more details on activity 5?
→ query([5],['address','description','url']) 

User: Is the place family-friendly?
→ query([],['age_group'])

User: Do I need a reservation?
→ query([],['reservation'])

User: Can I bring my kids to activity 10?
→ query([10],['age_group'])

User: Can you tell me more about activity 33?
→ query([33],['address','description','url'])

Do not extract other predicates with the query predicate.
Always extract the query predicate when the user is asking you for any type of information.

===========================================================
SUMMARY OF GUIDING PRINCIPLES
-----------------------------------------------------------
- Only extract predicates when the user explicitly expresses a preference.
- Always extract the query predicate when the user is asking you for any type of information.
- Never use the `any` option until the user explicitly says they don’t care about that particular predicate.
- If a user changes their mind, return only the updated predicates.
- Never assume a preference.
- The price range is not a tag.
- Tags should not be a list unless there are multiple interests provided by the user.
- Location should not be a list unless there are multiple locations provided by the user.
- For tags, update to a completely fresh list if the user changes their mind.
- Do not extract other predicates with the query predicate.
- Only include an arry of predicates as your final output. Never include any additional text or explanations.
- For multi-word locations, combine them into a single word (e.g., "fort worth" → "fortworth").
- **Never include `tags(any)` unless the user explicitly says they have no preference for.**"""
            ),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm

    return chain.invoke(messages)

def print_relax_results(results):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """'You are tasked with helping the user adjust search criteria to produce some results. 
                You will be given the critiera that the user needs to relax.
                Kindly instruct the user to adjust that critiera so they can produce results.
                For cities, the user only has austin, houston, or dfw. Suggest they choose a different city from the list, or relax the city filter altogether.
                'Kindly suggest the action the user needs to perform based on the inputs without additional text or explanations.'
                EX: city -> Okay! You can try choosing a different city from the list (austin, houston,dfw) or relaxing the city filter"""
            ),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm

    return chain.invoke(results)

def print_query_results(results):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """'You are tasked with formatting the outputs of a user's query in a human-readable format.'
                'You will be given the user's query and the results of the query so you can formulate the appropriate response.'
                'Do not mention anything about availability to the user'
                'Examples:
                Query = query([],['city'], Results = [austin,houston,dfw] ### Output: I have austin, houston, and dfw.
                Query = query([],['tags]), Results = [recreation,bar,club...] ### Output: I have plenty for you to choose from! <Use a random subset of result for options> What are you in the mood for?
                Query = query([11],['address','description','url'], Results = [[11, Alamo Drafhouse Cinema, 320 E 6th St, Austin, TX 78701, https://drafthouse.com/austin, Alamo Drafthouse Cinema is a unique movie theater that serves food and drinks during the film.]] ### Output: Of course! The Alamo Drafhouse Cinema is a unique experience where you can enjoy food and drinks during the film. Its address is 320 E 6th St, Austin, TX 78701. You can learn more at: https://drafthouse.com/austin.)
                'Produce a kind response to the user that provides all the information you are given in a human-readable format without additional text or explanations'"""
            ),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm

    return chain.invoke(results)

def prompt_missing_info(missing_info):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """'Your task is to ask the user for the missing information in a friendly and engaging manner.'
                'You will be given a type of missing information that the user needs to provide in order to get activity suggestions.'
                'Example: missing_input(city) -> Ask the user to provide a location they are interested in. The user will ideally choose a city from Texas.'
                'Example: missing_input(tags) -> Ask the user to provide their interests or if they had a specific type of activity in mind.'
                'Example: missing_input(price) -> ask the user to provide their budget.'
                'Example: missing_input(time) -> Would you like sometime in the morning, afternoon, or evening.'
                'Just simply ask them the appropriate question based on the missing information without additional text or explanations.'"""
            ),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm

    return chain.invoke(missing_info)

def prompt_unavailble(unavailable):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """'You are a helpful assistant. The user provided me some information that is not available for querying in the database. Your task is to redirect the user to provide some queryable information. Do not include any additional text or explanations.'
                'Your input is the type of unavailable information.'
                'You and only you are the assistant there are no additional users.'
                'For example, if the user provided a location that is not available in the database, ask them to provide a different location.'
                'Example: unavailable_tags, Output: "I am sorry, but I do not have any activities available for the interests you mentioned. I have ___ available. Would you like to try one of those options?"',
                'Example: unavailable_city, Output: "Sorry, I do not have any activities available in that city. I have Austin, Houston, and DFW available. Would you like to try one of those locations?"',
                'The user can choose from austin, dfw, and houston for locations',
                'Simply apologize for not having it available and query the user to use a new input in the form of a question.'
                'Use the a random selection of the following tags to help you fill in the blank with the unavailable_tags input:
                [nature, outdoors, park, hiking, kayaking, paddleboarding, swimming, adventure, scenic, relaxing, recreation]
                [games, fun, arcade, bowling, escape_room, mini_golf, golf]
                [culture, art, museum, gallery, theater, history, live_music]
                [nightlife, bar, club, dancing, music_venue]
                [wellness, yoga, meditation, spa, sauna, nature_walk]
                [food, dining, restaurant, cafe, food_truck, brewery, wine_tasting]
                [tours, food_tour, museum_tour, city_tour, walking_tour, boat_tour]
                [shopping, boutiques, markets, vintage, mall]
                [entertainment, amusement_park, zoo, aquarium, theme_park, movies]
                [romantic, couples, date_night, intimate, cozy, sunset_view]'
                'Just simply ask them the appropriate question based on the unavailable information without additional text or explanations.'"""
            ),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm

    return chain.invoke(unavailable)

print("Welcome to the Texas activity suggestion system! You can provide any of the following: city, budget, interests, time of day, reservation requirements, and age groups.\n")

user_input = input("You: ")
messages = add_message(HumanMessage(user_input))

while True:
    response = control_logic(messages)
    if re.search(r'\bquit\b', response.content, re.IGNORECASE):
        print("Bot: Goodbye! Thanks for chatting!")
        break
    elif re.search(r'\bthank(s| you)?\b', response.content, re.IGNORECASE):
        print("Bot: You're welcome!")
        messages = []
    elif re.search(r'\b(not relevant|irrelevant)\b', response.content, re.IGNORECASE):
        print("Bot: That's not relevant to activity suggestions.")
        messages = []
    elif re.search(r'\b(hi|hello|hey|greeting)\b', response.content, re.IGNORECASE):
        print("Bot: Hello! How can I assist you today?")
        messages = []
    elif re.search(r'\b(retry)\b', response.content, re.IGNORECASE):
        state['retry'] = True
        messages = call_prolog_recommend(messages)
    elif re.search(r'\b(relax)\b', response.content, re.IGNORECASE):
        messages = call_prolog_relax(messages)
    else:
        response2 = extract_predicates(messages)
        if re.search(r'query\(\[(.*?)\],\s*\[(.*?)\]\)', response2.content, re.IGNORECASE):
            match = re.search(r'query\(\[(.*?)\],\s*\[(.*?)\]\)', response2.content, re.IGNORECASE)
            list1_str, list2_str = match.groups()
            q = [safe_eval(list1_str), safe_eval(list2_str)]
            messages = call_prolog_query(messages,q)
        else:
            parse_predicates(response2.content)
            messages = call_prolog_recommend(messages)

    user_input = input("You: ")
    messages = add_message(HumanMessage(user_input))