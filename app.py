from flask import Flask, request, redirect, render_template, url_for, jsonify
import re
import pandas as pd
import os
import csv
import openai
from langchain.agents import agent_types
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models.azure_openai import AzureChatOpenAI
#from langchain_openai import ChatOpenAI

from openai import OpenAI
from flask_cors import CORS
import json
from tabulate import tabulate
import uuid
import time
from langchain_core.tools import StructuredTool
from CpoThread import CpoThread
#from zod_csv import parseCSVContent, zcsv
#from zod import z
#import seaborn as sns

client = OpenAI(api_key='')
#75c95b56f9344bb380e34ef51b8771aa
SUMMARY_SYSTEM_PROMPT='Imagine you are a Central Procurement Officer in a consumer bank. List the key insights from the below content. Your facts should be short and consice.'

app = Flask(__name__)
CORS(app)

# Shared value
shared_value = [0.2]

# Create a new thread
#thread = CpoThread(shared_value)

# Start the thread
#thread.start()

def normalize_text(s):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n","")
    s = s.replace("Key Insights:","")
    s = s.strip()
    
    return s.lower()

def sanitize_keys(key):
    sanitize_key = re.sub(r'[a-zA-Z0-9_]','_',key)
    return sanitize_key
def get_column_data(data):
    column_data = {}
    columns = ["Title","Contractenddate", "Contractstartdate", "Costsavings"]
    for item in columns:
        value_list = [item1[item] for item1 in data[:4]]             
        column_data = {item: value_list}
        columns_json.append(column_data)
    return columns_json

def replace_space_with_underscores(data):
    new_data = []
    for item in data:
        updated_data = {}
        for key, value in item.items():
            updated_key = sanitize_keys(key)
            updated_data[updated_key] = value
        new_data.append(updated_data)
    return new_data

def generate_summary_unstructured_data(query):
    messages = [{"role": "system", "content": SUMMARY_SYSTEM_PROMPT}, {"role":"user","content": query}]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        #tools=tools,
        #tool_choice="auto",  # auto is default, but we'll be explicit
    )
    #print('openai call resp',response.choices[0].message)
    return normalize_text(response.choices[0].message.content)

def process_request_codeinterpreter(client, file, query):
    response = ""
    prompt_instructions = {
  "Objective": "Act as an intelligent code interpreter interfacing with a Python REPL tool to perform data analysis on a dataset based on natural language queries.",
  "Dataset": "Access to a pandas DataFrame named 'dataset' with fields including Title,Contractenddate,Costsavings,Contractstartdate,Costavoidance,Type,Currency,Coverage,Category,Pastspent,Cost,Requestsubmitdate,Vendor,Status,Sentby,Sentto,Summary etc.",
  "Actions": [
    "Convert natural language queries into Python commands to manipulate the dataset.",
    "Use pandas operations such as filtering, grouping, and aggregating based on user queries."
  ],
  "Error Handling": [
    "Request clarifications for ambiguous queries.",
    "Ensure sensitive data is not exposed in outputs."
  ],
  "Performance Metrics": "Ensure quick and accurate execution of Python code based on the data in the DataFrame.",
  "Examples": [
    {
      "Query": "What was the total cost of all contracts in last month?",
      "Action": "dataset[dataset['Cost'].dt.month == (today.month - 1)]['Cost'].sum()"
    },
    {
      "Query": "How many new contracts did we acquire in the current year?",
      "Action": "dataset[(dataset['ContractstartDate'].dt.year == today.year) & ['Vendor'].nunique()"
    }
  ],
  "Execution Guidelines": [
    "Analyze the intent and required operations from the query.",
    "Translate the query into an executable Python snippet.",
    "Execute the code and return the results directly to the user.",
    "Maintain state for follow-up questions without repeating context."
  ]
}

    # Step 1: Create an Assistant
    assistant = client.beta.assistants.create(name="Central Procurement Data Analyst Assistant",
                                              instructions=prompt_instructions,
                                              model="gpt-4-1106-preview",
                                              tools=[{"type": "code_interpreter"}],
                                              file_ids=[file.id]
                                             )
    print(assistant)
    # Step 2: Create a Thread
    thread = client.beta.threads.create()
    # Step 3: Add a Message to a Thread
    message = client.beta.threads.messages.create(thread_id=thread.id,
                                                  role="user",
                                                  content= query)
    
    print("question", query)
    
    # Step 4: Run the Assistant
    run = client.beta.threads.runs.create(thread_id=thread.id,
                                          assistant_id=assistant.id,
                                          instructions=query)
    print(run.model_dump_json(indent=4))

    # Wait for 5 seconds
    time.sleep(50)
    # Retrieve the run status
    run_status = client.beta.threads.runs.retrieve(thread_id=thread.id,
                                                       run_id=run.id)
    print(run_status.model_dump_json(indent=4))
        
    # If run is completed, get messages
    if run_status.status == 'completed':
                messages = client.beta.threads.messages.list(
                thread_id=thread.id)
                # Loop through messages and print content based on role
                for msg in messages.data:
                     role = msg.role
                     response = msg.content[0].text.value
                     print(f"{role.capitalize()}: {response}")
                     break
                else:
                   print("Waiting for the Assistant to process...")
                   time.sleep(5)

    return response
    

#client = OpenAI(api_key='')

jsonArray = []
data = {}
columns_json = []
columns = []

# Define the csv-task tool

@app.route('/gettable', methods=['GET'])
def getTableResponse():
    question = request.args.get('convert')
    messages=[{'role': 'system', 'content':'You are a JSON content creator. Convert the given text in to Json with key value pairs. Always consider top5 relant keys only. Do not consider all the data if the keys are more than 5. Remember that the json should be created based on uniqueFor example if the given text as ```The dataframe has 30 rows then you have to give the json with key name as no.of contracts and the value as 30```with this context create the json for the given user request. do not include \n charactes in the response'},{'role': 'user', 'content':question}]
    print(question)
    chatRes = client.chat.completions.create(model='gpt-3.5-turbo',messages=messages,temperature=0)
    response = chatRes.choices[0].message.content
    return json.dumps(response)
               
@app.route('/upload', methods=['POST'])
def upload_excel_to_csv():
    try:
        # Check if the 'excelFile' field exists in the request
        if 'excelFile' not in request.files:
            return jsonify(message='No file provided'), 400
        print(request)

        excel_file = request.files['excelFile']

        # Save the uploaded Excel file temporarily
        #temp_path = 'temp.xlsx'
        # Clean up: remove the temporary Excel file
        #os.remove("output.csv")
        temp_path = 'output.csv'
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print("removed file")
              
        excel_file.save(temp_path)
        #df = pd.read_csv(temp_path)
        #global_df.add(df, ignore_index=True)
        print('File Uploaded to the server')
        #with open('output.csv', 'r', encoding='utf-8') as file:
    except Exception as e:
        return jsonify(error=str(e)), 500
    
    with open('output.csv', 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        csvReader = csv.DictReader(file)
        for row in csvReader:
            unique_id = str(uuid.uuid4())
            key = unique_id
            row['id'] = key
            summary = row["Description"] + row['Costevaluation']
            gpt_summary = generate_summary_unstructured_data(summary)
            row['Summary'] = gpt_summary
            row['Title'] = row['Title'].lower()
            row['Vendor'] = row['Vendor'].lower()
            row['Type'] = row['Type'].lower()
            row['Currency'] = row['Currency'].lower()
            row['Coverage'] = row['Coverage'].lower()
            row['Category'] = row['Category'].lower()
            row['Requestref']= row['Requestref'].lower()
            row['Status'] = row['Status'].lower()
            row['Sentby'] = row['Sentby'].lower()
            row['Sentto'] = row['Sentto'].lower()
            jsonArray.append(row)
        
    with open('data.json', 'w', encoding='utf-8') as jsonf:
         #data_sanitize = replace_space_with_underscores(jsonArray)
         #column_desc = get_column_data(data_sanitize)
         #print('processed Data', jsonArray)
         jsonf.write(json.dumps(jsonArray, indent=4))
         print("data json created")

    with open('data.json') as file:
         jsonData = json.load(file)
         df = pd.DataFrame(jsonData)
         df.to_csv('processed_data.csv', index=False)

    with open('processed_data.csv') as file:
         updatedf = pd.read_csv('processed_data.csv')
         columns_to_drop = ['Description', 'Remarks','Costevaluation','Source','Customresponse','Stage','Createdtime','id']
         updatedf.drop(columns=columns_to_drop, axis=1, inplace=True)
         updatedf.to_csv('dataframe.csv', index=False)
         print("processed the file and saved")


        # Read the Excel file into a pandas DataFrame
        #df = pd.read_excel(temp_path, engine='openpyxl')

        # Convert DataFrame to CSV
        #csv_path = 'output.csv'
        #df.to_csv(csv_path, index=False)

        # Clean up: remove the temporary Excel file
        #os.remove(temp_path)

    return jsonify(message=f'CSV file saved at {temp_path}'), 200

    

@app.route('/query', methods=['GET'])
def csvquery():
    question = request.args.get('question')
    #azurechatAI = openai.AzureOpenAI(azure_deployment="gpt-4-32k", api_version="2023-07-01-preview", azure_endpoint="https://rm-copilot.openai.azure.com",api_key='75c95b56f9344bb380e34ef51b8771aa')
    print(shared_value[0])
    chat = ChatOpenAI(model_name="gpt-4", openai_api_key='',temperature=0.4)
    #chat = ChatOpenAI(model="gpt-3.5-turbo", api_key='',temperature=0.2)
    #chat = OpenAI(model_name="gpt-4", openai_api_key='',temperature=0.2)
    df = pd.read_csv('./dataframe.csv')
    if (' ra ' in str(question).lower()):
        df = df[df['Category'] == 'it: resource augmentation']
        question = question + ' in the summary'
        print(df)
    if (' ras ' in str(question).lower()):
        df = df[df['Category'] == 'it: resource augmentation']
        question = question + ' in the summary'
    if (' hardware ' in str(question).lower()):
        df = df[df['Category'] == 'it: hardware']
        question = question + ' in the summary'
        print(df)
    if (' hiring ' in str(question).lower()):
        df = df[df['Category'] == 'it: resource augmentation']
        question = question + ' as per the summary field'
    if (' cpi ' in str(question).lower()) or (' cpi' in str(question).lower()) or (' gdp ' in str(question).lower()) or (' inflation ' in str(question).lower()):
        question = question + ' ,from the summary, cost and title'
    
    # Initialize the client
    #client = openai.OpenAI(api_key='')


    #file = client.files.create(
        #file=open("dataframe.csv", "rb"),
        #purpose='assistants'
       #)
    #response = process_request_codeinterpreter(client, file, str(question).lower())
    #print('code interpreter response', response)
    #df = pd.read_excel('output.xlsx')
    #csv_agent = create_csv_agent(OpenAI(temperature=0),'output.csv',verbose = True);
    customPrompt = '''
You are recognized as a Python Pandas expert, specializing in the analysis of procurement data in CSV format, with a focus on contract details, projects, applications, requests, or purchases:

CSV Structure: Your CSV includes fields such as Title, Contractenddate, Costsavings, Contractstartdate, Costavoidance, Type, Currency, Coverage, Category, Pastspent, Cost, Requestsubmitdate, Vendor, Status, Sentby, Sentto, and Summary.
Search Operations: Always perform search operations using the str.contains method with the search string in lowercase to ensure case-insensitive filtering in string data. Never use == operation in your Action.
Field Usage Restrictions: Never include Cost, CostAvoidance, Pastspent, Costsavings fields simultaneously in any calculation or action to prevent data sensitivity issues.
Date Formatting: The fields 'Contractstartdate' and 'Contractenddate' use date formats either as dd/mm/yyyy or dd/m/yyyy, representing the start and end dates of a project or contract.
Monetary Fields: The Costsavings, Costavoidance, Pastspent, and Cost fields relate to financial aspects of contracts and purchases, where:
Costsavings indicates dollar savings from purchases or maintenance.
Costavoidance and Pastspent relate to costs avoided and previous expenditures, respectively.
Cost signifies the dollar amount spent on specific procurements or contracts.
Categorical Distinctions:
'Category' defines service types such as IT: Hardware, IT: Software, and IT: Resource Augmentation.
'Type' details contract types like Purchase (Fixed Cost), Maintenance, and Escrow.
Keyword Searches: Primarily search for keywords in the 'Summary' column. If the keyword isn't found there, broaden the search to other relevant columns before extending to the entire dataset.
Vendor Filtering: When filtering by vendor, use the str.contains function on the 'Vendor' column.
Focus on Category and Type:
Filter using 'IT: Resource Augmentation' in the 'Category' column, not the 'Type' column.
Output Constraints: In your final output, avoid using the term "dataframe." Refer to the dataset as "CPO data."
Selective Data Display: Avoid returning all columns in your final answer. Instead, focus on using only the required column names. Exclude fields such as Requestref, Coverage, Currency, Sentby, and Sentto from your output.
Example Use Case:
Query: "How many contracts are there for Cognizant?"
Action Input: df[df['Vendor'].str.contains('Cognizant', case=False)]
Final Answer: "There are 21 contracts for Cognizant."
    '''
    
    agent =  create_pandas_dataframe_agent(chat,
                                           df,
                                           verbose=True,
                                           #prefix= PREFIX,
                                           #suffix= SUFFIX,
                                           #input_variables= Optional[List[str]] = None,
                                           handle_parsing_errors=True,
                                           #chain_type_kwargs = {
                                               #'prompt': customPrompt,
                                               #'input_variables': ['input','agent_scratchpad']
                                           #}
                                           #number_of_head_rows=10                                      
                                           )
    #print(agent)
    print("CPO agent ready to Servce");
    
    if question:
        query = str(question).lower()
        print('Question is', query)
        
        response =  agent.invoke(query)
        print('api response', response)
        # Convert the response string to a dictionary
        #response_dict = eval(response)  # Note: Be cautious when using eval; consider using safer alternatives

        # Extract the output value
        #output_value = response.get("output")

        #print("Captured output:", output_value)

        # Create a color palette (e.g., 'PuBu')
        #color_palette = sns.light_palette("blue", as_cmap=True)

        # Apply background gradient to the DataFrame
        #styled_df = df.style.background_gradient(cmap=color_palette)

        # Display the styled DataFrame using tabulate
        #print(tabulate(styled_df, headers='keys', tablefmt='psql'))

        #print(tabulate(response.output, headers='keys', tablefmt='psql'))
        #return tabulate(response, tablefmt='psql')
        # Split the string into words
        #words = output_value.split()

        # Print the words horizontally
        #print(" ".join(words))
        #out = " ".join(words)
        #print(tabulate(out, tablefmt='html'))
        #table = tabulate(output_value, tablefmt='grid')

        #return table
        return response        
      
    else:
       return "Hello! Please ask your question"
   
if __name__ == '__main__':
     app.run(host='127.0.0.1', port='8443', ssl_context='adhoc')