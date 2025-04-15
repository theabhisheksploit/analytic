import streamlit as st  
from pandasai import Agent
from pandasai import SmartDataframe
from pandasai.responses.response_parser import ResponseParser
import pandas as pd
import numpy as np
import os
from pandasai.llm.local_llm import LocalLLM
from openai import OpenAI

def app():
    st.title(':red[🤖 Analyze Your Data with AI]')
    st.subheader(":gray[Don't know How to use analytic Tool, then use our AI to analyse your data  ]", divider='rainbow')
    st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)
    file = st.file_uploader('Drop CSV, Excel & Json File to analyse with our AI', type=['csv', 'xlsx', 'json', 'GeoJson'])
    st.info("Generative Data Quality & Generative Data Analysis will Helps in Taking Data driven Decision with AI Power")
    if file is not None:
            if file.name.endswith('csv'):
                data = pd.read_csv(file)
            elif file.name.endswith('xlsx'):
                data = pd.read_excel(file)
            else:
                data = pd.read_json(file)
            st.info('Your file is successfully uploaded', icon='🚨')
            st.dataframe(data)
            st.subheader(":rainbow[Let's Analyse]", divider='rainbow')

            model_choice = st.selectbox("Choose Your AI Model", ["None", "With PandasAI Key","JLL GPT 3.5 Turbo 16K"]) #, "GPT 4 8K", "GPT 4 TURBO","Baidu Ernie 4.0","GPT 4o"

            if model_choice == "With PandasAI Key":
                openai_api_key = st.text_input("Put Your AI API Key", key="chatbot_api_key", type="password")
                if openai_api_key:
                    os.environ['PANDASAI_API_KEY'] = openai_api_key
                    try:
                        agent = SmartDataframe(data)
                        st.subheader("💬 Now You can Start Analysing Your Data with the power of AI ")
                        prompt = st.text_input("Enter your prompt:")
                        if st.button("Generate"):
                            if prompt:
                                with st.spinner("Generating response..."):
                                    try:
                                        response = agent.chat(prompt)
                                        st.write(response)
                                    except Exception as e:
                                        st.error(f"An error occurred: {e}")
                            else:
                                st.info("Please enter a prompt.")
                    except pd.errors.EmptyDataError:
                        st.error("The uploaded file is empty.")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                else:
                    st.warning("Please add your OpenAI API key to continue.")
            
            elif model_choice == "JLL GPT 3.5 Turbo 16K":
                try:
                    BASE_URL = "https://api-prod.jll.com/csp/chat-service/api/v1"
                    OKTA_ACCESS_TOKEN = "eyJraWQiOiJuWkV1WFh4emNCVFhvSFVnN1hnSEUwU282SlZldFNKQ09KUXhmVDVQVGEwIiwiYWxnIjoiUlMyNTYifQ.eyJ2ZXIiOjEsImp0aSI6IkFULkliYUVRcVVkbGZLeW03cUozVG9GWExoclY2bVQwZzVVLV9OdWJVS0RiZlkiLCJpc3MiOiJodHRwczovL2psbC5va3RhLmNvbS9vYXV0aDIvYXVzMWh3cmluaHFPUXBGOG4waDgiLCJhdWQiOiJhcGk6Ly9kZWZhdWx0IiwiaWF0IjoxNzI5MTU2OTgzLCJleHAiOjE3MjkxNjA1ODMsImNpZCI6IjBvYTFza3lpcGV0MXRxUjJKMGg4IiwidWlkIjoiMDB1MXd6ODhjemJIaGh6SVIwaDgiLCJzY3AiOlsib3BlbmlkIiwicHJvZmlsZSIsImVtYWlsIl0sImF1dGhfdGltZSI6MTcyOTE1Njk4MCwic3ViIjoiQWJoaXNoZWsuR3VwdGExQGpsbC5jb20iLCJBREdVSUQiOiJCRnpFZHBrdUxrYTE5SzVzZnIySlVRPT0iLCJjb25zdW1lcmNsYWltIjoib2t0YWF1dGhzZXJ2ZXI9YXBpbTtqbGxzeXN0ZW11c2VyPWFwaS1jb25zdW1lci1jaGF0Z3B0LW9pZGM7cHNpZD1KTExHUFRAU0VSVklDRTsifQ.nqWppa6LFrzIpkusEuHXiPqCW2_-FtsiXH8Z9duDGSnogDyhmM9kN56cc1h_TSzUo8R8HFD8i-njt3juZU7xIiHSo2UcUSPuDGcxiEFt0L8zh7ILjrd4sqV6zyYk6a7KgeruCYpgN62Fbr_odJK8IvIT68DOgSv52zpzfr8toaY0zLpKyZxgftZxk14I2OmGc_l6ko3JrQN6bnB2bd0QqLjl9KGzkoakZOlhy8gPRWV84dS94dyIo1kPXUdoBQUlvms3jS8WPKFAzt6VYM6qYZzMH3eniPRxhPtjSkw2b2fMRdyknWqzWHFfYtpPscKeVETdUqBP55W9CCc2PyZRlg"
                    SUBSCRIPTION_KEY = "4c58287ee89d47afa78c202ec3f092ab"
                    task = st.text_input("Enter your data analysis or cleaning task:")
                    if st.button("Magic Button"):
                        client = OpenAI(
                        api_key=OKTA_ACCESS_TOKEN,
                        base_url=BASE_URL,
                        default_headers={"Subscription-Key": SUBSCRIPTION_KEY}
                            )
                        chat_completion = client.chat.completions.create(
                        messages=[
                        {"role": "user", "content": f"Analyze or clean the following data, and fill missing value for int column with mean and categorical column with median and only give clean data, never give code and give me data in sheet format:\n```\n{data.head().to_markdown(index=False)}\n```\nTask: {task}"},
                        ],
                        model="GPT_35_TURBO",
                        max_tokens=15000
                        )
                        response_text = chat_completion.choices[0].message.content
        
                        st.write(response_text)
                        st.dataframe(response_text)
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")

