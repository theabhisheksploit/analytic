import streamlit as st 
import pandas as pd 
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np 
from prophet import Prophet
import io
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import json
import time

# JLL brand colors
RED = '#E30613'
GRAY = '#58595b'
LIGHT_GRAY = '#939598'
Ocean = '#BCDEE6'
Sand = '#E0C6AD'
Space = '#003E51'
Midnight = '#131E29'
Dune = '#D1B9A7'
Azure = '#40798D'
Sea = '#95C6D5'
Orchid = '#955991'
meadow = '#A5C6A5'
Ceder = '#7D6F64'
Periwinkle = '#AABCF4'
Forest = '#497749'
Dusk = '#D0B5D0'
Glacier = '#667579'
Sky = '#0C7BA1'
Cloud = '#B6C0C2'
Success = '#08475E'

 # Configuration
googlegemini apikey = 

def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def call_jll_gpt_api(prompt):
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OKTA_ACCESS_TOKEN}",
        "Subscription-Key": SUBSCRIPTION_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "model": "GPT_35_TURBO",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 14000
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests_retry_session().post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                if e.response is not None and e.response.status_code == 500:
                    return "The JLL GPT API is currently experiencing issues. Please try again later."
                else:
                    return f"An error occurred while calling the API: {str(e)}"
            else:
                time.sleep(2 ** attempt)  # Exponential backoff

def get_chart_insights(chart_type, chart_data, x_axis=None, y_axis=None):
    prompt = f"Analyze this {chart_type} chart. "
    prompt += f"Data: {chart_data.to_json(orient='records')}\n"
    if x_axis:
        prompt += f"X-axis: {x_axis}\n"
    if y_axis:
        prompt += f"Y-axis: {y_axis}\n"
    prompt += "Provide insights about the data shown in this chart, including trends, patterns, or notable points."
    
    return call_jll_gpt_api(prompt)

def app():
    st.title(':red[📈Data Analytics Platform with AI Powered Insights]')
    st.subheader(':gray[Analyse your Data with ALL-IN-ONE Analytic Platform]', divider='rainbow')
    st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)
    
    files = st.file_uploader('Drop CSV & Excel Files', accept_multiple_files=True, type=['csv','xlsx','xls'])
    
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    if files:
        dfs = []  # List to store individual dataframes
        for file in files:
            with st.spinner(f'Loading and processing {file.name}...'):
                if file.name.endswith('csv'):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)
                dfs.append(df)
                st.info(f'{file.name} is successfully uploaded and processed', icon='✅')
        
        if len(dfs) == 1:
            st.subheader(f"Preview of {files[0].name}")
            st.dataframe(dfs[0].head())
            st.session_state.data = dfs[0]
        else:
            st.subheader(":red[Merge Options]")
            common_columns = set.intersection(*[set(df.columns) for df in dfs])
            if len(common_columns) > 0:
                merge_column = st.selectbox("Select column to merge on:", list(common_columns))
            else:
                st.warning("No common columns found across all files. Please select columns manually for merging.")
                merge_columns = []
                for i, df in enumerate(dfs):
                    col = st.selectbox(f"Select column from file {i+1} ({files[i].name}):", list(df.columns))
                    merge_columns.append(col)
                merge_column = st.text_input("Enter a name for the merged column:")
            
            if st.button("Merge Files"):
                if len(common_columns) > 0:
                    merged_data = dfs[0]
                    for df in dfs[1:]:
                        merged_data = pd.merge(merged_data, df, on=merge_column, how='outer')
                else:
                    for i, df in enumerate(dfs):
                        dfs[i] = df.rename(columns={merge_columns[i]: merge_column})
                    merged_data = dfs[0]
                    for df in dfs[1:]:
                        merged_data = pd.merge(merged_data, df, on=merge_column, how='outer')
                
                st.success('Files merged successfully!', icon='✅')
                st.dataframe(merged_data.head())
                st.session_state.data = merged_data
            
            st.subheader("File Previews")
            cols = st.columns(len(dfs))
            for i, df in enumerate(dfs):
                with cols[i]:
                    st.write(f"Preview of {files[i].name}")
                    st.dataframe(df.head())

    if st.session_state.data is not None:
        data = st.session_state.data

        st.subheader(':red[Data Issues Detection and Cleaning]', divider='rainbow')
        if st.button('Detect and Clean Data Issues'):
            with st.spinner('Analyzing and cleaning your data...'):
                missing_values = data.isnull().sum()
                duplicate_rows = data.duplicated().sum()
                inappropriate_types = {}
                for col in data.columns:
                    if data[col].dtype == 'object':
                        try:
                            pd.to_numeric(data[col])
                            inappropriate_types[col] = 'Numeric'
                        except:
                            pass
        
                data_cleaned = data.copy()
                data_cleaned = data_cleaned.drop_duplicates()
                
                numeric_columns = data_cleaned.select_dtypes(include=[np.number]).columns
                categorical_columns = data_cleaned.select_dtypes(exclude=[np.number]).columns

                filled_cells = pd.DataFrame(index=data_cleaned.index, columns=data_cleaned.columns, data=False)

                for col in numeric_columns:
                    mean_value = data_cleaned[col].mean()
                    filled_mask = data_cleaned[col].isnull()
                    data_cleaned.loc[filled_mask, col] = mean_value
                    filled_cells.loc[filled_mask, col] = True

                for col in categorical_columns:
                    mode_value = data_cleaned[col].mode().iloc[0] if not data_cleaned[col].mode().empty else 'Unknown'
                    filled_mask = data_cleaned[col].isnull()
                    data_cleaned.loc[filled_mask, col] = mode_value
                    filled_cells.loc[filled_mask, col] = True

                for col, new_type in inappropriate_types.items():
                    if new_type == 'Numeric':
                        data_cleaned[col] = pd.to_numeric(data_cleaned[col], errors='coerce')

                st.write("Data Cleaning Summary:")
                st.write(f"- Missing values found: {missing_values.sum()}")
                st.write(f"- Duplicate rows removed: {duplicate_rows}")
                st.write(f"- Columns with inappropriate types converted: {len(inappropriate_types)}")
                
                st.write("\nCleaned Data:")
                
                def highlight_filled(val):
                    return f'background-color: {JLL_Orchid}' if val else ''
                
                styled_df = data_cleaned.style.apply(lambda _: filled_cells.applymap(highlight_filled), axis=None)
                st.dataframe(styled_df)

                st.session_state.data = data_cleaned
                data = data_cleaned

        st.subheader(':red[Basic Information of your Dataset]', divider='rainbow')
        tab1, tab2, tab3, tab4= st.tabs(['Summary','Top & Bottom rows', 'Data Types', 'Columns'])
        with tab1:
            with st.spinner('Generating summary...'):
                st.write(f'There are {data.shape[0]} rows and {data.shape[1]} columns in your Dataset')
                st.subheader(':gray[Statistical Summary of your Dataset]')
                st.dataframe(data.describe())
        with tab2:
            with st.spinner('Loading rows...'):
                st.subheader(':gray[Top Rows]')
                toprows= st.slider('Number of rows you want', 0, 20, key='topslider')
                st.dataframe(data.head(toprows))
                
                st.subheader(':gray[Bottom Rows]')
                bottomrows= st.slider('Number of rows you want',0, 20, key= 'bottomslider')
                st.dataframe(data.tail(bottomrows))
        with tab3:
            with st.spinner('Analyzing data types...'):
                st.subheader(':gray[Data Types of your columns]')
                st.dataframe(data.dtypes)
        with tab4:
            with st.spinner('Listing columns...'):
                st.subheader(':gray[Columns name of your Dataset]')
                st.write(list(data.columns))

        st.subheader(':red[Column Value Counter]', divider='rainbow')
        with st.expander('Value Distribution Insights'):
            col1, col2 = st.columns(2)
            with col1:
                column= st.selectbox('Select Column to Analyze', options=list(data.columns))
            with col2:
                toprows= st.number_input('Display Top N Values', min_value= 1, step=1)
                countAnalyse= st.button('Generate Insights')
            if countAnalyse:
                with st.spinner('Generating insights...'):
                    resultAnalyse = data[column].value_counts().reset_index().head(toprows)
                    st.dataframe(resultAnalyse)
                    st.subheader('Visualization', divider='gray')
                    fig= px.bar(data_frame= resultAnalyse, x= column, y= 'count', text='count', template= 'plotly_white', color_discrete_sequence=[JLL_RED, JLL_Ocean, JLL_Sand, JLL_Space, JLL_Midnight, JLL_Dune, JLL_Azure, JLL_Sea, JLL_Orchid, JLL_meadow, JLL_Ceder, JLL_Periwinkle, JLL_Forest, JLL_Dusk, JLL_Glacier, JLL_Sky, JLL_Cloud, JLL_Success])
                    st.plotly_chart(fig)
                    fig= px.pie(data_frame= resultAnalyse, names=column, values= 'count', color_discrete_sequence=[JLL_RED, JLL_GRAY, JLL_LIGHT_GRAY, JLL_Ocean, JLL_Sand, JLL_Space, JLL_Midnight, JLL_Dune, JLL_Azure, JLL_Sea, JLL_Orchid, JLL_meadow, JLL_Ceder, JLL_Periwinkle, JLL_Forest, JLL_Dusk, JLL_Glacier, JLL_Sky, JLL_Cloud, JLL_Success])
                    st.plotly_chart(fig)

                    # Get AI insights
                    with st.spinner('Generating AI insights...'):
                        insights = get_chart_insights('multiple', resultAnalyse, x_axis=column, y_axis='count')
                        st.subheader(":grey[AI Insights]")
                        st.write(insights)

        st.subheader(':red[Analyse Your Data With Multi Columns]', divider='rainbow')
        with st.expander('Multi Column Distribution Analysis'):
            col1, col2, col3= st.columns(3)
            with col1:
                groupby_col= st.multiselect('Select Column to Analyze', options= list(data.columns))
            with col2:
                operation_col= st.selectbox('Select Column for Operations', options=list(data.columns))
            with col3:
                operation_val= st.selectbox('Select Operation to perform', options=['sum','max', 'min','mean', 'median', 'count'])
            
            if groupby_col:
                with st.spinner('Analyzing data...'):
                    operation_map = {
                        'sum': 'Total',
                        'mean': 'Average',
                        'max': 'Maximum',
                        'min': 'Minimum',
                        'median' : 'Median',
                        'count' : 'Count'
                    }

                    resultgroup = data.groupby(groupby_col).agg(
                        **{operation_map[operation_val]: (operation_col, operation_val)}
                    ).reset_index()
                    st.dataframe(resultgroup)

                    st.header(':gray[Visualise your Data]', divider='gray')
                    graph= st.selectbox('Choose your Graph Type', options=['line','bar','scatter','pie','sunburst'])
                    if graph == 'line':
                        x_axis= st.selectbox('Choose x axis', options= list(resultgroup.columns))
                        y_axis= st.selectbox('Choose y axis', options= list(resultgroup.columns))
                        color= st.selectbox('Color information', options= [None] + list(resultgroup.columns))
                        with st.spinner('Generating line chart...'):
                            fig = px.line(
                                data_frame=resultgroup,
                                x=x_axis,
                                y=y_axis,
                                color=color,
                                markers=True,
                                line_dash=color,
                                color_discrete_sequence=[JLL_Azure]
                            )
                            st.plotly_chart(fig)
                    elif graph == 'bar':
                        x_axis= st.selectbox('Choose x axis', options= list(resultgroup.columns))
                        y_axis= st.selectbox('Choose y axis', options= list(resultgroup.columns))
                        color= st.selectbox('Color information', options= [None] + list(resultgroup.columns))
                        facet_col= st.selectbox('Column Information', options= [None] + list(resultgroup.columns))
                        with st.spinner('Generating bar chart...'):
                            fig= px.bar(data_frame= resultgroup, x=x_axis, y= y_axis, color= color, facet_col= facet_col, barmode= 'group', color_discrete_sequence=[JLL_RED, JLL_GRAY, JLL_LIGHT_GRAY])
                            st.plotly_chart(fig)
                    elif graph == 'scatter':
                        x_axis= st.selectbox('Choose x axis', options= list(resultgroup.columns))
                        y_axis= st.selectbox('Choose y axis', options= list(resultgroup.columns))
                        color= st.selectbox('Color information', options= [None] + list(resultgroup.columns))
                        size= st.selectbox('Size column', options= [None] + list(resultgroup.columns))
                        with st.spinner('Generating scatter plot...'):
                            fig= px.scatter(data_frame= resultgroup, x= x_axis, y= y_axis, color= color, size= size, color_discrete_sequence=[JLL_RED, JLL_GRAY, JLL_LIGHT_GRAY])
                            st.plotly_chart(fig)
                    elif graph == 'pie':
                        values= st.selectbox('Choose your value', options= list(resultgroup.columns))
                        names= st.selectbox('Choose labels', options= list(resultgroup.columns))
                        with st.spinner('Generating pie chart...'):
                            fig= px.pie(data_frame= resultgroup, values= values, names= names, color_discrete_sequence=[JLL_RED, JLL_GRAY, JLL_LIGHT_GRAY])
                            st.plotly_chart(fig)
                    elif graph == 'sunburst':
                        path= st.multiselect('Choose your path', options= list(resultgroup.columns))
                        with st.spinner('Generating sunburst chart...'):
                            fig= px.sunburst(data_frame= resultgroup, path= path, values= operation_map[operation_val], color_discrete_sequence=[JLL_RED, JLL_GRAY, JLL_LIGHT_GRAY])
                            st.plotly_chart(fig)

                    # Get AI insights
                    with st.spinner('Generating AI insights...'):
                        insights = get_chart_insights(graph, resultgroup, x_axis, y_axis)
                        st.subheader("AI Insights")
                        st.write(insights)

        st.subheader(':red[Correlation Analysis]', divider='rainbow')
        with st.expander('Correlation Heatmap'):
            if st.button('Generate Correlation Heatmap'):
                with st.spinner('Generating correlation heatmap...'):
                    numeric_data = data.select_dtypes(include=[np.number])
                    corr_matrix = numeric_data.corr()
                    fig = px.imshow(corr_matrix, 
                                    color_continuous_scale='RdBu_r', 
                                    zmin=-1, zmax=1, 
                                    aspect="auto")
                    fig.update_layout(title='Correlation Heatmap')
                    st.plotly_chart(fig)

                                        # Get AI insights
                    with st.spinner('Generating AI insights...'):
                        insights = get_chart_insights('correlation heatmap', corr_matrix)
                        st.subheader("AI Insights on Correlation")
                        st.write(insights)

        st.subheader(':red[Distribution Analysis]', divider='rainbow')
        with st.expander('Distribution Plots'):
            col1, col2 = st.columns(2)
            with col1:
                numeric_column = st.selectbox('Select Numeric Column', options=data.select_dtypes(include=[np.number]).columns)
            with col2:
                plot_type = st.selectbox('Select Plot Type', options=['Histogram', 'Box Plot'])
            
            if st.button('Generate Distribution Plot'):
                with st.spinner('Generating distribution plot...'):
                    if plot_type == 'Histogram':
                        fig = px.histogram(data, x=numeric_column, color_discrete_sequence=[JLL_RED])
                    else:  # Box Plot
                        fig = px.box(data, y=numeric_column, color_discrete_sequence=[JLL_RED])
                    fig.update_layout(title=f'{plot_type} of {numeric_column}')
                    st.plotly_chart(fig)

                    # Get AI insights
                    with st.spinner('Generating AI insights...'):
                        insights = get_chart_insights(plot_type.lower(), data, x_axis=numeric_column if plot_type == 'Histogram' else None, y_axis=numeric_column if plot_type == 'Box Plot' else None)
                        st.subheader(f"AI Insights on {plot_type}")
                        st.write(insights)

        st.subheader(':red[Analyse Your Time Series]', divider='rainbow')
        with st.expander('Time Series Analysis'):
            col1, col2 = st.columns(2)
            with col1:
                date_col = st.selectbox('Select Date Column', options=list(data.columns))
            with col2:
                value_col = st.selectbox('Select Value Column', options=list(data.columns))
            
            if st.button('Perform Time Series Analysis'):
                with st.spinner('Performing time series analysis...'):
                    # Convert date column to datetime
                    data[date_col] = pd.to_datetime(data[date_col])
                    
                    # Prepare data for Prophet
                    df_prophet = data[[date_col, value_col]].rename(columns={date_col: 'ds', value_col: 'y'})
                    
                    # Create and fit the model
                    model = Prophet()
                    model.fit(df_prophet)
                    
                    # Create future dataframe for predictions
                    future = model.make_future_dataframe(periods=365)
                    
                    # Make predictions
                    forecast = model.predict(future)
                    
                    # Plot the forecast using Plotly
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='markers', name='Actual', marker=dict(color=JLL_GRAY)))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color=JLL_RED)))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(dash='dot', color=JLL_LIGHT_GRAY), name='Upper Bound'))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(dash='dot', color=JLL_LIGHT_GRAY), name='Lower Bound'))
                    fig.update_layout(title='Time Series Forecast', xaxis_title='Date', yaxis_title='Value')
                    st.plotly_chart(fig)
                    
                    # Plot the components using Plotly
                    trend = go.Figure()
                    trend.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], mode='lines', name='Trend', line=dict(color=JLL_RED)))
                    trend.update_layout(title='Trend', xaxis_title='Date', yaxis_title='Trend')
                    st.plotly_chart(trend)
                    
                    # Check for and plot available seasonalities
                    seasonalities = ['yearly', 'weekly', 'daily']
                    for seasonality in seasonalities:
                        if seasonality in forecast.columns:
                            seasonal = go.Figure()
                            seasonal.add_trace(go.Scatter(x=forecast['ds'], y=forecast[seasonality], mode='lines', name=f'{seasonality.capitalize()} Seasonality', line=dict(color=JLL_RED)))
                            seasonal.update_layout(title=f'{seasonality.capitalize()} Seasonality', xaxis_title='Date', yaxis_title='Seasonality')
                            st.plotly_chart(seasonal)
                    
                    # Show the forecast dataframe
                    st.write("Forecast Data:")
                    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

                    # Get AI insights
                    with st.spinner('Generating AI insights...'):
                        insights = get_chart_insights('time series', forecast, x_axis='ds', y_axis='yhat')
                        st.subheader("AI Insights on Time Series Forecast")
                        st.write(insights)

        st.subheader(':red[AI-Powered Analysis]', divider='rainbow')
        with st.expander('Get AI Insights'):
            user_prompt = st.text_area("Enter your question about the data:")
            if st.button('Get AI Insights'):
                with st.spinner('Generating AI insights...'):
                    # Prepare a context about the data
                    data_context = f"The dataset has {data.shape[0]} rows and {data.shape[1]} columns. "
                    data_context += f"Columns: {', '.join(data.columns)}. "
                    data_context += f"Here's a sample of the data: {data.head().to_json()}"
                    
                    # Combine user prompt with data context
                    full_prompt = f"{data_context}\n\nUser question: { user_prompt + 'never give code and if ask then give me data in sheet format and only give solutions as output never give suggesstions' }"
                    
                    # Call the API
                    ai_response = call_jll_gpt_api(full_prompt)
                    
                    # Display the response
                    st.write("AI Insights:")
                    st.write(ai_response)

    else:
        st.info("Please upload files to begin analysis.")

if __name__ == "__main__":
    app()
