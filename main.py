import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import linear_model
import statsmodels.api as sm
from itertools import combinations


import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def load_data():
    st.write("Upload a csv file")
    uploaded_file = st.file_uploader("Choose a file",'csv')
    use_example_file = st.checkbox("Use example file",False,help="Use in-built example file for demo")

    status = False
    if use_example_file:
        uploaded_file = "default_file.csv"
        status = True
    
    if uploaded_file:
        #st.write(uploaded_file)
        if(uploaded_file == None):
            status = False
        else:
            status = True
    to_return = [uploaded_file,status]

    return to_return

def read_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df


def main():

    global output_df
    st.title("Bank Data Analysis")
    status = False
    set_index  = False
    basic_done = False
    #st.write(status)
    uploaded_file,status = load_data()
    #st.write(status)
    if(status == True):
        df = read_data(uploaded_file)
        st.write(df.head())
        st.write('Choose a column to set as an index. Data of column must be in date_time ')
        col_list = list(df.columns.values)
        col_list.insert(0,'None')
        indexx = st.selectbox('Which column you want as index:',col_list)

        st.write('You Selected: ',indexx)

        try:
            df[indexx] = pd.to_datetime(df[indexx])
            set_index = True
        except:
            set_index = False
            st.write("Error select proper column")

        if(set_index == True):
            df = df.set_index(indexx)
            st.write(df.head())
            #st.write("Current frequency")
            #st.write(df.index.freq)
            try:
                st.line_chart(df,width=1000,height=500)
            except:
                st.write('Error, Choose another column')
            basic_done = True

        if(set_index == True):
            st.header("Description of Data")
            st.write(df.describe())

            st.header("Plotting")
            column = st.radio("Choose a column to plot",df.columns)

            fig = plt.figure(figsize=(10,5))
            
            sns.lineplot(data=df[column])
            sns.set_theme()
            st.pyplot(fig)

            st.header("Plotting data from multiple columns")
            options = st.multiselect('Choose Multiple columns',df.columns)
            

            try:
                st.write(options[0])
                options = list(options)
                fig = plt.figure(figsize=(10,5))
                sns.lineplot(data=df[options])
                sns.set_theme()
                fig2 = px.line(df[options])
                st.plotly_chart(fig2)
                

                #st.pyplot(fig2)
            except:
                pass

            st.header("Correlation")
            st.write(df.corr(numeric_only=True))

            st.subheader("Heat Map")
            #fig = plt.figure(figsize=(10,5))
            #sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='Blues')
            fig = px.imshow(df.corr(numeric_only=True), text_auto=True,template='ggplot2')
            fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20)
            )

            st.plotly_chart(fig)

            st.subheader("Pair Plot")
            fig = plt.figure(figsize=(10,5))
            fig = sns.pairplot(df,height=2.5)
            
            st.pyplot(fig)

            st.header("Regression")
            st.write("Collecting all parameters for which p-value is less than or equal to 0.05")
            columns = df.columns
            
            main_parameters = []
            for col in df.columns:
                df_x = df[[col]]
                x = df_x
                y = df['ROE']
                
                # with sklearn
                regr = linear_model.LinearRegression()
                regr.fit(x, y)

                

                # with statsmodels
                x = sm.add_constant(x) # adding a constant
                
                model = sm.OLS(y, x).fit()
                predictions = model.predict(x) 
                
                if(model.pvalues[1] <= 0.05):
                    print("\n")
                    if(col != 'ROE'):
                        main_parameters.append(col)
                        st.write(f'Parameter : {col} is have significant impact on ROE')
                        st.write('Intercept: \n', regr.intercept_)
                        st.write('Coefficients: \n', regr.coef_)
                        st.write(f'fvalue:{model.fvalue}')
                        st.write(f'pvalue : {model.pvalues[1]}')
                    
                        print("...........................................")
                else:
                    st.write(f'Parameter : {col}, P-Value :{model.pvalues[1]} do not have significant impact on ROE')
                    print("......................................................................")


                print("...........................................")
            
            st.write("The significant parameters",main_parameters)


            st.subheader("Drawing Regression Line")
            for col in df.columns:
                if col != "ROE":
                    
                    fig = sns.lmplot(x=col,y='ROE',data=df,fit_reg=True,height=4, aspect=1)
                    sns.set_theme()
                    st.pyplot(fig)

            st.header("Mutiple Regression")
            st.subheader("Taking All significant parameters")
            st.write("We take all significant parameters and try to build a multiple regression model")
            st.write(df[main_parameters])
            x = df[main_parameters]
            y = df['ROE']

            x = sm.add_constant(x)
            model = sm.OLS(y,x).fit()
            
            st.write(model.summary())

            st.write(model.pvalues)
            st.write("Checking if all pvalues are below 0.05")
            check_values = lambda lst: all(value < 0.05 for value in lst)
            if (check_values):
                we_found_our_model = True
                st.write("P-values for all parametes are below 0.05")
                st.write(model.params)
                st.write(model.params[0])
                final_euqation = f'ROE = {round(model.params[0],3)}{round(model.params[1],3)}x ({main_parameters[0]}){round(model.params[2],3)}x ({main_parameters[1]}){round(model.params[3],3)}x ({main_parameters[2]})'
                st.write(final_euqation)

            if(we_found_our_model == False):

                st.subheader("Taking Two Parameters at a time")

                res = list(combinations(main_parameters, 2))
                for p1,p2 in res:
                    df_x = df[[p1,p2]]
                    x = df_x
                    y = df['ROE']

                    # with sklearn
                    regr = linear_model.LinearRegression()
                    regr.fit(x, y)

                    # with statsmodels
                    x = sm.add_constant(x) # adding a constant

                    model = sm.OLS(y, x).fit()
                    predictions = model.predict(x) 
                    st.write(f"First parameter: {p1}, Second parameter: {p2}")
                    st.write(model.summary())
                    st.write("......................................................................")
                    st.write("\n\n\n\n")







if __name__ == '__main__':
    main()