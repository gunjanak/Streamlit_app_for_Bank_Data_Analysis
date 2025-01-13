import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import linear_model
import statsmodels.api as sm
from itertools import combinations
import torch
import torch.nn as nn
from torch.optim import Adam

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def accuracy(Y,z):
  #returs accuracy and y_hat
  y_hat = torch.round(z.data)
  is_correct = y_hat==Y
  is_correct = is_correct.numpy().tolist()
  # print(is_correct)
  val_epoch_accuracies = np.mean(is_correct)
  Y_hat = y_hat.data.numpy()
  list_to_return = [val_epoch_accuracies,y_hat]
  return list_to_return


class MyNeuralNet(nn.Module):
  def __init__(self):
    super().__init__()
    #define the hidden layer
    self.input_to_hidden_layer = nn.Linear(2,8)
    #define activation function of hidden Layer
    self.hidden_layer_activation = nn.ReLU()

    #define hidden layer
    self.hidden_layer_to_hidden_layer = nn.Linear(8,8)

    #define ourput layer
    self.hidden_to_output_layer = nn.Linear(8,1)

    #define activation function to ouput layer
    self.output_layer_activation = nn.Sigmoid()

  #Define feed forward network based on above definitions
  def forward(self,x):
    x = self.input_to_hidden_layer(x)
    x = self.hidden_layer_activation(x)
    x = self.hidden_layer_to_hidden_layer(x)
    x = self.hidden_layer_activation(x)
    x = self.hidden_to_output_layer(x)
    x = self.output_layer_activation(x)
    return x

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


def for_pytorch():
    the_file = "circles.csv"
    circles = pd.read_csv(the_file)
    x1 = circles['X1'].values
    x2 = circles['X2'].values
    y = circles['Y'].values
    #combining two inputs in one
    x = [x1,x2]
    X = torch.Tensor(x)
    Y = torch.Tensor(y)
    print(f"Size of X: {X.size()}")
    #Trasposing input
    X = torch.transpose(X,0,1)
    print(f"New size of X: {X.size()}")

    #preparing output for neural network
    print(f"Size of Y: {Y.size()}")
    Y = torch.unsqueeze(Y,1)
    print(f"Size of Y: {Y.size()}")
    
    
    mynet = MyNeuralNet()
    
    
    #Binary cross entropy
    loss_func = torch.nn.BCELoss()
    
    
    opt = Adam(mynet.parameters(),lr=0.01)
    
    loss_history = []
    accuracy_history = []
    interval_prediction = []
    epochs = 101

    for epoch in range(epochs):
        opt.zero_grad()

        #feeding data to network
        z = mynet(X)

        #calculating accuracy
        acc = accuracy(Y,z)

        #Accuracy after each epoch
        accuracy_epoch = acc[0]

        #Network prediction after each epoch
        y_hat = acc[1]

        #Calculating loss
        loss_value = loss_func(z,Y)

        #backpropagation
        loss_value.backward()

        #perform a single optimization step (parameter update)
        opt.step()

        #converting loss_value to numpy
        loss_value = loss_value.data.numpy()

        #Appending loss_value to loss_history
        loss_history.append(loss_value)

        #Appending accuracy_epoch to accuracy_history
        # st.write(accuracy_epoch)
        accuracy_history.append(accuracy_epoch)

        #collecting predicted values after 10th epoch
        if(epoch%10 == 0):
            interval_prediction.append(y_hat)
    
    # Create the scatter plot
    fig, ax = plt.subplots()
    ax.scatter(X[:,0],X[:,1],c=Y, alpha=0.5)
    ax.set_title("Scatter Plot")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    

    # Display the plot in Streamlit
    st.pyplot(fig)
    
    # Create the figure
    epoch = st.slider("Select Epoch", 0, 100, 0, step=10)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(X[:, 0], X[:, 1], c=interval_prediction[epoch // 10], cmap='viridis')
    ax.set_title(f"After {epoch} epochs")
    st.pyplot(fig)
   
        
    
    
    
    
    
    
    
def main():

    global output_df
    st.write(torch.__version__)
    for_pytorch()
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
            st.write("Check values")
            st.write(check_values)
            if (check_values):
                we_found_our_model = True
                st.write("P-values for all parametes are below 0.05")
                st.write(model.params)
                st.write(model.params[0])
                #final_euqation = f'ROE = {round(model.params[0],3)}{round(model.params[1],3)}x ({main_parameters[0]}){round(model.params[2],3)}x ({main_parameters[1]}){round(model.params[3],3)}x ({main_parameters[2]})'
                #st.write(final_euqation)

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