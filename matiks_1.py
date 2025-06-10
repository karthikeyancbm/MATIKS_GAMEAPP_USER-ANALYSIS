import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

# Loading the model:

with open("game_class_model_new.pkl","rb") as cls_file_2:
    game_cls_mdl_new = pickle.load(cls_file_2)

## Loading the model:

with open("regg_model_new.pkl","rb") as file_1_new:
    game_reg_mdl_new = pickle.load(file_1_new)

def revenue_pred(x):
    data = np.array([x])
    rev_pred = game_reg_mdl_new.predict(data)
    return f"{rev_pred[0]:.2f}"

def subscriber_pred(x):
    data = np.array([x])
    sub_pred = game_cls_mdl_new.predict(data)
    return f"The customer will subscribe for {rev_sub_dic[sub_pred[0]]}"

def year_revenue(df):
    year_df = df.groupby('Year')['Total_Revenue_USD'].sum().reset_index()
    return year_df

def year_rev_plot(df):
    year_df = df.groupby('Year')['Total_Revenue_USD'].sum().reset_index()
    yr = year_df['Year'].tolist()
    rev = year_df['Total_Revenue_USD'].tolist()
    fig,ax = plt.subplots(figsize=(6,4))
    sns.barplot(data=year_df,x=yr,y=rev,hue='Year',legend=False)
    ax.set_title('Yearwise Revenue')
    return fig
    
def month_revenue(df):
    month_df = df.groupby('Month')['Total_Revenue_USD'].sum().reset_index()
    return month_df

def month_rev_plot(df):
    month_df = df.groupby('Month')['Total_Revenue_USD'].sum().reset_index()
    mnth = month_df['Month'].tolist()
    rev = month_df['Total_Revenue_USD'].tolist()
    fig,ax = plt.subplots(figsize=(6,4))
    sns.barplot(data = month_df,x=mnth,y=rev,hue=mnth,legend=False)
    plt.xticks(rotation=90)
    ax.set_title('Monthwise Revenue')
    return fig

def day_revenue(df):
    day_df = df.groupby('Day')['Total_Revenue_USD'].sum().reset_index()
    return day_df

def day_revenue_plot(df):
    day_df = df.groupby('Day')['Total_Revenue_USD'].sum().reset_index()
    dy = day_df['Day'].tolist()
    rev = day_df['Total_Revenue_USD'].tolist()
    fig,ax = plt.subplots(figsize=(6,4))
    sns.barplot(data = day_df,x=dy,y=rev,hue=dy,legend=False)
    plt.xticks(rotation=90)
    ax.set_title('Daywise Revenue')
    plt.show()
    return fig

def get_dau(df):
    dau = df.groupby('Day')['Username'].size().reset_index()
    dau.rename(columns={'Username':'DAU'},inplace=True)
    df = df.sort_values('Day').reset_index(drop=True)
    return dau

def dau_plot(df):
    dau = df.groupby('Day')['Username'].size().reset_index()
    dau.rename(columns={'Username':'DAU'},inplace=True)
    day = dau['Day'].tolist()
    user = dau['DAU'].tolist()
    fig,ax = plt.subplots(figsize=(6,4))
    sns.lineplot(data=dau,x=day,y=user)
    ax.set_title('DAU')
    plt.show()
    return fig

def get_mau(df):
    mau = df.groupby('Month')['Username'].count().reset_index()
    mau.rename(columns={'Username':'MAU'},inplace=True)
    return mau

def mau_plot(df):
    mau = df.groupby('Month')['Username'].count().reset_index()
    mau.rename(columns={'Username':'MAU'},inplace=True)
    mnth = mau['Month'].tolist()
    user = mau['MAU'].tolist()
    fig,ax = plt.subplots(figsize=(6,4))
    sns.lineplot(data = mau,x=mnth,y=user)
    plt.xticks(rotation=90)
    ax.set_title('Monthly Active Users')
    plt.show()
    return fig

def user_revenue(df):
    user_rev_df = df.groupby('Username')['Total_Revenue_USD'].sum().reset_index().sort_values(by='Total_Revenue_USD',ascending=False)
    return user_rev_df

def countrywise_revenue(df):
    country_rev_df = df.groupby('Country')['Total_Revenue_USD'].sum().reset_index().sort_values(by='Total_Revenue_USD',ascending=False)
    return country_rev_df

def cls_count(df):
    cls_df = rev_sum['Class'].value_counts().reset_index()
    cls_df.rename(columns={'count':'Count'},inplace = True)
    return cls_df

def custmer_cls_plt(df):
    cls_df = rev_sum['Class'].value_counts().reset_index()
    cls_df.rename(columns={'count':'Count'},inplace = True)
    classes = cls_df['Class'].tolist()
    values = cls_df['Count'].tolist()
    explod = [0,0,0.2]
    fig,ax = plt.subplots(figsize=(2,2))
    plt.pie(values,labels=classes,explode=explod,autopct='%1.1f%%')
    plt.title('Customer Classification based on Revenue by customer')
    plt.show()
    return fig
    
def break_down(df):
    df_breakdown = df[['Device_Type','Subscription_Tier','Preferred_Game_Mode']].value_counts().reset_index(name='User_count').sort_values(by='User_count',ascending=False)
    return df_breakdown

def sub_cls(df):
    sub_cls_df = df_breakdown.groupby('Subscription_Tier')['User_count'].sum().reset_index(name='User_count').sort_values(by='User_count',ascending=False)
    return sub_cls_df

def sub_cls_plot(df):
    sub_cls_df = df_breakdown.groupby('Subscription_Tier')['User_count'].sum().reset_index(name='User_count').sort_values(by='User_count',ascending=False)
    sub_count = sub_cls_df['Subscription_Tier'].tolist()
    values = sub_cls_df['User_count'].tolist()
    explod = [0,0,0,0.2]
    fig,ax = plt.subplots(figsize=(2,2))
    plt.pie(values,labels=sub_count,explode=explod,autopct='%1.1f%%')
    plt.title('Customer Classfication based on Subscription')
    plt.show()
    return fig

def active_days(df):
    act_days_df = df[['Username','Active_time']].sort_values(by='Active_time',ascending=False)
    return act_days_df

def usage_freq(df):
    usage_freq_df = df['usage_segment'].value_counts().reset_index()
    return usage_freq_df

def usage_freq_plot(df):
    usage_freq_df = df['usage_segment'].value_counts().reset_index()
    fig,ax = plt.subplots(figsize=(2,2))
    sns.barplot(data=usage_freq_df,x='usage_segment',y='count',hue='usage_segment')
    plt.title('Usage Frequency')
    plt.show()
    return fig

def churn_data(df):
    churn_df = df['churn_risk'].value_counts().reset_index()
    churn_df['churn_risk'] = churn_df['churn_risk'].apply(lambda x: 'CHURNABLE' if x == True else 'NOTCHURNABLE')    
    return churn_df

def churn_risk_plot(df):
    churn_df = df['churn_risk'].value_counts().reset_index()
    churn_df['churn_risk'] = churn_df['churn_risk'].apply(lambda x: 'CHURNABLE' if x == True else 'NOTCHURNABLE')
    fig,ax = plt.subplots(figsize=(6,4))
    sns.barplot(data=churn_df,x='churn_risk',y='count',hue='churn_risk',legend=False)
    plt.title('Churn_risk Customers Count')
    plt.show()
    return fig

def sub_churn_risk(df):
    sub_churn_df = df.groupby(['Subscription_Tier','churn_risk']).size().unstack()
    return sub_churn_df

def sub_wise_churn_risk_plot(df):
    sub_churn_df = df.groupby(['Subscription_Tier','churn_risk']).size().unstack()
    fig,ax = plt.subplots(figsize=(6,4))
    sub_churn_df.plot(kind='bar',ax=ax)
    plt.title('Subscriptionwise Churn_risk')
    plt.show()
    return fig

def device_churn_risk(df):
    device_df = df.groupby(['Device_Type', 'churn_risk']).size().unstack()
    return device_df

def device_churn_risk_plot(df):
    device_df = df.groupby(['Device_Type', 'churn_risk']).size().unstack()
    fig,ax = plt.subplots(figsize=(6,4))
    device_df.plot(kind='bar',ax=ax)
    plt.title('Devicetype Churn_risk')
    plt.show()
    return fig

def game_churn_risk(df):
    game_churn_df = df.groupby(['Preferred_Game_Mode', 'churn_risk']).size().unstack()
    return game_churn_df

def game_churn_plot(df):
    game_churn_df = df.groupby(['Preferred_Game_Mode', 'churn_risk']).size().unstack()
    fig,ax = plt.subplots(figsize=(6,4))
    game_churn_df.plot(kind='bar',ax=ax)
    plt.title('Game_mode Churn_risk')
    plt.show()
    return fig

def frequency_segment(sessions_per_day):
    if sessions_per_day >= 1:
        return 'Daily'
    elif sessions_per_day >= 0.2:
        return 'Weekly'
    elif sessions_per_day > 0:
        return 'Occasional'
    else:
        return 'Inactive'

df = pd.read_csv("matiks_new.csv")

## Removal of unwatnted columns for ML
#df.drop(columns=['User_ID','Username','Email','Signup_Date','Last_Login'],inplace=True)

country_list = df['Country'].unique().tolist()

age_list = df['Age'].unique().tolist()

session_list = df['Total_Play_Sessions'].unique().tolist()

avg_session_list = df['Avg_Session_Duration_Min'].unique().tolist()

played_hours_list = df['Total_Hours_Played'].unique().tolist()

purchase_count_list = df['In_Game_Purchases_Count'].unique().tolist()

score_list = df['Achievement_Score'].unique().tolist()

active_time_list = df['Active_time'].unique().tolist()

revenue_list = df['Total_Revenue_USD'].unique().tolist()

gender_list = df['Gender'].unique().tolist()

device_list = df['Device_Type'].unique().tolist()

game_list = df['Game_Title'].unique().tolist()

subscription_list = df['Subscription_Tier'].unique().tolist()

referral_list = df['Referral_Source'].unique().tolist()

game_mode_list = df['Preferred_Game_Mode'].unique().tolist()

rank_tier_list = df['Rank_Tier'].unique().tolist()

rank_list = df['Rank_Tier'].unique().tolist()

coun_list = {col:i for i,col in enumerate(country_list,1)}

rev_count_list = {i:col for i,col in enumerate(country_list,1)}

gender_dic = {key :i for i,key in enumerate(gender_list,1)}

rev_gender_dic = {i:col for i,col in enumerate(gender_list,1)}

device_dic = {key : i for i,key in enumerate(device_list,1)}

rev_device_dic = {i:col for i,col in enumerate(device_list,1)}

game_dic = {key : i for i,key in enumerate(game_list,1)}

rev_game_dic = {i:col for i,col in enumerate(game_list,1)}

sub_dic = { key : i for i,key in enumerate(subscription_list,1)}

rev_sub_dic = {i:col for i,col in enumerate(subscription_list,1)}

referral_dic = {key:i for i,key in enumerate(referral_list,1)}

rev_referral_dic = {i:col for i,col in enumerate(referral_list,1)}

game_mode_dic ={ key:i for i,key in enumerate(game_mode_list,1)}

rev_game_mode_dic = {i:col for i,col in enumerate(game_mode_list,1)}

rank_tier_dic = {key : i for i, key in enumerate(rank_tier_list,1)}

rev_rank_tier_dic = {i:col for i,col in enumerate(rank_tier_list,1)}

df.drop(columns= ['Email'],inplace = True)

df['Signup_Date'] = pd.to_datetime(df['Signup_Date'])
df['Last_Login'] =pd.to_datetime(df['Last_Login'])
df['Year'] = df['Signup_Date'].dt.year
df['Month'] = df['Signup_Date'].dt.month_name()
df['Day'] = df['Signup_Date'].dt.day_name()
df['Date'] = df['Signup_Date'].dt.date
df['Date'] = pd.to_datetime(df['Date'],format='%m-%d-%y')
df['week'] = df['Signup_Date'].dt.to_period('W').dt.to_timestamp()
df['day'] = df['Signup_Date'].dt.to_period('D').dt.to_timestamp()
df['month'] = df['Signup_Date'].dt.to_period('M').dt.to_timestamp()
rev_sum = df.groupby('Username')['Total_Revenue_USD'].sum().reset_index().sort_values(by='Total_Revenue_USD',ascending=False)
rev_sum['Class'] = rev_sum['Total_Revenue_USD'].apply(lambda x:'Premium' if x >= 100 else 'Middle' if x >=50 else 'Low')
df_breakdown = df[['Device_Type','Subscription_Tier','Preferred_Game_Mode']].value_counts().reset_index(name='User_count').sort_values(by='User_count',ascending=False)
df['Active_days'] = df['Active_time']+1
df['Sessions_per_day'] = df['Total_Play_Sessions']/df['Active_days']
df['Sessions_per_week'] = df['Total_Play_Sessions']/(df['Active_days']/7)
df['Sessions_per_month'] = df['Total_Play_Sessions']/(df['Active_days']/30)
df['usage_segment'] = df['Sessions_per_day'].apply(frequency_segment)
df['churn_risk'] = (
    (df['Active_days'] <= 3) |
    (df['Total_Play_Sessions'] <= 2) |
    (df['Sessions_per_day'] < 0.2) |
    (df['Avg_Session_Duration_Min'] < 5) |
    (df['Total_Revenue_USD'] == 0)
)
df['High_value_customers'] = rev_sum[rev_sum['Class'] == 'Premium']['Class']

month_order = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]
df['Month'] = pd.Categorical(df['Month'],categories=month_order,ordered=True)
df = df.sort_values('Month').reset_index(drop=True)

weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df['Day']= pd.Categorical(df['Day'], categories=weekday_order, ordered=True)
df = df.sort_values('Day').reset_index(drop=True)

st._config.set_option('theme.base','dark')

st.set_page_config(layout='wide')
title_text = '''<h1 style='font-size: 55px;text-align: center;color:purple;background-color: lightgrey;'>Matiks - User Analysis</h1>'''
st.markdown(title_text, unsafe_allow_html=True)


with st.sidebar:

    select = option_menu("MAIN MENU",['HOME','ABOUT','EDA','PREDICTION'])

if select == 'HOME':

    pass

if select == 'ABOUT':

    pass

if select == 'EDA':

    st.markdown("""
<style>

	.stTabs [role="tab"] {font-size: 52px;
		gap: 2px;
    }

	.stTabs [role="tab"] {
		height: 45px;
        white-space: pre-wrap;
		background-color: #C0C0C0;
		border-radius: 4px 4px 0px 0px;
		gap: 10px;
    padding-top: 10px;
		padding-bottom: 10px;
    padding: 30px 40px;
    width: 800px;
    }

	.stTabs [aria-selected="true"] {
  		background-color: #C0C0C0;
      color:red;font-weight:bold;color:blue;
      font-size: 31px;
	}

</style>""", unsafe_allow_html=True)
  
    st.markdown("""
    <style>
    .stTabs [role="tab"] {
        font-size: 84px;font-weight: bold;
        padding: 10px 20px;color:blue;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b;  
        color: black;  
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    
    tab1,tab2,tab3,tab4,tab5 = st.tabs(['REVENUE - YEAR','REVENUE - MONTH','REVENUE - DAY','DAU - DAILY ACTIVE USERS','MAU - MONTHLY ACTIVE USERS'])

    tab6,tab7,tab8,tab9,tab10 = st.tabs(['REVENUE - USER & COUNTRY','CUSTOMER CLASS','BREAKDOWN','SUBSCRIPTION CLASSIFICATION','ACTIVE DAYS & USAGE FREQUENCY'])

    tab11,tab12,tab13,tab14 = st.tabs(['CHURNRISK CUSTOMERS','CHURNRISK - SUBSCRIPTION','CHURNRISK - DEVICE','CHURNRISK - GAMEMODE'])

    with tab1:

        col1,col2 = st.columns(2)

        if st.button('Submit',use_container_width= True):

            with col1:

                year = year_revenue(df)
                st.dataframe(year,width=600,height=150)

            with col2:
                year_plt = year_rev_plot(df)
                st.pyplot(year_plt)



    with tab2:

        col1,col2 = st.columns(2)

        if st.button('Month_Revenue',use_container_width= True):

            with col1:
                month_rev = month_revenue(df)
                st.dataframe(month_rev,width=600,height=480)
            with col2:
                month_rev_plt = month_rev_plot(df)
                st.pyplot(month_rev_plt)

    with tab3:

        col1,col2 = st.columns(2)

        if st.button('DAY_REVENUE',use_container_width= True):

            with col1:

                day_rev = day_revenue(df)
                st.dataframe(day_rev,width=600,height=300)

            with col2:

                day_rev_plt = day_revenue_plot(df)
                st.pyplot(day_rev_plt)

    with tab4:

        col1,col2 = st.columns(2)

        if st.button('DAILY ACTIVE USERS',use_container_width= True):

            with col1:

                dau_df = get_dau(df)
                st.dataframe(dau_df,width=600,height=500)

            with col2:

                dau_plt = dau_plot(df)
                st.pyplot(dau_plt)

    with tab5:

        col1,col2 = st.columns(2)

        if st.button('MONTHLY ACTIVE USERS',use_container_width= True):
        

            with col1:

                mau_df = get_mau(df)
                st.dataframe(mau_df,width=600,height=450)

            with col2:
                mau_df_plt = mau_plot(df)
                st.pyplot(mau_df_plt)


    with tab6:

        col1,col2 = st.columns(2)

        if st.button('SUBMIT',use_container_width= True):

            with col1:

                country = countrywise_revenue(df)
                st.dataframe(country,width=600,height=450)
            
            with col2:

                user_rev = user_revenue(df)
                st.dataframe(user_rev,width=600,height=450)

    with tab7:

        col1,col2 = st.columns(2)

        if st.button('CUSTOMER_CLASS',use_container_width= True):

            with col1:

                customer_class = cls_count(df)
                st.dataframe(customer_class,width=600,height=150)

            with col2:

                customer_class_plt = custmer_cls_plt(df)
                st.pyplot(customer_class_plt)

    with tab8:

        if st.button('BREAK_DOWN',use_container_width= True):

            break_down = break_down(df)
            st.dataframe(break_down,width=600,height=600)

    with tab9:

        col1,col2 = st.columns(2)

        if st.button('SUBSCRIPTION_CLASS',use_container_width= True):

            with col1:

                subscript = sub_cls(df)
                st.dataframe(subscript,width=600,height=180)

            with col2:

                subscript_plt = sub_cls_plot(df)
                st.pyplot(subscript_plt)

    with tab10:

        col1,col2 = st.columns(2)

        if st.button('ACTIVE_DAYS',use_container_width= True):

            with col1:

                active_day = active_days(df)
                st.dataframe(active_day,width=600,height=500)

            with col2:

                usage_frequency = usage_freq(df)
                st.dataframe(usage_frequency,width=600,height=180)

    with tab11:

        col1,col2 = st.columns(2)

        if st.button('CHURN_RISK CUSTOMERS',use_container_width= True):

            with col1:

                churnrisk_customers_count= churn_data(df)
                st.dataframe(churnrisk_customers_count,width=600,height=120)

            with col2:

                churnrisk_customers_count_plt = churn_risk_plot(df)
                st.pyplot(churnrisk_customers_count_plt)

    with tab12:

        col1,col2 = st.columns(2)

        if st.button('SUB_CHURN_RISK',use_container_width= True):

            with col1:

                subscription_churn_risk = sub_churn_risk(df)
                st.dataframe(subscription_churn_risk,width=600,height=180)

            with col2:

                subscription_churn_risk_plt = sub_wise_churn_risk_plot(df)
                st.pyplot(subscription_churn_risk_plt)

    with tab13:

        col1,col2 = st.columns(2)

        if st.button('DEVICE_CHURN_RISK',use_container_width= True):

            with col1:

                device_churn_risk_df = device_churn_risk(df)
                st.dataframe(device_churn_risk_df,width=600,height=150)

            with col2:

                device_churn_risk_plt  = device_churn_risk_plot(df)
                st.pyplot(device_churn_risk_plt)

    with tab14:

        col1,col2 =st.columns(2)

        if st.button('GAME_CHURN_RISK',use_container_width= True):

            with col1:

                game_mode_churnrisk_df = game_churn_risk(df)
                st.dataframe(game_mode_churnrisk_df,width=600,height=150)

            with col2:

                game_mode_churnrisk_plt = game_churn_plot(df)
                st.pyplot(game_mode_churnrisk_plt)

if select == 'PREDICTION':

        title_text = '''<h1 style='font-size: 32px;text-align: center;color:#00ff80;'>Revenue Prediction and Subscription Prediction</h1>'''
        st.markdown(title_text, unsafe_allow_html=True)

        st.markdown("""<style> .stTabs [role="tab"] {font-size: 32px;
		gap: 2px;
        }

        .stTabs [role="tab"] {
            height: 40px;
            white-space: pre-wrap;
            background-color: #C0C0C0;
            border-radius: 4px 4px 0px 0px;
            gap: 10px;
        padding-top: 10px;
            padding-bottom: 10px;
        padding: 30px 40px;
        width: 800px
        }

        .stTabs [aria-selected="true"] {
            background-color: #C0C0C0;
        color:red;font-weight:bold;
        font-size: 31px;
        }

        </style>""", unsafe_allow_html=True)
        
        st.markdown("""
            <style>
            .stTabs [role="tab"] {
                font-size: 64px;  
                padding: 10px 20px;  
            }
            .stTabs [aria-selected="true"] {
                background-color: #ff4b4b;  
                color: black;  
                font-weight: bold; 
            }
            </style>
            """, unsafe_allow_html=True)
        
        tab1,tab2 = st.tabs(['REVENUE PREDICTION','SUBSCRIBER PREDICTION'])

        with tab1:

                st.markdown("<h5 style=color:grey>To predict the Revenue, please provide the following information:",unsafe_allow_html=True)
                st.write('')

                col1,col2 = st.columns(2)

                with col1:

                    country = st.selectbox('Country',country_list,index=None)
                    age = st.selectbox('Age',age_list,index=None)
                    gender = st.selectbox('Gender',gender_list,index=None)
                    device = st.selectbox('Device_Type',device_list,index=None)
                    game = st.selectbox('Game_Title',game_list,index=None)
                    sessions = st.selectbox('Play_sessions',session_list,index=None)
                    avg_session = st.selectbox('Average_Session',avg_session_list,index=None)
                    play_hour = st.selectbox('Total_Playhours',played_hours_list,index=None)

                with col2:
                    purchase_count = st.selectbox('Game_Purchasescount',purchase_count_list,index=None)
                    subscription = st.selectbox('Subscription_Tier',subscription_list,index=None)
                    referral = st.selectbox('Referral_Source',referral_list,index=None)
                    game_mode = st.selectbox('Preferred_Game_Mode',game_mode_list,index=None)
                    rank = st.selectbox('Rank_Tier',rank_list,index=None)
                    score = st.selectbox('Achievement_Score',score_list,index=None)
                    active_time = st.selectbox('Active_time',active_time_list,index=None)

                    if st.button(':violet[Predict]',use_container_width=True):
                        #Encoding the chosen categorical columns
                        country = coun_list[country]
                        gender = gender_dic[gender]
                        device = device_dic[device]
                        game = game_dic[game]
                        subscription = sub_dic[subscription]
                        referral = referral_dic[referral]
                        game_mode = game_mode_dic[game_mode]
                        rank = rank_tier_dic[rank]                       

                         
                        data = [country,age,gender,device,game,sessions,avg_session,play_hour,purchase_count,
                                 subscription,referral,game_mode,rank,score,active_time]
                         
                        prediction = revenue_pred(data)
                        st.header(f"Predicted Revenue is : ${prediction}")
                

        with tab2:

                st.markdown("<h5 style=color:grey>To predict the Subscription, please provide the following information:",unsafe_allow_html=True)
                st.write('')

                col1,col2 = st.columns(2)

                with col1:
                     
                    Country = st.selectbox('country',country_list,index=None)
                    Age = st.selectbox('age',age_list,index=None)
                    Gender = st.selectbox('gender',gender_list,index=None)
                    Device = st.selectbox('device_Type',device_list,index=None)
                    Game = st.selectbox('game_Title',game_list,index=None)
                    Sessions = st.selectbox('play_sessions',session_list,index=None)
                    Avg_session = st.selectbox('average_Session',avg_session_list,index=None)
                    Play_hour = st.selectbox('total_Playhours',played_hours_list,index=None)
                
                with col2:
                     
                    Purchase_count = st.selectbox('game_Purchasescount',purchase_count_list,index=None)
                    Revenue= st.selectbox('total_Revenue_USD',revenue_list,index=None)
                    Referral = st.selectbox('referral_Source',referral_list,index=None)
                    Game_mode = st.selectbox('preferred_Game_Mode',game_mode_list,index=None)
                    Rank = st.selectbox('rank_Tier',rank_list,index=None)
                    Score = st.selectbox('achievement_Score',score_list,index=None)
                    Active_time = st.selectbox('active_time',active_time_list,index=None)

                    if st.button(':violet[Prediction]',use_container_width=True):
                         
                        Country = coun_list[Country]
                        Gender = gender_dic[Gender]
                        Device = device_dic[Device]
                        Game = game_dic[Game]
                        Referral = referral_dic[Referral]
                        Game_mode = game_mode_dic[Game_mode]
                        Rank = rank_tier_dic[Rank]                          

                         
                        data = [Country,Age,Gender,Device,Game,Sessions,Avg_session,Play_hour,Purchase_count,
                                 Revenue,Referral,Game_mode,Rank,Score,Active_time]
                         
                        prediction = subscriber_pred(data)
                        st.header(prediction)
                     
