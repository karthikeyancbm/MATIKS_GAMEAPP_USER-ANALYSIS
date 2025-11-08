import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import streamlit_option_menu
from streamlit_option_menu import option_menu
import math


st.set_page_config(layout='wide')


pdm_file = st.file_uploader("Upload Padam File",type=["csv", "xlsx"])

if st.button('SUBMIT',type='primary'):
    

    if pdm_file !=  None:

        if "pdm_file" not in st.session_state:

            st.session_state["pdm_file"] = pdm_file

        if pdm_file.name.endswith(".csv"):

            df = pd.read_csv(pdm_file,dtype=str)

        else:

            df = pd.read_excel(pdm_file,dtype=str)

            columns_lst = ['Document Date','Crdit Card No', 'Autho. No', 'Amt.Recd.On Card']

            lst = []
            for col in df.columns:
                if df[col].astype(str).str.contains('S.No|Doc',na=False).any():
                    lst.append(col)

            start_index = df[df[lst[0]].str.contains('S.No|Doc',na=False)].index.tolist()
            start_index = int(start_index[0])

            
            cols = df.loc[start_index].tolist()

            df.columns = cols

            df.columns = [str(col).strip() for col in df.columns]

            df = df.loc[start_index:]

            df.drop(index=start_index,inplace=True)

            df = df.reset_index(drop=True)

            for i in df.columns:
                if i not in lst:
                    df.rename(columns={'Card No.':'Crdit Card No','Autho.No.':'Autho. No','Amount':'Amt.Recd.On Card'},inplace=True)

            cols_lst = df.columns.drop(['Document Date','Crdit Card No','Autho. No','Amt.Recd.On Card']).tolist()

            df.drop(columns=cols_lst,inplace=True,errors='ignore')            
            
            df = df.reset_index(drop=True)
            ind_lst =df.index.tolist()
            last_index = ind_lst[-1]
            df.drop(index=last_index,inplace=True)
            df['card_auth'] = df['Crdit Card No']+df['Autho. No']            
            df['Discrepancy'] = df.apply(lambda x: 'Yes' if (len(str(x['Crdit Card No']))<4) or (len(str(x['Autho. No']))<6)  else 'No',axis=1)
            df['Amt.Recd.On Card'] = pd.to_numeric(df['Amt.Recd.On Card'],errors='coerce')
            df['Amt.Recd.On Card'] = df['Amt.Recd.On Card'].astype(float)
            st.write(df['Amt.Recd.On Card'].sum())            
            st.session_state["pdm_df"] = df

if "pdm_df" in st.session_state:
    st.subheader("Padam File Data")
    st.dataframe(st.session_state["pdm_df"])
    st.info(f"{st.session_state['pdm_df'].shape[0]} : Rows")
    st.info(f"Padam Amount:{math.ceil(st.session_state["pdm_df"]['Amt.Recd.On Card'].sum())}")



fed_file = st.file_uploader('Upload the file',accept_multiple_files=True,type=['xlsx'])


if st.button('submit',type='primary'):

    if fed_file != None:

        if "fed_file" not in st.session_state:
            
            st.session_state["fed_files"] = fed_file

            if fed_file:
                dfs=[]
                for upload in fed_file:
                    df1 = pd.read_excel(upload)
                    dfs.append(df1)
                
                df1 = pd.concat(dfs,ignore_index=True)   

                fed_col_lst = df1.columns.drop(['PAID Date','Channel Type','Card Number','Auth Code','Txn Amt','MSF','IGST','NET']).tolist()

                df1.drop(columns=fed_col_lst,inplace=True)

                df1.drop(columns=['Channel Type','MSF','IGST','NET'],inplace=True)

                df1['Card Number'] = df1['Card Number'].apply(lambda x:x[-4:])

                df1['Card Number'] = df1['Card Number'].astype(str)
                df1['Auth Code'] = df1['Auth Code'].astype(str)

                df1['card_auth'] = df1['Card Number']+df1['Auth Code']
                st.session_state['df1'] = df1
                    
if 'df1' in st.session_state:
    st.info("Federal Statement Data")
    st.dataframe(st.session_state['df1'])
    st.info(f'Federal Amount : Rs{st.session_state['df1']['Txn Amt'].sum()}')

if st.button('Federal_Match',type='primary'):

    if 'df1' not in st.session_state:

        st.session_state['df1'] = df1

    if 'pdm_df' not in st.session_state:

        st.session_state['pdm_df'] = df
    

    merged = pd.merge(st.session_state['pdm_df'],st.session_state['df1'],on='card_auth',how='inner')

    unmatch_fed_df = st.session_state['df1'][~st.session_state['df1']['card_auth'].isin(st.session_state['pdm_df']['card_auth'])]

    st.session_state['merged'] = merged

    st.session_state['unmatch_fed_df'] = unmatch_fed_df


if 'merged' in st.session_state:
    st.dataframe(st.session_state['merged'])   
    st.info(f"Federal Matched Amount :{st.session_state['merged']['Amt.Recd.On Card'].sum()}")
    st.info('Padam File after Removal of Federal Matches')
    del_lst = st.session_state['merged']['card_auth'].tolist()
    fed_ind_lst =st.session_state['pdm_df'][st.session_state['pdm_df']['card_auth'].isin(del_lst)].index.tolist()
    st.session_state['pdm_df'].drop(index=fed_ind_lst,inplace=True)
    st.dataframe(st.session_state['pdm_df'])
    st.info(f"Padam Amount: Rs.{st.session_state['pdm_df']['Amt.Recd.On Card'].sum()}")
    st.info('Federal Unmatched')
if 'unmatch_fed_df' in st.session_state:
    st.dataframe(st.session_state['unmatch_fed_df'] )
    st.info(f"Federal Unmatched Amount : Rs. {st.session_state['unmatch_fed_df'] ['Txn Amt'].sum()}")


hd_file = st.file_uploader('Upload the files',accept_multiple_files=True,type=['xlsx'])

if st.button('Submit',type='primary'):
     if hd_file:
        dfs=[]
        for upload in hd_file:
            df2 = pd.read_excel(upload)
            dfs.append(df2)
        
        df2 = pd.concat(dfs,ignore_index=True)   

        cols_2 = df2.columns.drop(['CARDNBR','APP_CODE','PYMT_CHGAMNT']).tolist()

        df2.drop(columns=cols_2,inplace=True)

        df2['CARDNBR'] = df2['CARDNBR'].astype(str)
        df2['CARDNBR'] = df2['CARDNBR'].apply(lambda x:x[-4:])
        df2['CARDNBR'] = df2['CARDNBR'].astype(str)
        df2['APP_CODE'] = df2['APP_CODE'].astype(str)
        df2['card_auth'] = df2['CARDNBR']+df2['APP_CODE']

        st.session_state['df2'] = df2
                    
if 'df2' in st.session_state:
    st.info("HDFC Statement Data")
    st.dataframe(st.session_state['df2'])
    st.info(f'HDFC Amount : Rs. {st.session_state['df2']['PYMT_CHGAMNT'].sum()}')


if st.button('HDFC_Match',type='primary'):

    if 'df2' not in st.session_state:

        st.session_state['df2'] = df2

    if 'pdm_df' not in st.session_state:

        st.session_state['pdm_df'] = df

    merged_1 = pd.merge(st.session_state['pdm_df'],st.session_state['df2'],on='card_auth',how='inner')

    unmatch_hd_df = st.session_state['df2'][~st.session_state['df2']['card_auth'].isin(st.session_state['pdm_df']['card_auth'])]
    
    st.dataframe(merged_1)
    st.info(f"HDFC Matched Amount :{merged_1['Amt.Recd.On Card'].sum()}")
    st.info('Padam File after Removal of HDFC Matches')
    del_lst_1 = merged_1['card_auth'].tolist()
    hd_ind_lst = st.session_state['pdm_df'][st.session_state['pdm_df']['card_auth'].isin(del_lst_1)].index.tolist()
    st.session_state['pdm_df'].drop(index=hd_ind_lst,inplace=True)
    st.dataframe(st.session_state['pdm_df'])
    st.info(f"Padam Final Amount :{st.session_state['pdm_df']['Amt.Recd.On Card'].sum()}")
    st.info("Unmatched HDFC File")
    st.dataframe(unmatch_hd_df)
    st.info(f"HDFC Unmatched Amount : Rs. {unmatch_hd_df['PYMT_CHGAMNT'].sum()}")



