import streamlit as st
import os
import pickle
import joblib
import pandas as pd
#import numpy as np
import glob
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from padelpy import padeldescriptor
import pathlib 
from pathlib import Path
from shutil import which
from os.path import abspath, dirname, join
import subprocess
import sklearn
import numpy as np

#page_configuration

st.set_page_config(
  page_title='acbr_aankalan',
  page_icon='🧬',
  initial_sidebar_state='expanded')


_PADEL_PATH = join(
    dirname(abspath(__file__)),
    'PaDEL-Descriptor',
    'PaDEL-Descriptor.jar'     
)
#subprocess.call(['java', '-jar', '\PaDEL-Descriptor.jar'])
#print(_PADEL_PATH)

#print (os.getenv("JAVA_HOME"))

fp_filename= "/models/fingerprints_xml.zip"
fp_target_name= "/models/fingerprints_xml_files"
archive_format = "zip"

absolute_path = os.path.dirname(__file__)
print(absolute_path)
fp_path = "models/fingerprints_xml"
fp_full_path = os.path.join(absolute_path, fp_path)
print(fp_full_path)


#print(os.path.dirname(os.path.abspath("molecule.smi"))) 
#print(os.path.abspath("molecule.smi"))
#print(os.path.abspath("CDK.csv"))


# Session state
if 'smiles_input' not in st.session_state:
  st.session_state.smiles_input = ''

# Utilities
if os.path.isfile('molecule.smi'):
  os.remove('molecule.smi') 
  

# The App    
st.title('📊 🎯 ACBR_Aankalan')
st.info('ACBR_Aankalan allow users to predict whether a query molecule is active/inactive towards the  target protein.')

tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs(['Classification', 'Regression', 'Virtual Screening','About ACBR_Aankalan?', 'Dataset', 'Model performance',  'Contact Us'])

#xml_files, Finp_list, fp, fingerprint = initialize_variables()


######### classification starts ###################
with tab1:
  st.header("Classification Module")
  option = st.selectbox(
   "Select the target",
   ("BCR-ABL", "HDAC6", "PARP4", "TELOMERASE"),
   index=None,
   placeholder="Select the target for classification", key= 'cls'
   )

  st.write('You selected:', option)
  ####### BCR-ABL classification starts ########

  if st.session_state.smiles_input == '' and option =="BCR-ABL":
    
    with st.form('my_form1'):
      st.subheader('Check the probability of compunds being active against the selected Target')

      smiles_txt = st.text_input('Enter SMILES notation', st.session_state.smiles_input)
      st.session_state.smiles_input = smiles_txt

      with st.expander('Example SMILES'):
        st.code('Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1')
      
      submit_button = st.form_submit_button('Submit')

      
      
      if submit_button:
        st.subheader('⚛️ Input molecule:')
        with st.expander('Show SMILES', expanded=True):
          #st.write('**SMILES**')
          st.text(st.session_state.smiles_input)

        with st.expander('Show chemical structures', expanded=True):
          #st.write('**Chemical structure**')
          smi = Chem.MolFromSmiles(st.session_state.smiles_input)
          Chem.Draw.MolToFile(smi, 'molecule.png', width=900)
          mol_image = Image.open('molecule.png')
          st.image(mol_image)

      # Input SMILES saved to file ############
      f = open('molecule.smi', 'w')
      f.write(f'{st.session_state.smiles_input}\tmol_001')
      f.close()

      
      # Compute PADEL descriptors
      if st.session_state.smiles_input != '':
        st.subheader('📝 Calculated Descriptors')
        if os.path.isfile('molecule.smi'):
          padeldescriptor(mol_dir='molecule.smi', 
                            d_file='descriptors.csv',
                            descriptortypes='models\\fingerprints_xml\\Fingerprinter.xml',
                            detectaromaticity=True,
                            standardizenitro=True,
                            standardizetautomers=True,
                            threads=2,
                            removesalt=True,
                            log=True,
                            fingerprints=True)

        descriptors = pd.read_csv('descriptors.csv')
        descriptors.drop('Name', axis=1, inplace=True)

        with st.expander('Show full set of descriptors as calculated for query molecule'):
          #st.write('**Full set of descriptors (calculated for query molecule)**')
          st.write(descriptors)
          st.write(descriptors.shape)


      # Load descriptor subset used in trained model
      if st.session_state.smiles_input != '':
        #model = pickle.load(open(r'C:\\Users\\user\\Desktop\\bcr_abl_pred\\myenv\\models\\rf_bcr_abl.bin', 'rb'))
        model = joblib.load('models/bcr_abl_rf_gini4_maxdepth4_n_est100_1.joblib')
        low_var_feat_names = pd.read_csv('models/cdk_low_var.csv')
        col= list(low_var_feat_names.columns.values)
        #cdk_low_var = descriptors.drop('Name', axis=1)

        cdk_low_var = descriptors.filter(col)
        preds = model.predict_proba(cdk_low_var)[0, 1]
        pred_round= round(preds, 2)

        with st.expander('Show subset of descriptors as used in trained model'):
          #st.write('**Subset of descriptors (used in trained model)**')
          st.write(cdk_low_var)
          st.write(cdk_low_var.shape)


      # Read saved  in classification model
      if st.session_state.smiles_input != '':
        st.subheader('🚀 Probability of the compound being active or inactive against the selected target')
        preds = model.predict_proba(cdk_low_var)[0, 1]
        pred_round= round(preds, 2)
        if pred_round < 0.5:
          st.error('Inactive ❌')
          st.write(pred_round)
        if pred_round >= 0.5:
          st.success('Active ✔️')
          st.write(pred_round)

          #### HDAC6 classification starts ##############

  elif st.session_state.smiles_input == '' and option =="HDAC6":
    
    with st.form('my_form1'):
      st.subheader('Check the probability of compunds being active against the selected Target')

      smiles_txt = st.text_input('Enter SMILES notation', st.session_state.smiles_input)
      st.session_state.smiles_input = smiles_txt

      with st.expander('Example SMILES'):
        st.code('Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1')
      
      submit_button = st.form_submit_button('Submit')

      
      
      if submit_button:
        st.subheader('⚛️ Input molecule:')
        with st.expander('Show SMILES', expanded=True):
          #st.write('**SMILES**')
          st.text(st.session_state.smiles_input)

        with st.expander('Show chemical structures', expanded=True):
          #st.write('**Chemical structure**')
          smi = Chem.MolFromSmiles(st.session_state.smiles_input)
          Chem.Draw.MolToFile(smi, 'molecule.png', width=900)
          mol_image = Image.open('molecule.png')
          st.image(mol_image)

      # Input SMILES saved to file
      f = open('molecule.smi', 'w')
      f.write(f'{st.session_state.smiles_input}\tmol_001')
      f.close()

      
      # Compute PADEL descriptors
      if st.session_state.smiles_input != '':
        st.subheader('📝 Calculated Descriptors')
        if os.path.isfile('molecule.smi'):
          
          padeldescriptor(mol_dir='molecule.smi', 
                            d_file='descriptors.csv',
                            descriptortypes='models/fingerprints_xml/Fingerprinter.xml',
                            detectaromaticity=True,
                            standardizenitro=True,
                            standardizetautomers=True,
                            threads=2,
                            removesalt=True,
                            log=True,
                            fingerprints=True)

        descriptors = pd.read_csv('descriptors.csv')
        descriptors.drop('Name', axis=1, inplace=True)

        with st.expander('Show full set of descriptors as calculated for query molecule'):
          #st.write('**Full set of descriptors (calculated for query molecule)**')
          st.write(descriptors)
          st.write(descriptors.shape)


      # Load descriptor subset used in trained model
      if st.session_state.smiles_input != '':
        #model = pickle.load(open(r'C:\\Users\\user\\Desktop\\bcr_abl_pred\\myenv\\models\\rf_bcr_abl.bin', 'rb'))
        model = joblib.load('models/bcr_abl_rf_gini4_maxdepth4_n_est100_1.joblib')
        low_var_feat_names = pd.read_csv('models\/cdk_low_var.csv')
        col= list(low_var_feat_names.columns.values)
        #cdk_low_var = descriptors.drop('Name', axis=1)

        cdk_low_var = descriptors.filter(col)
        preds = model.predict_proba(cdk_low_var)[0, 1]
        pred_round= round(preds, 2)
 
        with st.expander('Show subset of descriptors as used in trained model'):
          #st.write('**Subset of descriptors (used in trained model)**')
          st.write(cdk_low_var)
          st.write(cdk_low_var.shape)


      # Read in saved classification model
      if st.session_state.smiles_input != '':
        st.subheader('🚀 Probability of the compound being active or inactive against the selected target')
        preds = model.predict_proba(cdk_low_var)[0, 1]
        pred_round= round(preds, 2)


        #pred = int(model.predict(query_desc_2))
        if pred_round < 0.5:
          st.error('Inactive ❌')
          st.write(pred_round)
        if pred_round >= 0.5:
          st.success('Active ✔️')
          st.write(pred_round)


###### PARP4 classification starts  #############
          

  elif st.session_state.smiles_input == '' and option =="PARP4":
    
    with st.form('my_form1'):
      st.subheader('Check the probability of compunds being active against the selected Target')

      smiles_txt = st.text_input('Enter SMILES notation', st.session_state.smiles_input)
      st.session_state.smiles_input = smiles_txt

      with st.expander('Example SMILES'):
        st.code('Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1')
      
      submit_button = st.form_submit_button('Submit')

      
      
      if submit_button:
        st.subheader('⚛️ Input molecule:')
        with st.expander('Show SMILES', expanded=True):
          #st.write('**SMILES**')
          st.text(st.session_state.smiles_input)

        with st.expander('Show chemical structures', expanded=True):
          #st.write('**Chemical structure**')
          smi = Chem.MolFromSmiles(st.session_state.smiles_input)
          Chem.Draw.MolToFile(smi, 'molecule.png', width=900)
          mol_image = Image.open('molecule.png')
          st.image(mol_image)

      # Input SMILES saved to file
      f = open('molecule.smi', 'w')
      f.write(f'{st.session_state.smiles_input}\tmol_001')
      f.close()

      
      # Compute PADEL descriptors
      if st.session_state.smiles_input != '':
        st.subheader('📝 Calculated Descriptors')
        if os.path.isfile('molecule.smi'):
          
          padeldescriptor(mol_dir='molecule.smi', 
                            d_file='Cdescriptors.csv',
                            descriptortypes='models/fingerprints_xml/Fingerprinter.xml',
                            detectaromaticity=True,
                            standardizenitro=True,
                            standardizetautomers=True,
                            threads=2,
                            removesalt=True,
                            log=True,
                            fingerprints=True)

        descriptors = pd.read_csv('descriptors.csv')
        descriptors.drop('Name', axis=1, inplace=True)

        with st.expander('Show full set of descriptors as calculated for query molecule'):
          #st.write('**Full set of descriptors (calculated for query molecule)**')
          st.write(descriptors)
          st.write(descriptors.shape)


      # Load descriptor subset used in trained model
      if st.session_state.smiles_input != '':
        #model = pickle.load(open(r'C:\\Users\\user\\Desktop\\bcr_abl_pred\\myenv\\models\\rf_bcr_abl.bin', 'rb'))
        model = joblib.load('models/bcr_abl_rf_gini4_maxdepth4_n_est100_1.joblib')
        low_var_feat_names = pd.read_csv('models/cdk_low_var.csv')
        col= list(low_var_feat_names.columns.values)
        #cdk_low_var = descriptors.drop('Name', axis=1)

        cdk_low_var = descriptors.filter(col)
        preds = model.predict_proba(cdk_low_var)[0, 1]
        pred_round= round(preds, 2)


        with st.expander('Show subset of descriptors as used in trained model'):
          #st.write('**Subset of descriptors (used in trained model)**')
          st.write(cdk_low_var)
          st.write(cdk_low_var.shape)


      # Read in saved classification model
      if st.session_state.smiles_input != '':
        st.subheader('🚀 Probability of the compound being active or inactive against the selected target')
        preds = model.predict_proba(cdk_low_var)[0, 1]
        pred_round= round(preds, 2)


        #pred = int(model.predict(query_desc_2))
        if pred_round < 0.5:
          st.error('Inactive ❌')
          st.write(pred_round)
        if pred_round >= 0.5:
          st.success('Active ✔️')
          st.write(pred_round)

####### Telomarase classification starts ############
          
  elif st.session_state.smiles_input == '' and option =="TELOMERASE":
    
    with st.form('my_form1'):
      st.subheader('Check the probability of compunds being active against the selected Target')

      smiles_txt = st.text_input('Enter SMILES notation', st.session_state.smiles_input)
      st.session_state.smiles_input = smiles_txt

      with st.expander('Example SMILES'):
        st.code('Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1')
      
      submit_button = st.form_submit_button('Submit')

      
      
      if submit_button:
        st.subheader('⚛️ Input molecule:')
        with st.expander('Show SMILES', expanded=True):
          #st.write('**SMILES**')
          st.text(st.session_state.smiles_input)

        with st.expander('Show chemical structures', expanded=True):
          #st.write('**Chemical structure**')
          smi = Chem.MolFromSmiles(st.session_state.smiles_input)
          Chem.Draw.MolToFile(smi, 'molecule.png', width=900)
          mol_image = Image.open('molecule.png')
          st.image(mol_image)

      # Input SMILES saved to file
      f = open('molecule.smi', 'w')
      f.write(f'{st.session_state.smiles_input}\tmol_001')
      f.close()

      
      # Compute PADEL descriptors
      if st.session_state.smiles_input != '':
        st.subheader('📝 Calculated Descriptors')
        if os.path.isfile('molecule.smi'):
          
          padeldescriptor(mol_dir='molecule.smi', 
                            d_file='descriptors.csv',
                            descriptortypes='models/fingerprints_xml/Fingerprinter.xml',
                            detectaromaticity=True,
                            standardizenitro=True,
                            standardizetautomers=True,
                            threads=2,
                            removesalt=True,
                            log=True,
                            fingerprints=True)

        descriptors = pd.read_csv('descriptors.csv')
        descriptors.drop('Name', axis=1, inplace=True)

        with st.expander('Show full set of descriptors as calculated for query molecule'):
          #st.write('**Full set of descriptors (calculated for query molecule)**')
          st.write(descriptors)
          st.write(descriptors.shape)


      # Load descriptor subset used in trained model
      if st.session_state.smiles_input != '':
        #model = pickle.load(open(r'C:\\Users\\user\\Desktop\\bcr_abl_pred\\myenv\\models\\rf_bcr_abl.bin', 'rb'))
        model = joblib.load('models/bcr_abl_rf_gini4_maxdepth4_n_est100_1.joblib')
        low_var_feat_names = pd.read_csv('models/cdk_low_var.csv')
        col= list(low_var_feat_names.columns.values)
        #cdk_low_var = descriptors.drop('Name', axis=1)

        cdk_low_var = descriptors.filter(col)
        preds = model.predict_proba(cdk_low_var)[0, 1]
        pred_round= round(preds, 2)
 
        with st.expander('Show subset of descriptors as used in trained model'):
          #st.write('**Subset of descriptors (used in trained model)**')
          st.write(cdk_low_var)
          st.write(cdk_low_var.shape)


      # Read in saved classification model
      if st.session_state.smiles_input != '':
        st.subheader('🚀 Probability of the compound being active or inactive against the selected target')
        preds = model.predict_proba(cdk_low_var)[0, 1]
        pred_round= round(preds, 2)


        #pred = int(model.predict(query_desc_2))
        if pred_round < 0.5:
          st.error('Inactive ❌')
          st.write(pred_round)
        if pred_round >= 0.5:
          st.success('Active ✔️')
          st.write(pred_round)


######### (Regression starts)########################
with tab2:
  st.header("Regression Module")
  option = st.selectbox(
   "Select the target",
   ("BCR-ABL", "HDAC6", "PARP4", "TELOMERASE"),
   index=None,
   placeholder="Select the target for Regression",key='reg'
   )

  st.write('You selected:', option)




  if st.session_state.smiles_input == '':
    
    with st.form('my_form2'):
      st.subheader('Check the pIC50 of the compound against the selected Target')

      smiles_txt = st.text_input('Enter SMILES notation', st.session_state.smiles_input)
      st.session_state.smiles_input = smiles_txt

      with st.expander('Example SMILES'):
        st.code('Clc1cccc(Nc2[nH]cnc3nnc(-c4ccccc4)c2-3)c1')
      
      submit_button = st.form_submit_button('Submit')

      
      
      if submit_button:
        st.subheader('⚛️ Input molecule:')
        with st.expander('Show SMILES', expanded=True):
          #st.write('**SMILES**')
          st.text(st.session_state.smiles_input)

        with st.expander('Show chemical structures', expanded=True):
          #st.write('**Chemical structure**')
          smi = Chem.MolFromSmiles(st.session_state.smiles_input)
          Chem.Draw.MolToFile(smi, 'molecule.png', width=900)
          mol_image = Image.open('molecule.png')
          st.image(mol_image)

      # Input SMILES saved to file
      f = open('molecule.smi', 'w')
      f.write(f'{st.session_state.smiles_input}\tmol_001')
      f.close()

      
      # Compute PADEL descriptors
      if st.session_state.smiles_input != '':
        st.subheader('📝 Calculated Descriptors')
        if os.path.isfile('molecule.smi'):
          
          padeldescriptor(mol_dir='molecule.smi', 
                            d_file='descriptors.csv',
                            descriptortypes='models/fingerprints_xml/Fingerprinter.xml',
                            detectaromaticity=True,
                            standardizenitro=True,
                            standardizetautomers=True,
                            threads=2,
                            removesalt=True,
                            log=True,
                            fingerprints=True)

        descriptors = pd.read_csv('descriptors.csv')
        descriptors.drop('Name', axis=1, inplace=True)

        with st.expander('Show full set of descriptors as calculated for query molecule'):
          #st.write('**Full set of descriptors (calculated for query molecule)**')
          st.write(descriptors)
          st.write(descriptors.shape)


      # Load descriptor subset used in trained model
      if st.session_state.smiles_input != '':
        #model = pickle.load(open(r'C:\\Users\\user\\Desktop\\bcr_abl_pred\\myenv\\models\\rf_bcr_abl.bin', 'rb'))
        model = joblib.load('models/bcr_abl_rf_gini4_maxdepth4_n_est100_1.joblib')
        low_var_feat_names = pd.read_csv('models/cdk_low_var.csv')
        col= list(low_var_feat_names.columns.values)
        #cdk_low_var = descriptors.drop('Name', axis=1)

        cdk_low_var = descriptors.filter(col)
        preds = model.predict_proba(cdk_low_var)[0, 1]
        pred_round= round(preds, 2)


        with st.expander('Show subset of descriptors as used in trained model'):
          #st.write('**Subset of descriptors (used in trained model)**')
          st.write(cdk_low_var)
          st.write(cdk_low_var.shape)


      # Read in saved classification model
      if st.session_state.smiles_input != '':
        st.subheader('🚀 pIC50 of the compound against the selected target')
        preds = model.predict_proba(cdk_low_var)[0, 1]
        pred_round= round(preds, 2)


        #pred = int(model.predict(query_desc_2))
        if pred_round < 0.5:
          st.error('Inactive ❌')
          st.write(pred_round)
        if pred_round >= 0.5:
          st.success('Active ✔️')
          st.write(pred_round)

########Regression ends ######################################
####### virtual screening starts     ##########################################
with tab3:
  st.header("Virtual Screening Module")
  with st.form(key='multi-structure module'):
        uploaded_file = st.file_uploader('Upload CSV file with IDs (first column) and SMILES (second column)', type='csv')
        submit_button_vs = st.form_submit_button(label='Submit')
        st.caption('Results of the virtual screening are written to the file "results_vs_*.csv".')
  if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    #df.dropna(inplace=True)
    #df.reset_index(drop = True, inplace=True)
    st.write("Displaying CSV file contents:")
    st.write(df)
    #df.to_csv(r'C:\\Users\\user\\Desktop\\bcr_abl_pred\\myenv\\models\\mol_vs.smi', sep='\t', index=False, header=False)
    df.to_csv('mol_vs.smi', sep='\t', index=False, header=False)
    if os.path.isfile('mol_vs.smi'):
          padeldescriptor(mol_dir='mol_vs.smi', 
                            d_file='desc_vs.csv',
                            descriptortypes='models/fingerprints_xml/Fingerprinter.xml',
                            detectaromaticity=True,
                            standardizenitro=True,
                            standardizetautomers=True,
                            threads=2,
                            removesalt=True,
                            log=True,
                            fingerprints=True)
    
    desc_vs1 = pd.read_csv('desc_vs.csv')
    #desc_vs1.drop('Name', axis=1, inplace=True)


    if uploaded_file is not None:
        #model = pickle.load(open(r'C:\\Users\\user\\Desktop\\bcr_abl_pred\\myenv\\models\\rf_bcr_abl.bin', 'rb'))
        model = joblib.load('models/bcr_abl_rf_gini4_maxdepth4_n_est100_1.joblib')
        low_var_feat_names = pd.read_csv('models/cdk_low_var.csv')
        col= list(low_var_feat_names.columns.values)
        #cdk_low_var = descriptors.drop('Name', axis=1)

        df_cdk_low_var_vs = desc_vs1.filter(col)
        st.write(df_cdk_low_var_vs)
        vs_prob = model.predict_proba(df_cdk_low_var_vs)[:, 1]
        vs_result_df = pd.DataFrame({'Probabilities': vs_prob})
        vs_result_df['Predictions'] = ['Inactive ❌' if prob < 0.5 else 'Active ✔️' for prob in vs_result_df['Probabilities']]
        st.write(vs_result_df)


################################# virtual screening ends ###########################

        
#tab4
with tab4:
  st.header('What is ACBR_Aankalan')
  st.write('Presenting Acbr_aankalan, a cutting-edge machine learning tool powered by artificial intelligence created for cancer research prediction analysis. Acbr_aankalan provides a comprehensive platform for the identification and assessment of small molecules with possible inhibitory effects on a range of cancer targets by utilising sophisticated algorithms. This platform gives researchers the capacity to accurately and efficiently anticipate the inhibitory activity of drugs against various cancer targets by integrating state-of-the-art machine learning algorithms.')

#####

#####
with tab5:
  st.header('Dataset')
  st.write('Coming Soon')

######
with tab6:
  
  st.markdown('''Model Performance
    
  ''')
with tab7:
  st.header('Project Coordinator- Prof. Madhu Chopra')
  st.markdown('Coming Soon')

############################## coding ends #################################