import streamlit as st
import os
import pickle
import joblib
import pandas as pd
import sklearn
import numpy as np
import glob
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay, accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report,confusion_matrix,auc, matthews_corrcoef
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier as xgb
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.decomposition import PCA
from lightgbm import LGBMClassifier as lgb

from PIL import Image
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import Descriptors, Draw, PandasTools
from rdkit.Chem import MolFromSmiles
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem import QED
from rdkit.Chem import Lipinski
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
#from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import MolFromSmiles
from rdkit.DataStructs import TanimotoSimilarity


from padelpy import padeldescriptor
import pathlib 
from pathlib import Path
from shutil import which
from os.path import abspath, dirname, join
import subprocess
import base64
import time
import matplotlib
import matplotlib.pyplot as plt

#page_configuration

st.set_page_config(
  #page_title='CancerAI',
  page_title='Acbr_Aankalan',
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
st.info('ACBR_Aankalan allow users to predict whether a query molecule is active/inactive towards the selected target cancer protein.')

#st.title('📊 🎯 💊 CancerAI 💻')
#st.info('CancerAI allow users to predict whether a query molecule/s is/are active/inactive towards the  selected target cancer protein.')


tab1,tab2,tab3,tab4,tab5,tab6, tab7 = st.tabs(['Classification', 'Virtual Screening', 'Help', 'About ACBR_Aankalan', 'Datasets', 'Model details',  'Contact Us'])
#tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs(['Classification', 'Regression', 'Virtual Screening','About ACBR_Aankalan?', 'Dataset', 'Model performance',  'Contact Us'])


######### classification starts ###################
with tab1:
  st.header("Classification Module")
  
  #st.markdown('<h1 style="color:blue; font-weight:bold;">Classification Module</h1>', unsafe_allow_html=True)
  option = st.selectbox(
   "Select the target",
   ("BCR-ABL", "HDAC6", "PARP1", "TELOMERASE"),
   index=None,
   placeholder="Select the target for classification", key= 'cls'
   )

  st.write('You selected:', option)

  #if st.session_state.smiles_input != '':
  if st.session_state.smiles_input == '':
    
    with st.form('my_form1'):
      st.subheader('Check the probability of compounds being active against the selected Target')

      smiles_txt = st.text_input('Enter SMILES notation', st.session_state.smiles_input)
      st.session_state.smiles_input = smiles_txt

      


      with st.expander('Example SMILES'):
        st.code('Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1')
      #start_time = time.time()
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
          mol_wt = round(Descriptors.MolWt(smi),2)
          logp = round(Descriptors.MolLogP(smi), 2)
          num_heavy_atoms = Descriptors.HeavyAtomCount(smi)
          hba= rdkit.Chem.Lipinski.NumHAcceptors(smi)
          qed_prop= rdkit.Chem.QED.properties(smi)
          aromat= rdkit.Chem.Lipinski.NumAromaticRings(smi)
          hbd= rdkit.Chem.Lipinski.NumHDonors(smi)
          rotat= rdkit.Chem.Lipinski.NumRotatableBonds(smi)
          tpsa= round(Descriptors.TPSA(smi), 2)
          mol_image = Image.open('molecule.png')
          st.image(mol_image)
          
      #start_time = time.time()
      # Input SMILES saved to file ############
      f = open('molecule.smi', 'w')
      start_time = time.time()
      f.write(f'{st.session_state.smiles_input}\tmol_001')
      f.close()

      
      # Compute PADEL descriptors
      if st.session_state.smiles_input != '':
        st.subheader('📝 Calculated Descriptors')
        if os.path.isfile('molecule.smi'):
          padeldescriptor(mol_dir='molecule.smi', 
                            
                            #descriptortypes=f'C:\\Users\\user\\Desktop\\bcr_abl_pred\\myenv\\models\\classification\\{option}\\{option}Fingerprinter.xml',
                            d_file='./descriptors.csv',
                            descriptortypes= f'./models/classification/{option}/{option}Fingerprinter.xml',

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
        #################### BCR-ABL (LGB CDK) ###############################
        if option == "BCR-ABL":
          #model = joblib.load(f'C:\\Users\\user\\Desktop\\bcr_abl_pred\\myenv\\models\\classification\\{option}\\{option}.joblib')
          #pipeline= joblib.load(f'C:\\Users\\user\\Desktop\\bcr_abl_pred\\myenv\\models\\classification\\{option}\\{option}.pkl')

          model = joblib.load(f'./models/classification/{option}/{option}.joblib')
          pipeline = joblib.load(f'./models/classification/{option}/{option}.pkl')
          X_test = pipeline.named_steps['variance_threshold'].transform(descriptors)
          preds = model.predict_proba(X_test)[0, 1]
          #pred_round= round(preds, 2)

          

          with st.expander('Show subset of descriptors as used in trained model'):
          #st.write('**Subset of descriptors (used in trained model)**')
           st.write(X_test)
           st.write(X_test.shape[1])

          if st.session_state.smiles_input != '':
            st.subheader('🚀 Probability of the compound being active or inactive against the selected target')
            pred_round= round(preds, 2)
            pred_round_formatted = f"{pred_round:.2f}"
        
          if pred_round < 0.5:
            st.error('Inactive ❌')
            st.write(f'The Probability of this compound against {option} is {pred_round_formatted}')
          if pred_round >= 0.5:
            st.success('Active ✔️')
            st.write(f'The Probability of this compound against {option} is {pred_round_formatted}')
            
            #bcr_abl_tanimoto_main_file_path = 'C:\\Users\\user\\Desktop\\bcr_abl_pred\\myenv\\models\\classification\\bcr_abl_tanimoto_list.csv'
            bcr_abl_tanimoto_main_file_path = './models/classification/bcr_abl_tanimoto_list.csv'

            if os.path.exists(bcr_abl_tanimoto_main_file_path):
              bcr_abl_tanimoto_main_df = pd.read_csv(bcr_abl_tanimoto_main_file_path) 
              query_file_path = 'molecule.smi'

              with open(query_file_path, 'r') as f:
                query_smiles, query_name = f.read().strip().split()

              def get_morgan_fingerprint(smiles, radius=2, n_bits=2048):
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                  return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                else:
                  return None
                
              
              query_fp = get_morgan_fingerprint(query_smiles)
              if query_fp:
                similarities = []

                for index, row in bcr_abl_tanimoto_main_df.iterrows():
                  smiles = row['Smiles']
                  chembl_id = row['ChEMBL_ID_Molecule_Name']
                  target_fp = get_morgan_fingerprint(smiles)

                  if target_fp:
                    similarity = TanimotoSimilarity(query_fp, target_fp)
                    similarities.append({
                    'ChEMBL_ID_Molecule_Name': chembl_id,
                    #'Smiles': smiles,
                    'Similarity': similarity
                     })

                similarity_df = pd.DataFrame(similarities)
                similarity_df['Similarity'] = similarity_df['Similarity'].round(2)
                similarity_df = similarity_df.sort_values(by='Similarity', ascending=False)

                top_row = similarity_df.iloc[0]

                # Display results
                #st.write("Details of the actual drug with the highest Tanimoto similarity with the predicted active query compound:")
                st.markdown('<h3 style="font-weight:bold;">Top <span style="color:blue;"> BCR-ABL </span> drug by Tanimoto similarity to the predicted active compound:</h3>',unsafe_allow_html=True)
                st.write(top_row)
                #st.dataframe(similarity_df)
                #st.dataframe(similarity_df.iloc[0])
              else:
                st.error("Invalid query SMILES. Please enter a valid SMILES string.")

################### BCR-ABL calculation ends here ##############################

############################## HDAC6 (LGB CDK) STARTS HERE ##################################

        elif option == "HDAC6":

          model = joblib.load(f'./models/classification/{option}/{option}.joblib')
          pipeline = joblib.load(f'./models/classification/{option}/{option}.pkl')
          X_test_hdac_lgb = pipeline.named_steps['variance_threshold'].transform(descriptors)
          X_test_reduced_hdac_lgb = pipeline.named_steps['rfe'].transform(X_test_hdac_lgb)
          preds = model.predict_proba(X_test_reduced_hdac_lgb)[0, 1]
          #pred_round= round(preds, 2)
          
          with st.expander('Show subset of descriptors as used in trained model'):
          #st.write('**Subset of descriptors (used in trained model)**')
           st.write(X_test_reduced_hdac_lgb)
           st.write(X_test_reduced_hdac_lgb.shape[1])


          if st.session_state.smiles_input != '':
            st.subheader('🚀 Probability of the compound being active or inactive against the selected target')
            pred_round= round(preds, 2)
            pred_round_formatted = f"{pred_round:.2f}"
        
          if pred_round < 0.5:
            st.error('Inactive ❌')
            st.write(f'The Probability of this compound against {option} is {pred_round_formatted}')
          if pred_round >= 0.5:
            st.success('Active ✔️')
            st.write(f'The Probability of this compound against {option} is {pred_round_formatted}')

            hdac6_tanimoto_main_file_path = './models/classification/hdac6_tanimoto_list.csv'
            if os.path.exists(hdac6_tanimoto_main_file_path):

              hdac6_tanimoto_main_df = pd.read_csv(hdac6_tanimoto_main_file_path) 
              query_file_path = 'molecule.smi'

              with open(query_file_path, 'r') as f:
                query_smiles, query_name = f.read().strip().split()

              # Fingerprint calculation using Morgan fingerprint
              def get_morgan_fingerprint(smiles, radius=2, n_bits=2048):
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                  return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                else:
                  return None
                
              
              query_fp = get_morgan_fingerprint(query_smiles)
              if query_fp:
                similarities = []

                for index, row in hdac6_tanimoto_main_df.iterrows():
                  smiles = row['Smiles']
                  chembl_id = row['ChEMBL_ID_Molecule_Name']
                  target_fp = get_morgan_fingerprint(smiles)

                  if target_fp:
                    similarity = TanimotoSimilarity(query_fp, target_fp)
                    similarities.append({
                    'ChEMBL_ID_Molecule_Name': chembl_id,
                    #'Smiles': smiles,
                    'Similarity': similarity
                     })

                similarity_df = pd.DataFrame(similarities)
                similarity_df['Similarity'] = similarity_df['Similarity'].round(2)
                similarity_df = similarity_df.sort_values(by='Similarity', ascending=False)

                top_row = similarity_df.iloc[0]
                #st.write("Details of the actual HDAC6 drug with the highest Tanimoto similarity for the query compound:")
                #st.write(top_row)
               

                # Display results
                st.markdown('<h3 style="font-weight:bold;">Top <span style="color:blue;">HDAC 6</span> drug by Tanimoto similarity to the predicted active compound:</h3>',unsafe_allow_html=True)
                #st.write("Details of the actual HDAC6 drug with the highest Tanimoto similarity with the predicted active query compound:")
                st.write(top_row)
                #st.dataframe(similarity_df)
                #st.dataframe(similarity_df.iloc[0])
              else:
                st.error("Invalid query SMILES. Please enter a valid SMILES string.")
   
             ############################## HDAC6 (LGB CDK) ENDS HERE ##################################

             ############################## PARP1 ( XGB PUBCHEM) STARTS HERE ##################################

        elif option == "PARP1":
          #model = joblib.load(f'C:\\Users\\user\\Desktop\\bcr_abl_pred\\myenv\\models\\classification\\{option}\\{option}.joblib')
          #pipeline= joblib.load(f'C:\\Users\\user\\Desktop\\bcr_abl_pred\\myenv\\models\\classification\\{option}\\{option}.pkl')

          model = joblib.load(f'./models/classification/{option}/{option}.joblib')
          pipeline = joblib.load(f'./models/classification/{option}/{option}.pkl')
          X_test_parp_xgb = pipeline.named_steps['variance_threshold'].transform(descriptors)
          X_test_reduced_parp_xgb = pipeline.named_steps['rfe'].transform(X_test_parp_xgb)
          preds = model.predict_proba(X_test_reduced_parp_xgb)[0, 1]
          #pred_round= round(preds, 2)
          
          with st.expander('Show subset of descriptors as used in trained model'):
          #st.write('**Subset of descriptors (used in trained model)**')
           st.write(X_test_reduced_parp_xgb)
           st.write(X_test_reduced_parp_xgb.shape[1])


          if st.session_state.smiles_input != '':
            st.subheader('🚀 Probability of the compound being active or inactive against the selected target')
            pred_round= round(preds, 2)
            pred_round_formatted = f"{pred_round:.2f}"
        
          if pred_round < 0.5:
            st.error('Inactive ❌')
            st.write(f'The Probability of this compound against {option} is {pred_round_formatted}')
          if pred_round >= 0.5:
            st.success('Active ✔️')
            st.write(f'The Probability of this compound against {option} is {pred_round_formatted}')

            parp1_tanimoto_main_file_path = './models/classification/parp1_tanimoto_list.csv'
            
            if os.path.exists(parp1_tanimoto_main_file_path):
             
              parp1_tanimoto_main_df = pd.read_csv(parp1_tanimoto_main_file_path) 
              query_file_path = 'molecule.smi'

              with open(query_file_path, 'r') as f:
                query_smiles, query_name = f.read().strip().split()

              # Fingerprint calculation using Morgan fingerprint
              def get_morgan_fingerprint(smiles, radius=2, n_bits=2048):
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                  return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                else:
                  return None
                
              
              query_fp = get_morgan_fingerprint(query_smiles)
              if query_fp:
                similarities = []

                for index, row in parp1_tanimoto_main_df.iterrows():
                  smiles = row['Smiles']
                  chembl_id = row['ChEMBL_ID_Molecule_Name']
                  target_fp = get_morgan_fingerprint(smiles)

                  if target_fp:
                    similarity = TanimotoSimilarity(query_fp, target_fp)
                    similarities.append({
                    'ChEMBL_ID_Molecule_Name': chembl_id,
                    #'Smiles': smiles,
                    'Similarity': similarity
                     })

                similarity_df = pd.DataFrame(similarities)
                similarity_df['Similarity'] = similarity_df['Similarity'].round(2)
                similarity_df = similarity_df.sort_values(by='Similarity', ascending=False)

                top_row = similarity_df.iloc[0]
                #st.write("Details of the actual drug with the highest Tanimoto similarity for the query compound:")
                #st.write(top_row)
               

                # Display results
                st.markdown('<h3 style="font-weight:bold;">Top <span style="color:blue;">PARP-1</span> drug by Tanimoto similarity to the predicted active compound:</h3>',unsafe_allow_html=True)
                st.write(top_row)
                #st.dataframe(similarity_df)
                #st.dataframe(similarity_df.iloc[0])
              else:
                st.error("Invalid query SMILES. Please enter a valid SMILES string.")
   

          
           ################### TELOMERASE (SVC KLEKOTAROTH) STARTS HERE  ###############
        elif option == "TELOMERASE":
          #model = joblib.load(f'C:\\Users\\user\\Desktop\\bcr_abl_pred\\myenv\\models\\classification\\{option}\\{option}.joblib')
          #pipeline= joblib.load(f'C:\\Users\\user\\Desktop\\bcr_abl_pred\\myenv\\models\\classification\\{option}\\{option}.pkl')

          model = joblib.load(f'./models/classification/{option}/{option}.joblib')
          pipeline = joblib.load(f'./models/classification/{option}/{option}.pkl')
          X_test_telo_svc = pipeline.named_steps['variance_threshold'].transform(descriptors)
          X_test_reduced_telo_svc = pipeline.named_steps['rfe'].transform(X_test_telo_svc)
          preds = model.predict_proba(X_test_reduced_telo_svc)[0, 1]
          #pred_round= round(preds, 2)
          
          with st.expander('Show subset of descriptors as used in trained model'):
          #st.write('**Subset of descriptors (used in trained model)**')
           st.write(X_test_reduced_telo_svc)
           st.write(X_test_reduced_telo_svc.shape[1])
           ################### TELOMERASE (SVC KLEKOTAROTH) ENDS HERE  ###############

          if st.session_state.smiles_input != '':
            st.subheader('🚀 Probability of the compound being active or inactive against the selected target')
            pred_round= round(preds, 2)
            pred_round_formatted = f"{pred_round:.2f}"
        
          if pred_round < 0.5:
            st.error('Inactive ❌')
            st.write(f'The Probability of this compound against {option} is {pred_round_formatted}')
          if pred_round >= 0.5:
            st.success('Active ✔️')
            st.write(f'The Probability of this compound against {option} is {pred_round_formatted}')

            telom_tanimoto_main_file_path = './models/classification/telom_tanimoto_list.csv'
            
            if os.path.exists(telom_tanimoto_main_file_path):
             # Read the file assuming it's a CSV with columns 'chembl_id' and 'smiles'
              #bcr_abl_tanimoto_main_df = pd.read_csv(bcr_abl_tanimoto_main_file_path)
              #bcr_abl_tanimoto_compounds = bcr_abl_tanimoto_main_df.to_dict('records')

              telom_tanimoto_main_df = pd.read_csv(telom_tanimoto_main_file_path) 
              query_file_path = 'molecule.smi'

              with open(query_file_path, 'r') as f:
                query_smiles, query_name = f.read().strip().split()

              # Fingerprint calculation using Morgan fingerprint
              def get_morgan_fingerprint(smiles, radius=2, n_bits=2048):
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                  return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                else:
                  return None
                
              
              query_fp = get_morgan_fingerprint(query_smiles)
              if query_fp:
                similarities = []

                for index, row in telom_tanimoto_main_df.iterrows():
                  smiles = row['Smiles']
                  chembl_id = row['ChEMBL_ID_Molecule_Name']
                  target_fp = get_morgan_fingerprint(smiles)

                  if target_fp:
                    similarity = TanimotoSimilarity(query_fp, target_fp)
                    similarities.append({
                    'ChEMBL_ID_Molecule_Name': chembl_id,
                    #'Smiles': smiles,
                    'Similarity': similarity
                     })

                similarity_df = pd.DataFrame(similarities)
                similarity_df['Similarity'] = similarity_df['Similarity'].round(2)
                similarity_df = similarity_df.sort_values(by='Similarity', ascending=False)

                top_row = similarity_df.iloc[0]
                #st.write("Details of the actual drug with the highest Tanimoto similarity for the query compound:")
                #st.write(top_row)
               
                st.markdown('<h3 style="font-weight:bold;">Top <span style="color:blue;">Telomerase</span> drug by Tanimoto similarity to the predicted active compound:</h3>',unsafe_allow_html=True)

                st.write(top_row)
                #st.dataframe(similarity_df)
                #st.dataframe(similarity_df.iloc[0])
              else:
                st.error("Invalid query SMILES. Please enter a valid SMILES string.")



     

        classif_data = [
        {'Property': 'Molecular Weight', 'Value': f"{round(mol_wt, 2):.2f}"},
        {'Property': 'LogP', 'Value': f"{round(logp, 2):.2f}"},
        {'Property': 'Number of Heavy Atoms', 'Value': f"{round(num_heavy_atoms, 2):.2f}"},
        {'Property': 'Number of H-Bond Acceptors', 'Value': f"{round(hba, 2):.2f}"},
        {'Property': 'Number of H-Bond Donors', 'Value': f"{round(hbd, 2):.2f}"},
        {'Property': 'Number of Rotational Bonds', 'Value': f"{round(rotat, 2):.2f}"},
        {'Property': 'Topological polar surface area (TPSA)', 'Value': f"{round(tpsa, 2):.2f}"},
        {'Property': 'Aromaticity', 'Value': f"{round(aromat, 2):.2f}"}
        ]
      
        #st.header('Drug-like properties')
        st.markdown('<h3 style="color:blue; font-weight:bold;">Drug-like properties</h3>', unsafe_allow_html=True)
        st.table(pd.DataFrame(classif_data))
      end_time = time.time()
      duration = end_time - start_time
      st.write('Time Taken:', round(duration,2), 'secs')

        
######### (Classification ends)########################  

######### (Regression starts)########################


########Regression ends ######################################


####### virtual screening starts     ##########################################
with tab2:
  st.header("Virtual Screening Module")
  #st.markdown('<h1 style="color:blue; font-weight:bold;">Virtual Screening Module</h1>', unsafe_allow_html=True)
  option = st.selectbox(
   "Select the target",
   ("BCR-ABL", "HDAC6", "PARP1", "TELOMERASE"),
   index=None,
   placeholder="Select the target for Virtual Screening",key='vs'
   )

  st.write('You selected:', option)

  with st.form(key='multi-structure module'):
        uploaded_file = st.file_uploader('Upload CSV file with SMILES (1st column) and id (2nd column)', type='csv')
        submit_button_vs = st.form_submit_button(label='Submit')
        st.caption('Results of the virtual Screening (VS) are written to the file "results_vs_*.csv".')
  if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    #df.dropna(inplace=True)
    #df.reset_index(drop = True, inplace=True)
    #st.markdown('<p style="color:blue; font-weight:bold;">This website is developed by Dr. Mohd. Shoaib Khan, PhD.</p>', unsafe_allow_html=True)

    st.markdown(f'<h2 style="color:maroon;">Result of Virtual Screening against {option}</h2>', unsafe_allow_html=True)

    #st.header(f'Result of Virtual Screening against {option}')
    st.write("Displaying CSV file contents:")
    #st.write(df)
    #df.to_csv(r'C:\\Users\\user\\Desktop\\bcr_abl_pred\\myenv\\models\\mol_vs.smi', sep='\t', index=False, header=False)
    df.to_csv('mol_vs.smi', sep='\t', index=False, header=False)
    if os.path.isfile('mol_vs.smi'):
          padeldescriptor(mol_dir='mol_vs.smi', 
                            #d_file=r'C:\\Users\\user\\Desktop\\bcr_abl_pred\\myenv\\desc_vs.csv',
                            #descriptortypes=f'C:\\Users\\user\\Desktop\\bcr_abl_pred\\myenv\\models\\classification\\{option}\\{option}Fingerprinter.xml',

                            d_file='./desc_vs.csv',
                            descriptortypes= f'./models/classification/{option}/{option}Fingerprinter.xml',
                            
                            detectaromaticity=True,
                            standardizenitro=True,
                            standardizetautomers=True,
                            threads=2,
                            removesalt=True,
                            log=True,
                            fingerprints=True)
    
    desc_vs1 = pd.read_csv('desc_vs.csv')
    desc_vs1.drop('Name', axis=1, inplace=True)


    if uploaded_file is not None:
        if option == "BCR-ABL":
          #model = joblib.load(f'C:\\Users\\user\\Desktop\\bcr_abl_pred\\myenv\\models\\classification\\{option}\\{option}.joblib')
          #pipeline= joblib.load(f'C:\\Users\\user\\Desktop\\bcr_abl_pred\\myenv\\models\\classification\\{option}\\{option}.pkl')

          model = joblib.load(f'./models/classification/{option}/{option}.joblib')
          pipeline = joblib.load(f'./models/classification/{option}/{option}.pkl')
          X_test_vs_bcr_abl = pipeline.named_steps['variance_threshold'].transform(desc_vs1)
          vs_prob = model.predict_proba(X_test_vs_bcr_abl)[:, 1]
          #vs_prob = np.round(model.predict_proba(X_test_vs_bcr_abl)[:, 1], 2)
          vs_prob = np.array(vs_prob)
          vs_prob_rounded = np.round(vs_prob, 2)

          



        elif option == "HDAC6":
          #model = joblib.load(f'C:\\Users\\user\\Desktop\\bcr_abl_pred\\myenv\\models\\classification\\{option}\\{option}.joblib')
          #pipeline= joblib.load(f'C:\\Users\\user\\Desktop\\bcr_abl_pred\\myenv\\models\\classification\\{option}\\{option}.pkl')

          model = joblib.load(f'./models/classification/{option}/{option}.joblib')
          pipeline = joblib.load(f'./models/classification/{option}/{option}.pkl')
          X_test_vs_hdac_lgb = pipeline.named_steps['variance_threshold'].transform(desc_vs1)
          X_test_vs_reduced_hdac_lgb = pipeline.named_steps['rfe'].transform(X_test_vs_hdac_lgb)
          vs_prob = model.predict_proba(X_test_vs_reduced_hdac_lgb)[:, 1]
          vs_prob = np.array(vs_prob)
          vs_prob_rounded = np.round(vs_prob, 2)


        elif option == "PARP1":
          #model = joblib.load(f'C:\\Users\\user\\Desktop\\bcr_abl_pred\\myenv\\models\\classification\\{option}\\{option}.joblib')
          #pipeline= joblib.load(f'C:\\Users\\user\\Desktop\\bcr_abl_pred\\myenv\\models\\classification\\{option}\\{option}.pkl')

          model = joblib.load(f'./models/classification/{option}/{option}.joblib')
          pipeline = joblib.load(f'./models/classification/{option}/{option}.pkl')
          X_test_vs_parp_xgb = pipeline.named_steps['variance_threshold'].transform(desc_vs1)
          X_test_vs_reduced_parp_xgb = pipeline.named_steps['rfe'].transform(X_test_vs_parp_xgb)
          vs_prob = model.predict_proba(X_test_vs_reduced_parp_xgb)[:, 1]
          #vs_prob = np.round(model.predict_proba(X_test_vs_reduced_parp_xgb)[:, 1], 2)
          vs_prob = np.array(vs_prob)
          vs_prob_rounded = np.round(vs_prob, 2)

        elif option == "TELOMERASE":
          #model = joblib.load(f'C:\\Users\\user\\Desktop\\bcr_abl_pred\\myenv\\models\\classification\\{option}\\{option}.joblib')
          #pipeline= joblib.load(f'C:\\Users\\user\\Desktop\\bcr_abl_pred\\myenv\\models\\classification\\{option}\\{option}.pkl')

          model = joblib.load(f'./models/classification/{option}/{option}.joblib')
          pipeline = joblib.load(f'./models/classification/{option}/{option}.pkl')
          X_test_vs_telo_svc = pipeline.named_steps['variance_threshold'].transform(desc_vs1)
          X_test_vs_reduced_telo_svc = pipeline.named_steps['rfe'].transform(X_test_vs_telo_svc)
          vs_prob = model.predict_proba(X_test_vs_reduced_telo_svc)[:, 1]
          vs_prob = np.array(vs_prob)
          vs_prob_rounded = np.round(vs_prob, 2)

        

        # drug like properties calculation
        #ids= []
        mol_wts = []
        logps = []
        num_heavy_atoms_list = []
        hbas = []
        qed_props = []
        aromats = []
        hbds = []
        rotats = []
        tpsas = []  
        # Iterate over each row of the DataFrame
        for index, row in df.iterrows():
          #smi = Chem.MolFromSmiles(row['smiles'])
          smi = Chem.MolFromSmiles(row[0])
          mol_wt = round(Descriptors.MolWt(smi), 2)
          logp = round(Descriptors.MolLogP(smi), 2)
          num_heavy_atoms = Descriptors.HeavyAtomCount(smi)
          hba = rdkit.Chem.Lipinski.NumHAcceptors(smi)
          qed_prop = rdkit.Chem.QED.properties(smi)
          aromat = rdkit.Chem.Lipinski.NumAromaticRings(smi)
          hbd = rdkit.Chem.Lipinski.NumHDonors(smi)
          rotat = rdkit.Chem.Lipinski.NumRotatableBonds(smi)
          tpsa = round(Descriptors.TPSA(smi), 2)

          # Append calculated properties to respective lists
          #ids.append(row['id'])
          mol_wts.append(mol_wt)
          logps.append(logp)
          num_heavy_atoms_list.append(num_heavy_atoms)
          hbas.append(hba)
          qed_props.append(qed_prop)
          aromats.append(aromat)
          hbds.append(hbd)
          rotats.append(rotat)
          tpsas.append(tpsa)



        #vs_prob = model.predict_proba(X_test_vs)[:, 1]
        vs_result_df = pd.DataFrame({'Probabilities': vs_prob_rounded})
        vs_result_df['Predictions'] = ['Inactive ❌' if prob < 0.5 else 'Active ✔️' for prob in vs_result_df['Probabilities']]
        vs_result_df['Mol wt'] = mol_wts
        vs_result_df['LogP'] = logps
        vs_result_df['Heavy atoms'] = num_heavy_atoms_list
        vs_result_df['Aromaticity'] = aromats
        vs_result_df['HBA'] = hbas
        vs_result_df['HBD'] = hbds
        vs_result_df['TPSA'] = tpsas
        vs_result_df['Rotational bonds'] = rotats

        vs_result_df = pd.concat([df, vs_result_df], axis=1)

        #st.write(vs_result_df)
        rows_per_page = 10

        # Get the total number of pages
        total_pages = -(-len(vs_result_df) // rows_per_page)  # Ceiling division to get the total number of pages

        # Get the current page number from the query parameter
        page_number = st.number_input('Page Number', min_value=1, max_value=total_pages, value=1)

        

        # Calculate the start and end indices of the rows to display on the current page
        start_idx = (page_number - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, len(vs_result_df))

        # Create the 'Predictions' column based on 'Probabilities'
        #vs_result_df['Predictions'] = ['Inactive' if prob < 0.5 else 'Active' for prob in vs_result_df['Probabilities']]

        # Display the rows for the current page
        st.write("<style>div[data-testid='stHorizontalScroll'] div div { width: 1500px !important; }</style>", unsafe_allow_html=True)
        st.table(vs_result_df.iloc[start_idx:end_idx])

        # Display pagination controls
        if total_pages > 1:
          st.write(f'Page {page_number} of {total_pages}')

        #csv_data = vs_result_df.iloc[start_idx:end_idx].to_csv(index=False)
        #button_label = 'Download the VS result'
        #button_id = 'download_csv'
        #st.download_button(label=button_label, data=csv_data, file_name='vs_results.csv', mime='text/csv', key=button_id)

        csv_data = vs_result_df.to_csv(index=False, encoding='utf-8-sig')
        b64 = base64.b64encode(csv_data.encode()).decode()
        button_label = 'Download VS result in CSV format'
        button_id = 'download_csv'
        href = f'<a href="data:file/csv;base64,{b64}" download="vs_results.csv">{button_label}</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        #st.table(vs_result_df)


################################# virtual screening ends ###########################

        
#tab3

#####
with tab3:
  st.header('Help')
  st.write('Coming Soon')

with tab4:
  st.header('What is ACBR_Aankalan')
  st.write('Presenting Acbr_aankalan, a cutting-edge machine learning tool powered by artificial intelligence created for cancer research prediction analysis. Acbr_aankalan provides a comprehensive platform for the identification and assessment of small molecules with possible inhibitory effects on a range of cancer targets by utilising sophisticated algorithms. This platform gives researchers the capacity to accurately and efficiently anticipate the inhibitory activity of drugs against various cancer targets by integrating state-of-the-art machine learning algorithms.')
  #st.header('What is CancerAI')
  #st.write('Presenting CancerAI, a cutting-edge machine learning tool powered by artificial intelligence created for cancer research prediction analysis. CancerAI provides a comprehensive platform for the identification and assessment of small molecules with possible inhibitory effects on a range of cancer targets by utilising sophisticated algorithms. This platform gives researchers the capacity to accurately and efficiently anticipate the inhibitory activity of drugs against various cancer targets by integrating state-of-the-art machine learning algorithms.')
#####

#####
with tab5:
  st.header('Detail of the Datasets used to train/test the models against each target')
  data = {
    "Target": ["BCR-ABL", "HDAC6", "PARP1", "Telomerase"],
    "Raw dataset": [2225, 4212, 2426, 388],
    "Unique dataset": [1561, 3053, 2013, 281],
    "Active (pIC50 > 6.5, IC50 ≤300nM)": [886, 1781, 1338, 117],
    "Inactive (pIC50 < 6.3, IC50 ≥500nM)": [675, 1134, 600, 164]
   }

  # Convert to DataFrame
  df = pd.DataFrame(data)

  # Display the table using Streamlit
  st.table(df)


######
with tab6:
  
  st.header('Detail of Best Model against each target')

  data = {
    "Cancer Target": ["BCR-ABL", "HDAC6", "PAPR1", "Telomerase"],
    "Fingerprint": ["CDK", "CDK", "PubChem", "Klekotaroth"],
    "Algorithm": ["LightGBM",  "LightGBM", "XGBoost", "SVC"],
    "Pipeline Steps": [
        "VarianceThreshold, SMOTE",
        "VarianceThreshold, SMOTE, RFE",
        "VarianceThreshold, SMOTE, RFE",
        "VarianceThreshold, SMOTE, RFE",
    ],
   "No. of Features": [868, 50, 200, 50],

    "Performance Metrics": [
        "Accuracy, ROC-AUC, Precision, Recall",
        "Accuracy, ROC-AUC, F1 Score",
        "Accuracy, ROC-AUC, Precision, Recall",
        "Accuracy, ROC-AUC"
    ]
  }

# Convert to DataFrame
  df = pd.DataFrame(data)

# Display the table
  st.table(df)

              

with tab7:
  #st.header('This website is developed by Dr. Mohd. Shoaib Khan, PhD')
  #st.markdown('<h3 style="color:blue; font-weight:bold;">This website is developed by Dr. Mohd. Shoaib Khan, PhD.</h3>', unsafe_allow_html=True)

  #st.markdown('<p>If you have any comments or need more information, please contact me through:</p>', unsafe_allow_html=True)
  #st.markdown('<p><strong>Email:</strong> <a href="mailto:shoaib.khan155@gmail.com">shoaib.khan155@gmail.com</a></p>', unsafe_allow_html=True)
  #st.markdown('<p><strong>LinkedIn:</strong> <a href="https://www.linkedin.com/in/shoaib-khan-imt" target="_blank">www.linkedin.com/in/shoaib-khan-imt</a></p>', unsafe_allow_html=True)


  st.markdown('<h4 style="color:blue; font-weight:bold;">This website is maintained by Professor Madhu Chopra.</h4>', unsafe_allow_html=True)

  st.markdown('<p>If you have any comments or need more information, please contact me through:</p>', unsafe_allow_html=True)
  st.markdown('<p><strong>Email:</strong> <a href="mailto:mchopra@acbr.du.ac.in">mchopra@acbr.du.ac.in</a></p>', unsafe_allow_html=True)
  st.markdown('<p> Visit our Lab:</p>', unsafe_allow_html=True)
  st.markdown('<p> Lab No. 204 & 309, Anticancer drug development lab, ACBR, university of Delhi, North Campus, Delhi-110007</p>', unsafe_allow_html=True)
############################## coding ends #################################