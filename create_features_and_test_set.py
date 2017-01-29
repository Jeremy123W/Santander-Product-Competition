#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 22:09:06 2016

@author: jeremy
"""


import csv
import datetime
import numpy as np
import pandas as pd


mapping_dict = {
'ind_empleado'  : {-99:0, 'N':1, 'B':2, 'F':3, 'A':4, 'S':5},
'sexo'          : {'V':0, 'H':1, -99:2},
'ind_nuevo'     : {'0':0, '1':1, -99:2},
'indrel'        : {'1':0, '99':1, -99:2},
'indrel_1mes'   : {-99:0, '1.0':1, '1':1, '2.0':2, '2':2, '3.0':3, '3':3, '4.0':4, '4':4, 'P':5},
'tiprel_1mes'   : {-99:0, 'I':1, 'A':2, 'P':3, 'R':4, 'N':5},
'indresi'       : {-99:0, 'S':1, 'N':2},
'indext'        : {-99:0, 'S':1, 'N':2},
'conyuemp'      : {-99:0, 'S':1, 'N':2},
'indfall'       : {-99:0, 'S':1, 'N':2},
'tipodom'       : {-99:0, '1':1},
'nomprov'       : {'GIRONA':0, 'ZAMORA':1, 'BARCELONA':2, 'SALAMANCA':3, 'BURGOS':4, 'HUESCA':5, 'NAVARRA':6,'AVILA':7, 'SEGOVIA':8, 'LUGO':9, 'LERIDA':10, 'MADRID':11, 'ALICANTE':12, 'SORIA':13, 'SEVILLA':14,'CANTABRIA':15, 'BALEARS, ILLES':16, 'VALLADOLID':17, 'PONTEVEDRA':18, 'VALENCIA':19, 'TERUEL':20,'CORUÑA, A':21, 'OURENSE':22, 'JAEN':23, 'CUENCA':24, 'BIZKAIA':25, 'CASTELLON':26, 'RIOJA, LA':27,'ALBACETE':28, 'BADAJOZ':29, 'MURCIA':30, 'CADIZ':31, -99:32,  'ALMERIA':33, 'GUADALAJARA':34,'PALENCIA':35, 'PALMAS, LAS':36, 'CORDOBA':37, 'HUELVA':38, 'GRANADA':39, 'ASTURIAS':40,'SANTA CRUZ DE TENERIFE':41,'MELILLA':42, 'TARRAGONA':43, 'ALAVA':44, 'CEUTA':45, 'MALAGA':46, 'CIUDAD REAL':47, 'ZARAGOZA':48, 'TOLEDO':49, 'LEON':50, 'GIPUZKOA':51, 'CACERES':52},
'ind_actividad_cliente' : {'0':0, '1':1, -99:2},
'segmento'      : {'02 - PARTICULARES':0, '03 - UNIVERSITARIO':1, '01 - TOP':2, -99:2},
'pais_residencia' : {'LV': 102, 'BE': 12, 'BG': 50, 'BA': 61, 'BM': 117, 'BO': 62, 'JP': 82, 'JM': 116, 'BR': 17, 'BY': 64, 'BZ': 113, 'RU': 43, 'RS': 89, 'RO': 41, 'GW': 99, 'GT': 44, 'GR': 39, 'GQ': 73, 'GE': 78, 'GB': 9, 'GA': 45, 'GN': 98, 'GM': 110, 'GI': 96, 'GH': 88, 'OM': 100, 'HR': 67, 'HU': 106, 'HK': 34, 'HN': 22, 'AD': 35, 'PR': 40, 'PT': 26, 'PY': 51, 'PA': 60, 'PE': 20, 'PK': 84, 'PH': 91, 'PL': 30, 'EE': 52, 'EG': 74, 'ZA': 75, 'EC': 19, 'AL': 25, 'VN': 90, 'ET': 54, 'ZW': 114, 'ES': 0, 'MD': 68, 'UY': 77, 'MM': 94, 'ML': 104, 'US': 15, 'MT': 118, 'MR': 48, 'UA': 49, 'MX': 16, 'IL': 42, 'FR': 8, 'MA': 38, 'FI': 23, 'NI': 33, 'NL': 7, 'NO': 46, 'NG': 83, 'NZ': 93, 'CI': 57, 'CH': 3, 'CO': 21, 'CN': 28, 'CM': 55, 'CL': 4, 'CA': 2, 'CG': 101, 'CF': 109, 'CD': 112, 'CZ': 36, 'CR': 32, 'CU': 72, 'KE': 65, 'KH': 95, 'SV': 53, 'SK': 69, 'KR': 87, 'KW': 92, 'SN': 47, 'SL': 97, 'KZ': 111, 'SA': 56, 'SG': 66, 'SE': 24, 'DO': 11, 'DJ': 115, 'DK': 76, 'DE': 10, 'DZ': 80, 'MK': 105, -99: 1, 'LB': 81, 'TW': 29, 'TR': 70, 'TN': 85, 'LT': 103, 'LU': 59, 'TH': 79, 'TG': 86, 'LY': 108, 'AE': 37, 'VE': 14, 'IS': 107, 'IT': 18, 'AO': 71, 'AR': 13, 'AU': 63, 'AT': 6, 'IN': 31, 'IE': 5, 'QA': 58, 'MZ': 27},
'canal_entrada' : {'013': 49, 'KHP': 160, 'KHQ': 157, 'KHR': 161, 'KHS': 162, 'KHK': 10, 'KHL': 0, 'KHM': 12, 'KHN': 21, 'KHO': 13, 'KHA': 22, 'KHC': 9, 'KHD': 2, 'KHE': 1, 'KHF': 19, '025': 159, 'KAC': 57, 'KAB': 28, 'KAA': 39, 'KAG': 26, 'KAF': 23, 'KAE': 30, 'KAD': 16, 'KAK': 51, 'KAJ': 41, 'KAI': 35, 'KAH': 31, 'KAO': 94, 'KAN': 110, 'KAM': 107, 'KAL': 74, 'KAS': 70, 'KAR': 32, 'KAQ': 37, 'KAP': 46, 'KAW': 76, 'KAV': 139, 'KAU': 142, 'KAT': 5, 'KAZ': 7, 'KAY': 54, 'KBJ': 133, 'KBH': 90, 'KBN': 122, 'KBO': 64, 'KBL': 88, 'KBM': 135, 'KBB': 131, 'KBF': 102, 'KBG': 17, 'KBD': 109, 'KBE': 119, 'KBZ': 67, 'KBX': 116, 'KBY': 111, 'KBR': 101, 'KBS': 118, 'KBP': 121, 'KBQ': 62, 'KBV': 100, 'KBW': 114, 'KBU': 55, 'KCE': 86, 'KCD': 85, 'KCG': 59, 'KCF': 105, 'KCA': 73, 'KCC': 29, 'KCB': 78, 'KCM': 82, 'KCL': 53, 'KCO': 104, 'KCN': 81, 'KCI': 65, 'KCH': 84, 'KCK': 52, 'KCJ': 156, 'KCU': 115, 'KCT': 112, 'KCV': 106, 'KCQ': 154, 'KCP': 129, 'KCS': 77, 'KCR': 153, 'KCX': 120, 'RED': 8, 'KDL': 158, 'KDM': 130, 'KDN': 151, 'KDO': 60, 'KDH': 14, 'KDI': 150, 'KDD': 113, 'KDE': 47, 'KDF': 127, 'KDG': 126, 'KDA': 63, 'KDB': 117, 'KDC': 75, 'KDX': 69, 'KDY': 61, 'KDZ': 99, 'KDT': 58, 'KDU': 79, 'KDV': 91, 'KDW': 132, 'KDP': 103, 'KDQ': 80, 'KDR': 56, 'KDS': 124, 'K00': 50, 'KEO': 96, 'KEN': 137, 'KEM': 155, 'KEL': 125, 'KEK': 145, 'KEJ': 95, 'KEI': 97, 'KEH': 15, 'KEG': 136, 'KEF': 128, 'KEE': 152, 'KED': 143, 'KEC': 66, 'KEB': 123, 'KEA': 89, 'KEZ': 108, 'KEY': 93, 'KEW': 98, 'KEV': 87, 'KEU': 72, 'KES': 68, 'KEQ': 138, -99: 6, 'KFV': 48, 'KFT': 92, 'KFU': 36, 'KFR': 144, 'KFS': 38, 'KFP': 40, 'KFF': 45, 'KFG': 27, 'KFD': 25, 'KFE': 148, 'KFB': 146, 'KFC': 4, 'KFA': 3, 'KFN': 42, 'KFL': 34, 'KFM': 141, 'KFJ': 33, 'KFK': 20, 'KFH': 140, 'KFI': 134, '007': 71, '004': 83, 'KGU': 149, 'KGW': 147, 'KGV': 43, 'KGY': 44, 'KGX': 24, 'KGC': 18, 'KGN': 11}
}
cat_cols = list(mapping_dict.keys())

target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
target_cols = target_cols[2:]
num_feat = 43
#51 with all buckets
def getTarget(row):
	tlist = []
	for col in target_cols:
		if row[col].strip() in ['', 'NA']:
			target = 0
		else:
			target = int(float(row[col]))
		tlist.append(target)
	return tlist

def getIndex(row, col):
	val = row[col].strip()
	if val not in ['','NA']:
		ind = mapping_dict[col][val]
	else:
		ind = mapping_dict[col][-99]
	return ind

def getAge(row):
	mean_age = 40.
	min_age = 0.
	max_age = 100.
	range_age = max_age - min_age
	age = row['age'].strip()
	if age == 'NA' or age == '':
		age = mean_age
	else:
		age = float(age)
		if age < min_age:
			age = min_age
		elif age > max_age:
			age = 50
	return round( (age - min_age) / range_age, 4)

def getCustSeniority(row):
	min_value = 0.
	max_value = 256.
	range_value = max_value - min_value
	missing_value = 0.
	cust_seniority = row['antiguedad'].strip()
	if cust_seniority == 'NA' or cust_seniority == '':
		cust_seniority = missing_value
	else:
		cust_seniority = float(cust_seniority)
		if cust_seniority < min_value:
			cust_seniority = min_value
		elif cust_seniority > max_value:
			cust_seniority = max_value
	return round((cust_seniority-min_value) / range_value, 4)

def getRent(row):
     min_value = 0.
     max_value = 1500000.
     range_value = max_value - min_value
     missing_value = 101850.
     rent = row['renta'].strip()
     if rent == 'NA' or rent == '':
         rent=missing_value
         nomprov = row['nomprov'].strip()
         #issue with processing the Ñ character
         if nomprov=='CORUÑA, A':
             nomprov = 'CORUNA, A'
         #replace missing rent based on age and province
         if not (rent_by_age_and_nomprov['nomprov']==nomprov).any():
             nomprov = 'NA'
         if nomprov in ['NA',''] and row['age'].strip() in ['NA','']:
             rent = missing_value
         elif nomprov in ['NA','']:
             age = int(row['age'].strip())
             if age<29:
                 rent =rent_by_age_and_nomprov.loc[(rent_by_age_and_nomprov['age'] =='[0, 29)'),'renta'].mean()

             elif age<40:
                 rent =rent_by_age_and_nomprov.loc[(rent_by_age_and_nomprov['age'] =='[29, 40)'),'renta'].mean()
             else:
                 rent = rent_by_age_and_nomprov.loc[(rent_by_age_and_nomprov['age'] =='[40, inf)'),'renta'].mean()
         elif row['age'].strip() in ['NA','']:
             rent = rent_by_age_and_nomprov.loc[(rent_by_age_and_nomprov['nomprov']==nomprov) ,'renta'].mean()
         else:
             age = int(row['age'].strip())
             if age<29:
                 rent = rent_by_age_and_nomprov.loc[(rent_by_age_and_nomprov['nomprov']==nomprov) & (rent_by_age_and_nomprov['age'] =='[0, 29)'),'renta'].mean()
             elif age<40:
                 rent = rent_by_age_and_nomprov.loc[(rent_by_age_and_nomprov['nomprov']==nomprov) & (rent_by_age_and_nomprov['age'] =='[29, 40)'),'renta'].mean()
             else:
                 rent = rent_by_age_and_nomprov.loc[(rent_by_age_and_nomprov['nomprov']==nomprov) & (rent_by_age_and_nomprov['age'] =='[40, inf)'),'renta'].mean() 
     else:
         rent = float(rent)
         if rent < min_value:
             rent = min_value
         elif rent > max_value:
             rent = max_value
     return round((rent-min_value) / range_value, 6)

def getrentabuckets(renta_norm,buckets):
    j=0
    for i in range(buckets):
        if renta_norm<=((1/buckets)*(i+1)):
            break
        else:
            j+=1
    return j
            
def getsenioritybuckets(seniority_norm,buckets):
    j=0
    for i in range(buckets):
        if seniority_norm<=((1/buckets)*(i+1)):
            break
        else:
            j+=1
    return j
        
     
def processData(in_file_name, cust_dict):
     x_vars_list = []
     y_vars_list = []
     cust_dict1={}
     cust_dict2={}
     cust_dict3={}
     cust_dict4={}
     cust_dict5={}
     for row in csv.DictReader(in_file_name):
		# use only five lagging months
          if row['fecha_dato'] not in ['2015-01-28','2015-02-28','2015-03-28','2015-04-28','2015-05-28', '2015-06-28']:
              continue
          if row['fecha_dato'] in ['2015-01-28','2015-02-28','2015-03-28','2015-04-28','2015-05-28', '2015-06-28']:
              cust_id = int(row['ncodpers'])
              x_vars = []
              for col in cat_cols:
                  x_vars.append( getIndex(row, col) )
              x_vars.append( getAge(row) )
              x_vars.append( getCustSeniority(row) )
              x_vars.append( getRent(row) )
              
              #feature engineering to bucket age
              if getAge(row) <= .18:
                  x_vars.append(0)
              elif getAge(row) ==.19:
                  x_vars.append(1)
              elif getAge(row) <=.28:
                  x_vars.append(2)
              elif getAge(row) ==.37:
                  x_vars.append(3)
              elif getAge(row) ==.50:
                  x_vars.append(4)
              elif getAge(row) ==.60:
                  x_vars.append(4)
              else:
                  x_vars.append(5)

              #feature engineering to bucket income and seniority   
              #x_vars.append(getrentabuckets(getRent(row),4))
              #x_vars.append(getrentabuckets(getRent(row),6))
              #x_vars.append(getrentabuckets(getRent(row),8))
              #x_vars.append(getrentabuckets(getRent(row),10))
              #x_vars.append(getsenioritybuckets(getCustSeniority(row),4))
              #x_vars.append(getsenioritybuckets(getCustSeniority(row),6))
              #x_vars.append(getsenioritybuckets(getCustSeniority(row),8))
              #x_vars.append(getsenioritybuckets(getCustSeniority(row),10))    
                  
                  
              if row['fecha_alta']=='':
                  x_vars.append(0)
              else:
                  join_year =int(row['fecha_alta'].split("-")[0].strip())
                  current_year =int(row['fecha_dato'].split("-")[0])
                  join_month = int(row['fecha_alta'].split("-")[1].split('-')[0])
                  current_month = int(row['fecha_dato'].split("-")[1].split('-')[0])
                  account_age = 12*(current_year-join_year)+current_month-join_month
                  x_vars.append(account_age)
              
          if row['fecha_dato'] == '2015-01-28':	
              target_list = getTarget(row)
              cust_dict1[cust_id]=target_list + x_vars
              continue    
          if row['fecha_dato'] == '2015-02-28':	
              target_list = getTarget(row)
              cust_dict2[cust_id]=target_list + x_vars
              continue
          if row['fecha_dato'] == '2015-03-28':	
              target_list = getTarget(row)
              cust_dict3[cust_id]=target_list + x_vars
              continue 
          if row['fecha_dato'] == '2015-04-28':	
              target_list = getTarget(row)
              cust_dict4[cust_id]=target_list + x_vars
              continue 
          if row['fecha_dato'] == '2015-05-28':	
               target_list = getTarget(row)
               cust_dict[cust_id] =  target_list[:]
               cust_dict5[cust_id]=target_list + x_vars
               continue

         

          if row['fecha_dato'] == '2015-06-28':
              prev_target_list = cust_dict.get(cust_id, [0]*22)
              target_list = getTarget(row)
              new_products = [max(x1 - x2,0) for (x1, x2) in zip(target_list, prev_target_list)]
              if sum(new_products) > 0:
                  for ind, prod in enumerate(new_products):
                      if prod>0:
                          assert len(prev_target_list) == 22
                          x_temp = []
                          x_temp.extend(x_vars + cust_dict5.get(cust_id,[-99]*num_feat) + cust_dict4.get(cust_id,[-99]*num_feat) +
                                             cust_dict3.get(cust_id,[-99]*num_feat) + cust_dict2.get(cust_id,[-99]*num_feat) + cust_dict1.get(cust_id,[-99]*num_feat))
                         # x_temp.extend(x_vars + cust_dict5.get(cust_id,[-99]*18) + [-99]*40 +[-99]*40+[-99]*40+[-99]*40)
                          x_vars_list.append(x_temp)
                          y_vars_list.append(ind)

     return x_vars_list, y_vars_list, cust_dict

def processData_test(in_file_name, cust_dict1,cust_dict2,cust_dict3,cust_dict4,cust_dict5):
     x_vars_list = []

     for row in csv.DictReader(in_file_name):
		
         
           if row['fecha_dato'] == '2016-06-28':
              cust_id = int(row['ncodpers'])
              x_vars=[]
              for col in cat_cols:
                  x_vars.append( getIndex(row, col) )
              x_vars.append( getAge(row) )
              x_vars.append( getCustSeniority(row) )
              x_vars.append( getRent(row) )
              #feature engineering to bucket age
              if getAge(row) <= .18:
                  x_vars.append(0)
              elif getAge(row) ==.19:
                  x_vars.append(1)
              elif getAge(row) <=.28:
                  x_vars.append(2)
              elif getAge(row) ==.37:
                  x_vars.append(3)
              elif getAge(row) ==.50:
                  x_vars.append(4)
              elif getAge(row) ==.60:
                  x_vars.append(4)
              else:
                  x_vars.append(5)
            
              #feature engineering to bucket income and seniority
              #x_vars.append(getrentabuckets(getRent(row),4))
              #x_vars.append(getrentabuckets(getRent(row),6))
              #x_vars.append(getrentabuckets(getRent(row),8))
              #x_vars.append(getrentabuckets(getRent(row),10))
              #x_vars.append(getsenioritybuckets(getCustSeniority(row),4))
              #x_vars.append(getsenioritybuckets(getCustSeniority(row),6))
              #x_vars.append(getsenioritybuckets(getCustSeniority(row),8))
              #x_vars.append(getsenioritybuckets(getCustSeniority(row),10))
              
              #changing date joined into the age of the account in months
              if(row['fecha_alta'])=="":
                  x_vars.append(0)
              else:
                  join_year =int(row['fecha_alta'].split("-")[0].strip())
                  current_year =int(row['fecha_dato'].split("-")[0])
                  join_month = int(row['fecha_alta'].split("-")[1].split('-')[0])
                  current_month = int(row['fecha_dato'].split("-")[1].split('-')[0])
                  account_age = 12*(current_year-join_year)+current_month-join_month
                  x_vars.append(account_age)
             
              x_temp =[]
              x_temp.extend(x_vars + cust_dict5.get(cust_id,[-99]*num_feat) + cust_dict4.get(cust_id,[-99]*num_feat) +
                                 cust_dict3.get(cust_id,[-99]*num_feat) + cust_dict2.get(cust_id,[-99]*num_feat) + cust_dict1.get(cust_id,[-99]*num_feat))
              x_vars_list.append(x_temp)
              

     return x_vars_list
 
def processData_test_lag(in_file_name):
     cust_dict1={}
     cust_dict2={}
     cust_dict3={}
     cust_dict4={}
     cust_dict5={}
     for row in csv.DictReader(in_file_name):
		# use five lagging months
          
          if row['fecha_dato'] not in ['2016-01-28','2016-02-28','2016-03-28','2016-04-28','2016-05-28', '2016-06-28']:
              continue
          if row['fecha_dato'] in ['2016-01-28','2016-02-28','2016-03-28','2016-04-28','2016-05-28', '2016-06-28']:
              cust_id = int(row['ncodpers'])
              x_vars = []
              for col in cat_cols:
                  x_vars.append( getIndex(row, col) )
              x_vars.append( getAge(row) )
              x_vars.append( getCustSeniority(row) )
              x_vars.append( getRent(row) )
              
              #feature engineering to bucket age
              if getAge(row) <= .18:
                  x_vars.append(0)
              elif getAge(row) ==.19:
                  x_vars.append(1)
              elif getAge(row) <=.28:
                  x_vars.append(2)
              elif getAge(row) ==.37:
                  x_vars.append(3)
              elif getAge(row) ==.50:
                  x_vars.append(4)
              elif getAge(row) ==.60:
                  x_vars.append(4)
              else:
                  x_vars.append(5)
          
              #feature engineering to bucket income and seniority    
              #x_vars.append(getrentabuckets(getRent(row),4))
              #x_vars.append(getrentabuckets(getRent(row),6))
              #x_vars.append(getrentabuckets(getRent(row),8))
              #x_vars.append(getrentabuckets(getRent(row),10))
              #x_vars.append(getsenioritybuckets(getCustSeniority(row),4))
              #x_vars.append(getsenioritybuckets(getCustSeniority(row),6))
              #x_vars.append(getsenioritybuckets(getCustSeniority(row),8))
              #x_vars.append(getsenioritybuckets(getCustSeniority(row),10))
              
              #changing date joined into the age of the account in months
              if(row['fecha_alta'])=="":
                  x_vars.append(0)
              else:
                  join_year =int(row['fecha_alta'].split("-")[0].strip())
                  current_year =int(row['fecha_dato'].split("-")[0])
                  join_month = int(row['fecha_alta'].split("-")[1].split('-')[0])
                  current_month = int(row['fecha_dato'].split("-")[1].split('-')[0])
                  account_age = 12*(current_year-join_year)+current_month-join_month
                  x_vars.append(account_age)
                  
          if row['fecha_dato'] == '2016-01-28':	
              target_list = getTarget(row)
              cust_dict1[cust_id]=target_list + x_vars
              continue    
          if row['fecha_dato'] == '2016-02-28':	
              target_list = getTarget(row)
              cust_dict2[cust_id]=target_list + x_vars
              continue
          if row['fecha_dato'] == '2016-03-28':	
              target_list = getTarget(row)
              cust_dict3[cust_id]=target_list + x_vars
              continue 
          if row['fecha_dato'] == '2016-04-28':	
              target_list = getTarget(row)
              cust_dict4[cust_id]=target_list + x_vars
              continue 
          if row['fecha_dato'] == '2016-05-28':	
               target_list = getTarget(row)
               cust_dict5[cust_id]=target_list + x_vars
               continue

     return cust_dict1,cust_dict2,cust_dict3,cust_dict4,cust_dict5

			


if __name__ == "__main__":
     start_time = datetime.datetime.now()
     data_path = "../input/"
     train_file =  open(data_path + "train_ver2.csv")
     rent_by_age_and_nomprov = pd.read_csv("rent_by_age_and_nomprov.csv")
     
     
     rent2 =rent_by_age_and_nomprov.loc[(rent_by_age_and_nomprov['age'] =='[29, 40)'),'renta'].mean()  

     #replace missing incomes with broader averages
     rent_by_age_and_nomprov = rent_by_age_and_nomprov.set_value(61,'renta',rent2)
     rent_by_age_and_nomprov = rent_by_age_and_nomprov.set_value(77,'renta',rent2)
  
     a =rent_by_age_and_nomprov['nomprov']
     print("Processed rent by age and nomprov table: ",datetime.datetime.now()-start_time)
     
     x_vars_list, y_vars_list, cust_dict = processData(train_file, {})
     
     train_X = np.array(x_vars_list,dtype=float)
     train_y = np.array(y_vars_list)

     del x_vars_list, y_vars_list
     train_file.close()
     
     
     print("train_X and train_y shape:",train_X.shape, train_y.shape)
     print("Train_X and train_y generated: ",datetime.datetime.now()-start_time)
     
     train_file =  open(data_path + "train_ver2.csv")
     cust_dict1,cust_dict2,cust_dict3,cust_dict4,cust_dict5 = processData_test_lag(train_file)
     train_file.close()
     
     
     test_file = open(data_path + "test_ver2.csv")
     x_vars_list = processData_test(test_file,cust_dict1,cust_dict2,cust_dict3,cust_dict4,cust_dict5)
     
     
     print("test_X generated: ",datetime.datetime.now()-start_time)
     del cust_dict1,cust_dict2,cust_dict3,cust_dict4,cust_dict5
     

     
     test_X = np.array(x_vars_list,dtype=float)
     
     np.savetxt("../input/test_X.csv", test_X, delimiter=",")
     np.savetxt("../input/train_X.csv", train_X, delimiter=",")
     np.savetxt("../input/train_y.csv", train_y, delimiter=",")
     

     print("train and test feature creation completed: ",datetime.datetime.now()-start_time)  