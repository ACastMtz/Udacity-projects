import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = 'http://08dc83d0-246a-4281-a39b-82285f763de5.southcentralus.azurecontainer.io/score'
# If the service is authenticated, set the key or token
key = 'Qa3dicRCEK5IRa91sY3pMXm5BAdR210X'

# Two sets of data to score, so we get two results back
data = {'data': [{'loan_amnt': 24375.0, 'term': 60, 'int_rate': 17.27, 'installment': 609.33, 'annual_inc': 55000.0, 'dti': 33.95, 'earliest_cr_line': 1999, 'open_acc': 13.0, 'pub_rec': 0.0, 'revol_bal': 24584.0, 'revol_util': 69.8, 'total_acc': 43.0, 'mort_acc': 1.0, 'pub_rec_bankruptcies': 0.0, 'zip_code': '11650', 'sub_grade_A2': 0, 'sub_grade_A3': 0, 'sub_grade_A4': 0, 'sub_grade_A5': 0, 'sub_grade_B1': 0, 'sub_grade_B2': 0, 'sub_grade_B3': 0, 'sub_grade_B4': 0, 'sub_grade_B5': 0, 'sub_grade_C1': 0, 'sub_grade_C2': 0, 'sub_grade_C3': 0, 'sub_grade_C4': 0, 'sub_grade_C5': 1, 'sub_grade_D1': 0, 'sub_grade_D2': 0, 'sub_grade_D3': 0, 'sub_grade_D4': 0, 'sub_grade_D5': 0, 'sub_grade_E1': 0, 'sub_grade_E2': 0, 'sub_grade_E3': 0, 'sub_grade_E4': 0, 'sub_grade_E5': 0, 'sub_grade_F1': 0, 'sub_grade_F2': 0, 'sub_grade_F3': 0, 'sub_grade_F4': 0, 'sub_grade_F5': 0, 'sub_grade_G1': 0, 'sub_grade_G2': 0, 'sub_grade_G3': 0, 'sub_grade_G4': 0, 'sub_grade_G5': 0, 'verification_status_Source Verified': 0, 'verification_status_Verified': 1, 'application_type_INDIVIDUAL': 1, 'application_type_JOINT': 0, 'initial_list_status_w': 0, 'home_ownership_OTHER': 0, 'home_ownership_OWN': 0, 'home_ownership_RENT': 0, 'purpose_credit_card': 1, 'purpose_debt_consolidation': 0, 'purpose_educational': 0, 'purpose_home_improvement': 0, 'purpose_house': 0, 'purpose_major_purchase': 0, 'purpose_medical': 0, 'purpose_moving': 0, 'purpose_other': 0, 'purpose_renewable_energy': 0, 'purpose_small_business': 0, 'purpose_vacation': 0, 'purpose_wedding': 0}, {'loan_amnt': 8000.0, 'term': 36, 'int_rate': 15.31, 'installment': 278.54, 'annual_inc': 22050.0, 'dti': 25.75, 'earliest_cr_line': 2005, 'open_acc': 8.0, 'pub_rec': 0.0, 'revol_bal': 4030.0, 'revol_util': 56.0, 'total_acc': 15.0, 'mort_acc': 0.0, 'pub_rec_bankruptcies': 0.0, 'zip_code': '30723', 'sub_grade_A2': 0, 'sub_grade_A3': 0, 'sub_grade_A4': 0, 'sub_grade_A5': 0, 'sub_grade_B1': 0, 'sub_grade_B2': 0, 'sub_grade_B3': 0, 'sub_grade_B4': 0, 'sub_grade_B5': 0, 'sub_grade_C1': 0, 'sub_grade_C2': 0, 'sub_grade_C3': 0, 'sub_grade_C4': 1, 'sub_grade_C5': 0, 'sub_grade_D1': 0, 'sub_grade_D2': 0, 'sub_grade_D3': 0, 'sub_grade_D4': 0, 'sub_grade_D5': 0, 'sub_grade_E1': 0, 'sub_grade_E2': 0, 'sub_grade_E3': 0, 'sub_grade_E4': 0, 'sub_grade_E5': 0, 'sub_grade_F1': 0, 'sub_grade_F2': 0, 'sub_grade_F3': 0, 'sub_grade_F4': 0, 'sub_grade_F5': 0, 'sub_grade_G1': 0, 'sub_grade_G2': 0, 'sub_grade_G3': 0, 'sub_grade_G4': 0, 'sub_grade_G5': 0, 'verification_status_Source Verified': 0, 'verification_status_Verified': 1, 'application_type_INDIVIDUAL': 1, 'application_type_JOINT': 0, 'initial_list_status_w': 1, 'home_ownership_OTHER': 0, 'home_ownership_OWN': 0, 'home_ownership_RENT': 1, 'purpose_credit_card': 0, 'purpose_debt_consolidation': 1, 'purpose_educational': 0, 'purpose_home_improvement': 0, 'purpose_house': 0, 'purpose_major_purchase': 0, 'purpose_medical': 0, 'purpose_moving': 0, 'purpose_other': 0, 'purpose_renewable_energy': 0, 'purpose_small_business': 0, 'purpose_vacation': 0, 'purpose_wedding': 0}]}
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print('If result is 1, loan has been repaid')
print(resp.json())


