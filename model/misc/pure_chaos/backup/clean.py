import pandas as pd
main_data_formatted = pd.read_csv('main_data_formatted_cond_3_qual_resp.csv')
main_data_formatted = main_data_formatted.query("rule_name == 'Zeta' or rule_name == 'Upsilon' or rule_name == 'Iota' or rule_name == 'Kappa' or rule_name == 'Omega'")
main_data_formatted = main_data_formatted.reset_index(drop=True)

main_data_formatted.to_csv('main_data_formatted_cond_3_qual_resp_clean.csv')
