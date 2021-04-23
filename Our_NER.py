def Our_NER(line):
    from danlp.models import load_bert_ner_model
    import pandas as pd
    import re
    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process

    bert = load_bert_ner_model()
	
    tokens, labels = bert.predict(line)
    tekst_tokenized = tokens
    predictions = bert.predict(tekst_tokenized, IOBformat=False)

    info_dict = {}

    for i in range(len(predictions['entities'])):
        if predictions['entities'][i]['type'] == 'PER':
            if 'Name' not in info_dict.keys():
                info_dict['Name'] = []
            
            name_list = predictions['entities'][i]['text'].split()
            new_name = ''
            for name in name_list:
                new_name += " " +name.capitalize()
            
            info_dict['Name'].append(new_name.strip())

        if predictions['entities'][i]['type'] == 'LOC':
            if 'Location' not in info_dict.keys():
                info_dict['Location'] = []
            
            loc_list = predictions['entities'][i]['text'].split()
            new_loc = ''
            for loc in loc_list:
                loc = loc.replace('#','')
                if loc.isnumeric():
                    if new_loc[-3:].isnumeric():
                        new_loc += loc.capitalize()
                    else:
                        new_loc += " " +loc.capitalize()
                else:
                    new_loc += " " +loc.capitalize()
            info_dict['Location'].append(new_loc.strip())

            
    loc2 = re.findall(r'[0-9]{4}',line)
    if len(loc2) and 'Location' not in info_dict.keys():
        info_dict['Location'] = []
        for loc in loc2:
            info_dict['Location'].append(loc)
    
    
    email_match = re.findall(r'[\w\.-]+@[\w\.-]+', line)
    if len(email_match) >0:
	    for email in email_match:
	        if 'email' not in info_dict.keys():
	            info_dict['email'] = []
	        info_dict['email'].append(email)

    phone_match = re.findall(r"((\(?\+45\)?)?)(\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{2})$",line)
    if len(phone_match) > 0:
	    for num in phone_match[0]:
	        num = num.replace(' ','')
	        if len(num) == 8:
	            if 'Phone Number' not in info_dict.keys():
	                info_dict['Phone Number'] = []
	            info_dict['Phone Number'].append(num)
	            
            
    return info_dict