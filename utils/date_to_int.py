from datetime import datetime

def date_to_int(date_str, base_str="2023-01-01"):
    date_format = "%Y-%m-%d"
    date = datetime.strptime(date_str, date_format)
    base_date = datetime.strptime(base_str, date_format)
    
    delta = (date - base_date).days
    return delta
