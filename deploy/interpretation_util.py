def get_density_class(count):
    if count == 0:
        return 0
    elif count <= 1:
        return 1
    elif count <= 5:
        return 2
    elif count <= 30:
        return 3
    else:
        return 4

def get_nugent_score(lactobacillus_count, gardnerella_bacteroides_count, curved_rod_count):
    total_score = 0
    lactobacillus_score = 4 - get_density_class(lactobacillus_count)
    gardnerella_score = get_density_class(gardnerella_bacteroides_count)
    curved_rod_score = int( (get_density_class(curved_rod_count) + 1) / 2)
    return lactobacillus_score + gardnerella_score + curved_rod_score, lactobacillus_score, gardnerella_score, curved_rod_score
    
def get_nugent_score_interpretation_int(nugent_score):
    if nugent_score <= 3:
        return 0
    elif nugent_score <= 6:
        return 1
    else:
        return 2

def get_nugent_score_interpretation_int_str_map(severity):
    return ["Normal", "Intermediate","Infection"][severity]

def get_nugent_score_interpretation_str(nugent_score):
    if nugent_score <= 3:
        return "Normal"
    elif nugent_score <= 6:
        return "Intermediate"
    else:
        return "Infection"