label_mapping = {
    "W" : 0,
    "S" : 1,
    "L" : 2,
    "I" : 3,
    "E" : 4
}

inv_label_mapping = {}
for key in label_mapping.keys():
    inv_label_mapping[label_mapping[key]] = key

def label2int(label: str, mapping=label_mapping):
    return mapping[label.strip()]

def int2label(label: int, mapping=inv_label_mapping):
    return mapping[label.strip()]
