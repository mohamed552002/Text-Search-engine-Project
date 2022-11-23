positional_dict={"good":[{}]}
positional_dict["good"][0].update({"file0":[]})
print(positional_dict["good"][0])
lst=[5,6,8,9]
positional_dict["good"][0]["file0"].append(5)
print(positional_dict["good"][0]["file0"])
positional_dict["good"][0].update({"file0":[]})
print(positional_dict["good"][0]["file0"])