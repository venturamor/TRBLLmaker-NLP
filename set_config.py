import yaml
import sys

f = open("config.yaml")
y = yaml.safe_load(f)
for i in range(1, len(sys.argv)):
    var, val = sys.argv[i].split("=")
    if var.find("_int") != -1:
        var = var.split('_int')[0]
        y["train_args"][var] = int(val)
    elif var.find("_float") != -1:
        var = var.split('_float')[0]
        y["train_args"][var] = float(val)
    else:
        y["train_args"][var] = val
f = open("config.yaml", "w")
yaml.dump(y, f)
f.close()
