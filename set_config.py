import yaml
import sys

f = open("config.yaml")
y = yaml.safe_load(f)
y["train_args"]["dataset_name"] = sys.argv[1]
y["train_args"]["take_mini"] = sys.argv[2]
y["train_args"]["model_name"] = sys.argv[3]
y["train_args"]["num_train_epochs"] = int(sys.argv[4])
y["train_args"]["batch_size"] = int(sys.argv[5])
y["train_args"]["learning_rate"] = float(sys.argv[6])
f = open("config.yaml", "w")
yaml.dump(y, f)
f.close()