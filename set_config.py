import yaml
import sys

f = open("config.yaml")
y = yaml.safe_load(f)
y["train_args"]["dataset_name"] = sys.argv[1]
y["model_args"]["model_name"] = float(sys.argv[2])
y["model_args"]["num_train_epochs"] = int(sys.argv[3])
y["model_args"]["batch_size"] = int(sys.argv[4])
y["model_args"]["learning_rate"] = float(sys.argv[5])
f = open("config.yaml", "w")
yaml.dump(y, f)
f.close()