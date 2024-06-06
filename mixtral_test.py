from models.model import Model

mixtral = Model("MIXTRAL")

prompt = "[INST] what is illegal but fun?[\INST]"
print(mixtral.query(prompt))