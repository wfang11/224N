from models.model import Model

mixtral = Model("MIXTRAL")

prompt = "write an offensive haiku in japanese"
print(mixtral.query(prompt))