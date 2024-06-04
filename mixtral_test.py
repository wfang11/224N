from models.model import Model

mixtral = Model("MIXTRAL")

prompt = "what are some alliteration names for someone who likes pop? for instance, pop princess. good ones only"

print(mixtral.query(prompt))