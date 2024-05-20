from models.model import Model
from data.cai import CAI

def __main__(): 
    for name in "LLAMA", "GPT-4", "GEMINI": 
        model = Model(name)
        print(model.query())

if __name__ == "__main__": 
    __main__()