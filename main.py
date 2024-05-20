from models.model import Model
from data.dataloader import load_cai, load_beavertails

def __main__(): 
    gemini = Model("GEMINI")
    print(gemini.query())
    # cai = load_cai()
    # print(cai['sft_train'][0])

if __name__== "__main__": 
    __main__()