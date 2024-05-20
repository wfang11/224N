from models.model import Model

def __main__(): 
    gemini = Model("GEMINI")
    print(gemini.query())

if __name__== "__main__": 
    __main__()