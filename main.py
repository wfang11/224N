from models.model import Model

def __main__(): 
    llama = Model("LLAMA")
    print(llama.query())


if __name__== "__main__": 
    __main__()