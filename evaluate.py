from models.model import Model
from data.cai import CAI

def __main__(): 
    model = Model("GPT-4")
    print(model.query())
    # cai = CAI()
    # data = cai.get_data()
    ### TODO: set up a pipeline for running inference and then evaluating 

if __name__ == "__main__": 
    __main__()