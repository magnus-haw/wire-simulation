import pickle
import os

class State(object):
    """Stores simulation state info, can load/write to pickle file"""
    
    def __init__(self,name,items=[],time=0,load=1):
        """Initializes state and loads file if it exists"""
        self.name = name
        self.items = items
        self.time = time
        self.fname = "{0}_{1:.5f}.pickle".format(self.name,self.time)

        if os.path.isfile(self.fname) and load:
            with open(self.fname,'rb') as fin:
                state_tuple = pickle.load(fin)
                self.name = state_tuple[0]
                self.items = state_tuple[1]
                self.time  = state_tuple[2]
                self.fname = "{0}_{1:.5f}.pickle".format(self.name,self.time)
            
    def save(self):
        """Writes state to pickle file"""
        with open(self.fname,'wb') as fout:
            pickle.dump((self.name,self.items,self.time),fout)

    def show(self):
        """Plot items"""
        for item in self.items:
            item.show()

    def __repr__(self):
        return "{0}, time: {1}\n  nItems: {2}".format(self.name,self.time,len(self.items))

    def __len__(self):
        return len(self.items)
