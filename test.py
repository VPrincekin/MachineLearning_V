class a:
    def __init__(self,name,core):
        self.name = name
        self.core = core

    def print_name(self,name):
        print(name)


    def run(self):
        self.print_name(self.name)

if __name__ == '__main__':

    w = a('wdh',99)

    w.run()



