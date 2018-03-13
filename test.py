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
    c='a'
    b='b'

    import logging

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    logging.info('{} and {}'.format(c,b))

    import numpy as np
    a = np.array([[1,2,3],[1,2,3],[1,2,5]])
    print(np.array(list(set(tuple(t) for t in a))))

