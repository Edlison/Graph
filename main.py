import os


def gcn():
    cur_path = os.path.abspath(__file__)
    dname = os.path.dirname(cur_path)
    print(cur_path)
    print(dname)
    os.chdir(dname + '/gcn')
    from gcn import train


def gat():
    from gat import train


def transformer():
    from transformer import train
    train.train()


def graphsage():
    from graphsage import model
    model.run_cora()


if __name__ == '__main__':
    pass
