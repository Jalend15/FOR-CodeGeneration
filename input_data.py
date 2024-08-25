import os
import json

def get_data(train=True):
    path = f'abstraction-and-reasoning-challenge/{"training" if train else "evaluation"}'
    data = {}
    for fn in os.listdir(path):
        with open(f'{path}/{fn}') as f:
            data[fn.rstrip('.json')] = json.load(f)
    def ast(g):
        return tuple(tuple(r) for r in g)
    return {
        'train': {k: [{
            'input': ast(e['input']),
            'output': ast(e['output']),
        } for e in v['train']] for k, v in data.items()},
        'test': {k: [{
            'input': ast(e['input']),
            'output': ast(e['output']),
        } for e in v['test']] for k, v in data.items()}
    }

def main():
    data = get_data(train=True)
    print(type(data['train']))
    print(data['train']['00d62c1b'][0])


if __name__ == '__main__':
    main()