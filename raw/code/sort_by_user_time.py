#!/usr/bin/env python3

import sys

if __name__ == '__main__':
    data = [ x.strip().split(',') for x in open(sys.argv[1], 'r').readlines() if x[0] != 'I' ]

    data = sorted(data, key=lambda x: (x[6], x[7], x[8], int(x[2])))

    with open(sys.argv[1] + '.sort', 'w') as fout:
        for x in data:
            print(','.join(x), file=fout)

