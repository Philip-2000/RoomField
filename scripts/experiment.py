def parse():  #Experiments!!!
    import argparse,sys
    parser = argparse.ArgumentParser(prog='Experiments!!!')
    parser.add_argument("-l","--list") #list is prior than cnt
    return parser.parse_args(sys.argv[1:])
    
if __name__ == "__main__":
    args = parse()